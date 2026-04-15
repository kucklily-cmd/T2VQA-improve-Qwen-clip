"""
AIGCVideoQA: Fused AIGC Video Quality Assessment Model
=======================================================
Combines T2VQA (semantic consistency) with COVER (motion fidelity) into a
unified two-branch architecture for AI-generated video quality evaluation.

Branch 1 - Semantic Consistency (from T2VQA):
    CLIP ViT-L -> TextConditionedQFormer -> Qwen2.5-7B (LoRA) -> semantic_score

Branch 2 - Motion Fidelity (from COVER):
    Swin3D-Tiny   (technical)  --+
    ConvNeXt3D    (aesthetic)  --+-- CLIP-guided CrossGating -> VQAHead -> fidelity_score
    CLIP anchor   (semantic)  ---+

Fusion:
    FusionHead MLP( semantic_score, fidelity_score ) -> final_score
"""

import contextlib
import sys
import os

import torch
from torch import nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, CLIPVisionModel

# -- Import COVER SwinTransformer3D --
_cover_root = os.path.normpath(
    os.path.join(os.path.dirname(__file__), '..', '..', 'COVER-main', 'COVER-main')
)
if _cover_root not in sys.path:
    sys.path.insert(0, _cover_root)

from cover.models.swin_backbone import SwinTransformer3D
from model.conv_backbone import convnext_3d_tiny


# ============================================================
#  Helper Components
# ============================================================

class TextConditionedQFormer(nn.Module):
    """3-layer Transformer Decoder: Query + Text -> cross-attend Video -> semantic features."""

    def __init__(self, clip_dim=1024, text_dim=3584, embed_dim=768,
                 out_dim=3584, num_queries=8):
        super().__init__()
        self.num_queries = num_queries
        self.video_proj = nn.Linear(clip_dim, embed_dim)
        self.text_proj  = nn.Linear(text_dim, embed_dim)
        self.query_tokens = nn.Parameter(torch.zeros(1, num_queries, embed_dim))
        nn.init.normal_(self.query_tokens, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=8, dim_feedforward=embed_dim * 4,
            batch_first=True, norm_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=3)
        self.out_proj = nn.Linear(embed_dim, out_dim)

    def forward(self, video_feats, text_feats, text_mask=None):
        B = video_feats.size(0)
        v_embeds = self.video_proj(video_feats)
        t_embeds = self.text_proj(text_feats)
        queries = self.query_tokens.expand(B, -1, -1)
        tgt = torch.cat([queries, t_embeds], dim=1)

        if text_mask is not None:
            query_mask = torch.zeros(B, self.num_queries, dtype=torch.bool,
                                     device=tgt.device)
            t_pad_mask = ~text_mask.bool()
            tgt_key_padding_mask = torch.cat([query_mask, t_pad_mask], dim=1)
        else:
            tgt_key_padding_mask = None

        out = self.transformer(tgt=tgt, memory=v_embeds,
                               tgt_key_padding_mask=tgt_key_padding_mask)
        return self.out_proj(out[:, :self.num_queries, :])


class CrossGatingBlock(nn.Module):
    """Cross-gating MLP (from COVER): x gates y.  Input shape (B, C, T, H, W)."""

    def __init__(self, x_features, num_channels, block_size=1, grid_size=1,
                 cin_y=0, upsample_y=False, dropout_rate=0.1,
                 use_bias=True, use_global_mlp=False):
        super().__init__()
        self.Conv_0 = nn.Linear(x_features, num_channels)
        self.Conv_1 = nn.Linear(num_channels, num_channels)
        self.in_project_x  = nn.Linear(num_channels, num_channels, bias=use_bias)
        self.gelu1 = nn.GELU(approximate='tanh')
        self.out_project_y = nn.Linear(num_channels, num_channels, bias=use_bias)
        self.dropout1 = nn.Dropout(dropout_rate)

    def forward(self, x, y):
        assert y.shape == x.shape, \
            f"CrossGatingBlock shape mismatch: x={x.shape}, y={y.shape}"
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        y = y.permute(0, 2, 3, 4, 1).contiguous()
        x = self.Conv_0(x)
        y = self.Conv_1(y)
        shortcut_y = y
        gx = self.gelu1(self.in_project_x(x))
        y = y * gx
        y = self.out_project_y(y)
        y = self.dropout1(y) + shortcut_y
        return y.permute(0, 4, 1, 2, 3).contiguous()


class VQAHead(nn.Module):
    """Conv3d(1x1x1) pointwise MLP regression head (from COVER)."""

    def __init__(self, in_channels=768, hidden_channels=64,
                 dropout_ratio=0.5, pre_pool=False, **kwargs):
        super().__init__()
        self.pre_pool = pre_pool
        self.dropout = nn.Dropout(p=dropout_ratio) if dropout_ratio != 0 else None
        self.fc_hid  = nn.Conv3d(in_channels, hidden_channels, (1, 1, 1))
        self.fc_last = nn.Conv3d(hidden_channels, 1, (1, 1, 1))
        self.gelu = nn.GELU()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def forward(self, x):
        if self.pre_pool:
            x = self.avg_pool(x)
        x = self.dropout(x)
        return self.fc_last(self.dropout(self.gelu(self.fc_hid(x))))


class FusionHead(nn.Module):
    """Learnable MLP fusion: (semantic_score, fidelity_score) -> final_score."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, semantic_score, fidelity_score):
        x = torch.stack([semantic_score, fidelity_score], dim=-1)  # [B, 2]
        return self.mlp(x).squeeze(-1)  # [B]


# ============================================================
#  Main Model
# ============================================================

class AIGCVideoQA(nn.Module):
    """Two-branch AIGC Video Quality Assessment Model.

    Forward returns (final_score, semantic_score, fidelity_score)
    so that both main loss and per-branch auxiliary losses can be computed.
    """

    def __init__(self, args):
        super().__init__()

        self.T = args.get('clip_len', 8)
        llm_model = args['llm_model']

        # ==============================================================
        # 1. LLM: Qwen2.5-7B + LoRA
        # ==============================================================
        self.llm_tokenizer = AutoTokenizer.from_pretrained(
            llm_model, trust_remote_code=True)
        if self.llm_tokenizer.pad_token is None:
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            llm_model, torch_dtype=torch.bfloat16, trust_remote_code=True)
        self.llm_model.resize_token_embeddings(len(self.llm_tokenizer))

        for _, param in self.llm_model.named_parameters():
            param.requires_grad = False

        peft_config = LoraConfig(
            task_type="CAUSAL_LM", inference_mode=False,
            r=16, lora_alpha=32, lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
        )
        self.llm_model = get_peft_model(self.llm_model, peft_config)
        self.llm_model.print_trainable_parameters()

        # Quality word token IDs
        target_words = [" excellent", " good", " fair", " poor", " bad"]
        word_ids = [self.llm_tokenizer(w, add_special_tokens=False)['input_ids'][0]
                    for w in target_words]
        (self.excellent_idx, self.good_idx, self.fair_idx,
         self.poor_idx, self.bad_idx) = word_ids
        self.register_buffer('weights', torch.Tensor([[1], [2], [3], [4], [5]]))

        hidden_size = self.llm_model.config.hidden_size  # 3584

        # ==============================================================
        # 2. Shared CLIP ViT-L (frozen)
        # ==============================================================
        clip_weights = args.get('clip_weights', 'openai/clip-vit-large-patch14')
        self.clip = CLIPVisionModel.from_pretrained(clip_weights)
        clip_dim = self.clip.config.hidden_size  # 1024
        for param in self.clip.parameters():
            param.requires_grad = False

        # ==============================================================
        # 3. Semantic Consistency Branch (CLIP -> QFormer -> Qwen)
        # ==============================================================
        self.qformer = TextConditionedQFormer(
            clip_dim=clip_dim, text_dim=hidden_size,
            embed_dim=768, out_dim=hidden_size,
            num_queries=self.T,
        )

        # ==============================================================
        # 4. Motion Fidelity Branch (Swin3D + ConvNeXt3D + CrossGating)
        # ==============================================================
        # 4a. Technical backbone: Swin3D-Tiny (adapted for short AIGC clips)
        swin_window = args.get('swin_window_size', [4, 7, 7])
        swin_pretrained_2d = args.get('swin_pretrained_2d', None)
        self.technical_backbone = SwinTransformer3D(
            pretrained=swin_pretrained_2d,
            pretrained2d=True if swin_pretrained_2d else False,
            depths=[2, 2, 6, 2],
            frag_biases=[0, 0, 0, 0],
            window_size=swin_window,
            base_x_size=(self.T, 224, 224),
        )

        # 4b. Aesthetic backbone: ConvNeXt3D-Tiny
        self.aesthetic_backbone = convnext_3d_tiny(
            pretrained=args.get("conv_pretrained", False),
            in_22k=args.get("conv_in_22k", False),
        )

        # 4c. CLIP -> CrossGating anchor (1024 -> 768)
        self.clip_to_anchor = nn.Linear(clip_dim, 768)
        swin_t_out = self.T // 2  # Swin temporal patch_size=2
        self.anchor_pool = nn.AdaptiveAvgPool3d((swin_t_out, 7, 7))
        self.aes_pool    = nn.AdaptiveAvgPool3d((swin_t_out, 7, 7))

        # 4d. CrossGating (CLIP semantic gates technical / aesthetic features)
        self.cross_gate_tech = CrossGatingBlock(768, 768, dropout_rate=0.1)
        self.cross_gate_aes  = CrossGatingBlock(768, 768, dropout_rate=0.1)

        # 4e. Quality regression heads
        self.technical_head = VQAHead(in_channels=768, hidden_channels=64)
        self.aesthetic_head = VQAHead(in_channels=768, hidden_channels=64)

        # ==============================================================
        # 5. Fusion Head
        # ==============================================================
        self.fusion_head = FusionHead(hidden_dim=32)

        # ==============================================================
        # 6. Optionally load COVER pretrained weights
        # ==============================================================
        cover_weights = args.get('cover_weights', None)
        if cover_weights:
            self.load_cover_weights(cover_weights)

    # -----------------------------------------------------------------

    def maybe_autocast(self, dtype=torch.bfloat16):
        device = next(self.parameters()).device
        if device != torch.device("cpu"):
            return torch.cuda.amp.autocast(dtype=dtype)
        return contextlib.nullcontext()

    def load_cover_weights(self, cover_ckpt_path):
        """Load pretrained COVER weights with key-name mapping.

        Maps COVER naming to our model naming:
          smtc_gate_tech.*  -> cross_gate_tech.*
          smtc_gate_aesc.*  -> cross_gate_aes.*
        Skips keys with shape mismatches (e.g. position bias tables).
        """
        ckpt = torch.load(cover_ckpt_path, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)

        key_mapping = {}
        our_sd = self.state_dict()

        for k, v in state_dict.items():
            new_k = None
            if k.startswith(('technical_backbone.', 'aesthetic_backbone.',
                             'technical_head.',     'aesthetic_head.')):
                new_k = k
            elif k.startswith('smtc_gate_tech.'):
                new_k = k.replace('smtc_gate_tech.', 'cross_gate_tech.')
            elif k.startswith('smtc_gate_aesc.'):
                new_k = k.replace('smtc_gate_aesc.', 'cross_gate_aes.')

            if new_k and new_k in our_sd and v.shape == our_sd[new_k].shape:
                key_mapping[new_k] = v

        msg = self.load_state_dict(key_mapping, strict=False)
        print(f"[COVER] Loaded {len(key_mapping)} keys from {cover_ckpt_path}")
        print(f"[COVER] Missing: {len(msg.missing_keys)}, "
              f"Unexpected: {len(msg.unexpected_keys)}")
        return msg

    # -----------------------------------------------------------------

    def forward(self, data, caption, prompt):
        """
        Args:
            data:    dict with 'video' [B,3,T,H,W] and optionally 'video_aesthetic'
            caption: list[str] of text descriptions (one per video)
            prompt:  str, LLM quality assessment prompt

        Returns:
            final_score:    [B] fused quality prediction
            semantic_score: [B] semantic consistency score (range ~1-5)
            fidelity_score: [B] motion fidelity score (unconstrained)
        """
        video = data['video']
        video_aesthetic = data.get('video_aesthetic', video)
        B, C, T, H, W = video.shape
        device = video.device

        # ==============================================================
        # Shared: CLIP feature extraction (frozen)
        # ==============================================================
        frames = video.transpose(1, 2).reshape(B * T, C, H, W)
        with torch.no_grad():
            clip_feats = self.clip(pixel_values=frames).pooler_output  # [B*T, 1024]
        clip_feats = clip_feats.view(B, T, -1)  # [B, T, 1024]

        # ==============================================================
        # Branch 1: Semantic Consistency (CLIP -> QFormer -> Qwen)
        # ==============================================================
        cap_tokens = self.llm_tokenizer(
            caption, padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            cap_embeds = self.llm_model.get_input_embeddings()(
                cap_tokens.input_ids)  # [B, L, 3584]

        semantic_embeds = self.qformer(
            video_feats=clip_feats,
            text_feats=cap_embeds,
            text_mask=cap_tokens.attention_mask,
        )  # [B, T, 3584]

        atts_semantic = torch.ones(
            semantic_embeds.size()[:-1], dtype=torch.long, device=device)

        full_prompt = ([f"Video description: {c}. {prompt}" for c in caption]
                       if isinstance(caption, list)
                       else [f"Video description: {caption}. {prompt}"] * B)

        llm_tokens = self.llm_tokenizer(
            full_prompt, padding="longest", return_tensors="pt").to(device)

        with self.maybe_autocast():
            text_embeds = self.llm_model.get_input_embeddings()(
                llm_tokens.input_ids)
            inputs_embeds = torch.cat(
                [semantic_embeds.to(text_embeds.dtype), text_embeds], dim=1)
            attention_mask = torch.cat(
                [atts_semantic, llm_tokens.attention_mask], dim=1)
            outputs = self.llm_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask)

        logits_last = outputs.logits[:, -1]
        q_logits = torch.stack([
            logits_last[:, self.excellent_idx],
            logits_last[:, self.good_idx],
            logits_last[:, self.fair_idx],
            logits_last[:, self.poor_idx],
            logits_last[:, self.bad_idx],
        ])  # [5, B]
        q_pred = (q_logits / 100).softmax(0)
        w = self.weights.expand(-1, B).to(device)
        semantic_score = torch.sum(q_pred * w, dim=0)  # [B]

        # ==============================================================
        # Branch 2: Motion Fidelity (Swin3D + ConvNeXt3D + CrossGating)
        # ==============================================================
        # ---- CLIP anchor ----
        clip_anchor = self.clip_to_anchor(clip_feats)              # [B, T, 768]
        clip_anchor = clip_anchor.permute(0, 2, 1)                 # [B, 768, T]
        clip_anchor = clip_anchor.unsqueeze(-1).unsqueeze(-1)      # [B, 768, T, 1, 1]
        clip_anchor = clip_anchor.expand(-1, -1, -1, 7, 7)        # [B, 768, T, 7, 7]
        clip_anchor = self.anchor_pool(clip_anchor)                # [B, 768, T//2, 7, 7]

        # ---- Technical pathway (Swin3D) ----
        tech_feat = self.technical_backbone(video)                 # [B, 768, T//2, 7, 7]
        if tech_feat.shape[2:] != clip_anchor.shape[2:]:
            tech_feat = F.adaptive_avg_pool3d(tech_feat, clip_anchor.shape[2:])
        tech_gated = self.cross_gate_tech(clip_anchor, tech_feat)
        tech_score = torch.mean(
            self.technical_head(tech_gated), dim=(1, 2, 3, 4))     # [B]

        # ---- Aesthetic pathway (ConvNeXt3D) ----
        aes_feat = self.aesthetic_backbone(video_aesthetic)         # [B, 768, T', H', W']
        aes_feat = self.aes_pool(aes_feat)                         # [B, 768, T//2, 7, 7]
        aes_gated = self.cross_gate_aes(clip_anchor, aes_feat)
        aes_score = torch.mean(
            self.aesthetic_head(aes_gated), dim=(1, 2, 3, 4))      # [B]

        fidelity_score = tech_score + aes_score                    # [B]

        # ==============================================================
        # Fusion
        # ==============================================================
        final_score = self.fusion_head(semantic_score, fidelity_score)  # [B]

        return final_score, semantic_score, fidelity_score


if __name__ == "__main__":
    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

    mock_args = {
        'clip_len': 8,
        'llm_model': 'Qwen/Qwen2.5-7B-Instruct',
        'clip_weights': 'openai/clip-vit-large-patch14',
        'swin_window_size': [4, 7, 7],
        'conv_pretrained': True,
    }

    model = AIGCVideoQA(args=mock_args).to(device)
    model.eval()

    caption = ['A dog running in a park'] * 2
    prompt = ('Evaluate how well the video matches the text description. '
              'The semantic consistency is')
    video = torch.randn(2, 3, 8, 224, 224).to(device)
    data = {'video': video, 'video_aesthetic': video}

    with torch.no_grad():
        final, sem, fid = model(data, caption, prompt)
    print(f"Final: {final}, Semantic: {sem}, Fidelity: {fid}")
