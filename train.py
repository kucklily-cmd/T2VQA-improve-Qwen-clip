import torch
import cv2
import random
import os
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from time import time
from tqdm import tqdm
import pickle
import math
import yaml
from collections import OrderedDict
import json
from functools import reduce
import copy

from model.model import AIGCVideoQA
from dataset.dataset import T2VDataset


# ============================================================
# Utility Functions
# ============================================================

def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            filename, prompt, label = line_split
            label = float(label)
            filename = osp.join(dataset_path, filename)
            video_infos.append(dict(filename=filename, prompt=prompt, label=label))
    random.shuffle(video_infos)
    return (
        video_infos[: int(ratio * len(video_infos))],
        video_infos[int(ratio * len(video_infos)) :],
    )


def rescale(pr, gt=None):
    if gt is None:
        pr = (pr - np.mean(pr)) / np.std(pr)
    else:
        pr = ((pr - np.mean(pr)) / np.std(pr)) * np.std(gt) + np.mean(gt)
    return pr


# ============================================================
# Loss Functions
# ============================================================

def plcc_loss(y_pred, y):
    """Pearson Linear Correlation Coefficient loss (scale-invariant)."""
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def rank_loss(y_pred, y):
    """Ranking loss: encourages correct relative ordering."""
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def composite_loss(final_score, semantic_score, fidelity_score, gt,
                   lambda_rank=0.3, lambda_aux=0.2):
    """Multi-task composite loss with auxiliary branch supervision.

    Components:
      1. Main loss:  PLCC(final, gt) + lambda_rank * Rank(final, gt)
      2. Aux losses: PLCC(semantic, gt) + PLCC(fidelity, gt)
         - Prevents one branch from dominating
         - Provides direct gradient signal to each branch
         - PLCC is scale-invariant, so different score ranges are OK
    """
    main_plcc = plcc_loss(final_score, gt)
    main_rank = rank_loss(final_score, gt)
    main_loss = main_plcc + lambda_rank * main_rank

    aux_semantic = plcc_loss(semantic_score, gt)
    aux_fidelity = plcc_loss(fidelity_score, gt)

    total = main_loss + lambda_aux * (aux_semantic + aux_fidelity)
    return total


# ============================================================
# Training & Validation
# ============================================================

# Semantic consistency focused prompt
PROMPT_TEMPLATE = (
    "Evaluate how well the video content matches the given text description. "
    "The semantic consistency of this video is"
)


def finetune_epoch(ft_loader, model, optimizer, scheduler, device, epoch=-1,
                   lambda_rank=0.3, lambda_aux=0.2):
    model.train()
    total_loss = 0.0

    for i, data in enumerate(tqdm(ft_loader, desc=f"Training epoch {epoch}")):
        optimizer.zero_grad()

        inputs = {}
        inputs["video"] = data["video"].to(device)

        y = data["gt_label"].float().detach().to(device)
        caption = data['prompt']

        final_score, semantic_score, fidelity_score = model(
            inputs, caption=caption, prompt=PROMPT_TEMPLATE)

        loss = composite_loss(final_score, semantic_score, fidelity_score, y,
                              lambda_rank=lambda_rank, lambda_aux=lambda_aux)

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    model.eval()
    avg_loss = total_loss / len(ft_loader)
    return avg_loss


def inference_set(
    inf_loader, model, device, best_,
    save_model=False, suffix="s", save_name="divide", save_type="head",
):
    results = []
    best_s, best_p, best_k, best_r = best_

    for i, data in enumerate(tqdm(inf_loader, desc="Validating")):
        result = dict()
        inputs = {}

        inputs['video'] = data['video'].to(device)

        with torch.no_grad():
            caption = data['prompt']
            final_score, _, _ = model(
                inputs, caption=caption, prompt=PROMPT_TEMPLATE)
            result["pr_labels"] = final_score.cpu().numpy()

        result["gt_label"] = data["gt_label"].item()
        del inputs
        results.append(result)

    gt_labels = [r["gt_label"] for r in results]
    pr_labels = [np.mean(r["pr_labels"]) for r in results]
    pr_labels = rescale(pr_labels, gt_labels)

    s = spearmanr(gt_labels, pr_labels)[0]
    p = pearsonr(gt_labels, pr_labels)[0]
    k = kendallr(gt_labels, pr_labels)[0]
    r = np.sqrt(((gt_labels - pr_labels) ** 2).mean())

    del results
    torch.cuda.empty_cache()

    if s + p > best_s + best_p and save_model:
        state_dict = model.state_dict()

        if save_type == "head":
            # Save only trainable components (not frozen CLIP/Qwen base)
            head_state_dict = OrderedDict()
            for key, v in state_dict.items():
                if any(t in key for t in [
                    "qformer", "lora",
                    "technical_backbone",
                    "clip_to_anchor",
                    "cross_gate_tech",
                    "technical_head",
                    "fusion_head",
                ]):
                    head_state_dict[key] = v
            print("Saving head-only keys:", len(head_state_dict))
            torch.save(
                {"state_dict": head_state_dict, "validation_results": best_},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )
        else:
            torch.save(
                {"state_dict": state_dict, "validation_results": best_},
                f"pretrained_weights/{save_name}_{suffix}_finetuned.pth",
            )

    best_s, best_p, best_k, best_r = (
        max(best_s, s), max(best_p, p), max(best_k, k), min(best_r, r),
    )

    print(
        f"For {len(inf_loader)} videos, \n"
        f"the accuracy [{suffix}]:\n"
        f"  SROCC: {s:.4f} best: {best_s:.4f}\n"
        f"  PLCC:  {p:.4f} best: {best_p:.4f}\n"
        f"  KROCC: {k:.4f} best: {best_k:.4f}\n"
        f"  RMSE:  {r:.4f} best: {best_r:.4f}"
    )
    return (best_s, best_p, best_k, best_r), (s, p, k, r)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="t2vqa.yml")
    parser.add_argument("-t", "--target_set", type=str, default="t2v")
    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    bests_ = []
    num_splits = 10 if opt.get("split_seed", -1) > 0 else 1

    for split in range(num_splits):
        model = AIGCVideoQA(opt["model"]["args"]).to(device)

        if opt.get("split_seed", -1) > 0:
            opt["data"]["train"] = copy.deepcopy(opt["data"][args.target_set])
            opt["data"]["eval"] = copy.deepcopy(opt["data"][args.target_set])
            split_duo = train_test_split(
                opt["data"][args.target_set]["args"]["data_prefix"],
                opt["data"][args.target_set]["args"]["anno_file"],
                seed=opt["split_seed"] * (split + 1),
            )
            (opt["data"]["train"]["args"]["anno_file"],
             opt["data"]["eval"]["args"]["anno_file"]) = split_duo

        # Build data loaders
        train_datasets, train_loaders = {}, {}
        for key in opt["data"]:
            if key.startswith("train"):
                ds = T2VDataset(opt["data"][key]["args"])
                train_datasets[key] = ds
                train_loaders[key] = torch.utils.data.DataLoader(
                    ds, batch_size=opt["batch_size"],
                    num_workers=opt["num_workers"], shuffle=True,
                )
                print(f"Train [{key}]: {len(ds)} videos")

        val_datasets, val_loaders = {}, {}
        for key in opt["data"]:
            if key.startswith("eval"):
                ds = T2VDataset(opt["data"][key]["args"])
                val_datasets[key] = ds
                val_loaders[key] = torch.utils.data.DataLoader(
                    ds, batch_size=1,
                    num_workers=opt["num_workers"], pin_memory=True,
                )
                print(f"Val   [{key}]: {len(ds)} videos")

        # ---- Trainable parameter groups ----
        # Trainable: qformer, lora, fidelity branch (Swin3D, ConvNeXt3D,
        # Trainable: qformer, lora, COVER fidelity branch (both Swin3D + ConvNeXt3D
        #            sub-branches, cross gates, heads), clip_to_anchor, fusion_head
        # Frozen:    CLIP, Qwen base
        param_groups = []
        trainable_keywords = [
            "qformer", "lora",
            # COVER technical sub-branch
            "technical_backbone",
            "cross_gate_tech",
            "technical_head",
            # COVER aesthetic sub-branch
            "aesthetic_backbone",
            "cross_gate_aes",
            "aesthetic_head",
            # shared
            "clip_to_anchor",
            "fusion_head",
        ]
        for name, param in model.named_parameters():
            if any(kw in name for kw in trainable_keywords):
                param.requires_grad = True
                param_groups.append(
                    {"params": param, "lr": opt["optimizer"]["lr"]})
            else:
                param.requires_grad = False

        optimizer = torch.optim.AdamW(
            lr=opt["optimizer"]["lr"],
            params=param_groups,
            weight_decay=opt["optimizer"]["wd"],
        )

        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int(opt["num_epochs"] * len(train_loader))

        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

        bests = {}
        for key in val_loaders:
            bests[key] = -1, -1, -1, 1000
        history_log = []

        # ---- Auto-resume ----
        start_epoch = 0
        os.makedirs("pretrained_weights", exist_ok=True)
        latest_ckpt_path = f"pretrained_weights/{opt['name']}_latest_{args.target_set}_{split}.pth"

        if os.path.exists(latest_ckpt_path):
            print(f"\n[*] Resuming from {latest_ckpt_path} ...")
            checkpoint = torch.load(latest_ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["state_dict"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            start_epoch = checkpoint["epoch"] + 1
            bests = checkpoint.get("bests", bests)
            log_file = f"training_history_split{split}.json"
            if os.path.exists(log_file):
                with open(log_file, "r", encoding="utf-8") as f:
                    try:
                        history_log = json.load(f)
                    except Exception:
                        pass
            print(f"[*] Resumed from epoch {checkpoint['epoch']}, next: {start_epoch}\n")

        # ---- Training loop ----
        lambda_rank = opt.get('loss', {}).get('lambda_rank', 0.3)
        lambda_aux  = opt.get('loss', {}).get('lambda_aux', 0.2)

        for epoch in range(start_epoch, opt["num_epochs"]):
            print(f"Epoch {epoch}:")
            epoch_train_loss = 0.0

            for key, train_loader in train_loaders.items():
                epoch_train_loss = finetune_epoch(
                    train_loader, model, optimizer, scheduler, device, epoch,
                    lambda_rank=lambda_rank, lambda_aux=lambda_aux)

            epoch_data = {"epoch": epoch, "train_loss": epoch_train_loss}

            for key in val_loaders:
                bests[key], current_metrics = inference_set(
                    val_loaders[key], model, device, bests[key],
                    save_model=opt["save_model"],
                    save_name=opt["name"] + "_head_" + args.target_set + f"_{split}",
                    suffix=key + "_s",
                    save_type="head",
                )
                s, p, k, r = current_metrics
                epoch_data[f"val_{key}_SRCC"] = float(s)
                epoch_data[f"val_{key}_PLCC"] = float(p)
                epoch_data[f"val_{key}_KRCC"] = float(k)
                epoch_data[f"val_{key}_RMSE"] = float(r)

            history_log.append(epoch_data)

            # ---- Safe checkpoint saving ----
            head_state_dict = OrderedDict()
            for k_name, v_param in model.state_dict().items():
                if any(kw in k_name for kw in trainable_keywords):
                    head_state_dict[k_name] = v_param

            latest_state = {
                "epoch": epoch,
                "state_dict": head_state_dict,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "bests": bests,
            }

            tmp_ckpt_path = latest_ckpt_path + ".tmp"
            json_file = f"training_history_split{split}.json"
            tmp_json_file = json_file + ".tmp"

            try:
                with open(tmp_json_file, "w", encoding="utf-8") as f:
                    json.dump(history_log, f, indent=4)
                os.replace(tmp_json_file, json_file)

                torch.save(latest_state, tmp_ckpt_path)
                os.replace(tmp_ckpt_path, latest_ckpt_path)
                print(f"[*] Checkpoint saved to {latest_ckpt_path}")
            except Exception as e:
                print(f"[!] WARNING: Save failed: {e}")
                for tmp_file in [tmp_ckpt_path, tmp_json_file]:
                    if os.path.exists(tmp_file):
                        try:
                            os.remove(tmp_file)
                        except Exception:
                            pass

        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
