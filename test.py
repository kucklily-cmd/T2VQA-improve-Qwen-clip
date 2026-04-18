import torch
import random
import os.path as osp
import argparse
from scipy.stats import spearmanr, pearsonr
from scipy.stats.stats import kendalltau as kendallr
import numpy as np
from tqdm import tqdm
import yaml
from collections import OrderedDict
import copy

from model.model import AIGCVideoQA
from dataset.dataset import T2VDataset


def train_test_split(dataset_path, ann_file, ratio=0.8, seed=42):
    random.seed(seed)
    print(seed)
    video_infos = []
    with open(ann_file, "r") as fin:
        for line in fin.readlines():
            line_split = line.strip().split("|")
            if len(line_split) == 3:
                filename, prompt, label = line_split
            elif len(line_split) == 4:
                filename, prompt, _, label = line_split
            else:
                raise ValueError(f"Invalid line with {len(line_split)} fields: {line.strip()}")
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


PROMPT_TEMPLATE = (
    "Evaluate how well the video content matches the given text description. "
    "The semantic consistency of this video is"
)


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
            head_state_dict = OrderedDict()
            trainable_keywords = [
                "qformer", "lora",
                "technical_backbone",
                "clip_to_anchor",
                "cross_gate_tech",
                "technical_head",
                "fusion_head",
            ]
            for key, v in state_dict.items():
                if any(t in key for t in trainable_keywords):
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
    return best_s, best_p, best_k, best_r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--opt", type=str, default="t2vqa.yml")
    parser.add_argument("-t", "--target_set", type=str, default="t2v")
    args = parser.parse_args()

    with open(args.opt, "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AIGCVideoQA(opt["model"]["args"]).to(device)

    # Load checkpoint
    try:
        ckpt = torch.load(opt['test_load_path'], map_location='cpu', weights_only=False)
    except TypeError:
        ckpt = torch.load(opt['test_load_path'], map_location='cpu')

    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

    # Remove keys from truly legacy architectures (old T2VQA 3-branch model).
    # COVER fidelity branch keys (aesthetic_backbone, cross_gate_aes,
    # aesthetic_head, technical_backbone, etc.) are VALID and must be kept.
    keys_to_remove = [k for k in state_dict.keys()
                      if any(t in k for t in [
                          "finetune_Qformer", "blip", "gate_mixer",
                          "slowfast", "motion_proj",
                          "aesthetic_conv3d", "aesthetic_pool", "aesthetic_proj",
                      ])]
    for k in keys_to_remove:
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    print(f"Loaded weights: {msg}")
    model.eval()

    if opt.get("split_seed", -1) > 0:
        opt["data"]["train"] = copy.deepcopy(opt["data"][args.target_set])
        opt["data"]["eval"]  = copy.deepcopy(opt["data"][args.target_set])
        split_duo = train_test_split(
            opt["data"][args.target_set]["args"]["data_prefix"],
            opt["data"][args.target_set]["args"]["anno_file"],
            seed=opt["split_seed"],
        )
        (opt["data"]["train"]["args"]["anno_file"],
         opt["data"]["eval"]["args"]["anno_file"]) = split_duo

    val_datasets, val_loaders = {}, {}
    for key in opt["data"]:
        if key.startswith("eval"):
            ds = T2VDataset(opt["data"][key]["args"])
            val_datasets[key] = ds
            val_loaders[key] = torch.utils.data.DataLoader(
                ds, batch_size=1,
                num_workers=opt["num_workers"], pin_memory=True,
            )

    bests = {}
    for key in val_loaders:
        bests[key] = -1, -1, -1, 1000
        bests[key] = inference_set(
            val_loaders[key], model, device, bests[key],
            save_model=False, suffix=key + "_s",
        )


if __name__ == "__main__":
    main()
