#!/usr/bin/env python3
import os
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
import torch.nn.functional as F

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from sam3.model_builder import build_sam3_image_model
from sam3.train.data.collator import collate_fn_api
from lora_layers import apply_lora_to_model, LoRAConfig
from train_sam3_lora_native import COCOSegmentDataset
from train_sam3_lora_native import merge_overlapping_masks


# =========================
# ✅ Metrics
# =========================
def compute_iou(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return inter / (union + 1e-6)


def compute_dice(pred, gt):
    inter = np.logical_and(pred, gt).sum()
    return (2 * inter) / (pred.sum() + gt.sum() + 1e-6)


def match_masks(preds, gts):
    matches = []
    used = set()

    for i, p in enumerate(preds):
        best_iou, best_j = 0, -1

        for j, g in enumerate(gts):
            if j in used:
                continue
            iou = compute_iou(p, g)
            if iou > best_iou:
                best_iou, best_j = iou, j

        if best_j != -1:
            used.add(best_j)
            matches.append((i, best_j, best_iou))

    return matches


# =========================
# ✅ 强制 resize
# =========================
def safe_resize(pred, gt_shape):

    if pred.shape == gt_shape:
        return pred

    p = torch.tensor(pred, dtype=torch.float32)[None, None]
    p = F.interpolate(p, size=gt_shape, mode="nearest")

    return (p.squeeze().numpy() > 0)


# =========================
# ✅ 保存（final修正版）
# =========================
def save_prediction_and_gt(pred, dataset, idx, output_root):

    dp = dataset[idx]
    img_obj = dp.images[0]

    for attr in ["file_path", "path", "file_name", "image_path"]:
        if hasattr(img_obj, attr):
            img_name = Path(getattr(img_obj, attr)).stem
            break
    else:
        img_name = f"img_{idx}"

    out_dir = Path(output_root) / img_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # ========= GT =========
    gts = []
    for k, obj in enumerate(dp.images[0].objects):
        if obj.segment is None:
            continue

        gt = obj.segment.numpy().astype(np.uint8)
        gts.append(gt)
        Image.fromarray(gt * 255).save(out_dir / f"gt_{k}.png")

    if len(gts) == 0:
        return

    h, w = gts[0].shape

    # ========= PRED =========
    if pred is None:
        return

    masks = pred["pred_masks"].cpu()
    logits = pred["pred_logits"].cpu()

    scores = torch.sigmoid(logits).view(-1)

    # ✅ shape统一
    if masks.dim() == 4:
        masks = masks[:, 0]

    # ✅ 对齐长度（防止 index error ✅）
    N = min(len(masks), len(scores))
    masks = masks[:N]
    scores = scores[:N]

    # ✅ prob → binary
    masks = torch.sigmoid(masks)
    binary_masks = (masks > 0.5).cpu()

    # ✅ dummy boxes
    dummy_boxes = torch.zeros((N, 4))

    # ✅ ✅ ✅ merge（安全调用）
    if N > 0 and len(binary_masks) == len(scores):
        binary_masks, scores, _ = merge_overlapping_masks(
            binary_masks,
            scores,
            dummy_boxes,
            iou_threshold=0.3
        )

    # ========= 保存 =========
    for j in range(len(binary_masks)):

        m = binary_masks[j].float()

        m = F.interpolate(
            m.unsqueeze(0).unsqueeze(0),
            size=(h, w),
            mode="nearest"
        ).squeeze().numpy().astype(np.uint8)

        Image.fromarray(m * 255).save(
            out_dir / f"pred_{j}.png"
        )


# =========================
# ✅ inference
# =========================
@torch.no_grad()
def inference(model, loader, dataset, device, output_dir):

    model.eval()

    for i, batch_dict in enumerate(tqdm(loader, desc="Inference")):

        input_batch = batch_dict["input"]

        def move(x):
            if isinstance(x, torch.Tensor):
                return x.to(device)
            elif isinstance(x, list):
                return [move(v) for v in x]
            elif hasattr(x, "__dataclass_fields__"):
                for f in x.__dataclass_fields__:
                    setattr(x, f, move(getattr(x, f)))
            return x

        input_batch = move(input_batch)

        # ✅ 不添加任何 bbox prompt（关键）
        with torch.amp.autocast("cuda"):
            outputs = model(input_batch)

        last = outputs[-1] if isinstance(outputs, list) else outputs

        pred = None
        if isinstance(last, dict):
            pred = {
                "pred_logits": last["pred_logits"].detach(),
                "pred_masks": last["pred_masks"].detach(),
            }

        save_prediction_and_gt(pred, dataset, i, output_dir)

        del outputs, last, pred, input_batch
        torch.cuda.empty_cache()


# =========================
# ✅ evaluate
# =========================
def evaluate(output_dir, json_path):

    all_dice, all_iou = [], []
    TP = FP = FN = 0

    for case in Path(output_dir).iterdir():

        if not case.is_dir():
            continue

        gts, preds = [], []

        for f in case.iterdir():
            img = np.array(Image.open(f)) > 0

            if f.name.startswith("gt_"):
                gts.append(img)
            elif f.name.startswith("pred_"):
                preds.append(img)

        if len(gts) == 0:
            continue

        gt_shape = gts[0].shape
        preds = [safe_resize(p, gt_shape) for p in preds]

        matches = match_masks(preds, gts)

        matched_p, matched_g = set(), set()

        for i, j, iou in matches:
            matched_p.add(i)
            matched_g.add(j)

            all_iou.append(iou)
            all_dice.append(compute_dice(preds[i], gts[j]))

            if iou > 0.3:
                TP += 1

        FP += len(preds) - len(matched_p)
        FN += len(gts) - len(matched_g)

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)

    metrics = {
        "Dice": float(np.mean(all_dice)) if all_dice else 0,
        "IoU": float(np.mean(all_iou)) if all_iou else 0,
        "Precision": float(precision),
        "Recall": float(recall),
        "F1": float(f1),
    }

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n========= Evaluation =========")
    print(metrics)


# =========================
# main
# =========================
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--test_data_dir", required=True)
    parser.add_argument("--output_dir", default="results")
    parser.add_argument("--json_out", default="metrics.json")

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda")

    print("Building model...")
    model = build_sam3_image_model(device=device.type, load_from_HF=True)

    import yaml
    cfg = yaml.safe_load(open(args.config))

    valid_keys = LoRAConfig.__init__.__code__.co_varnames
    lora_cfg = {k: v for k, v in cfg["lora"].items() if k in valid_keys}

    model = apply_lora_to_model(model, LoRAConfig(**lora_cfg))

    print("Loading weights...")
    model.load_state_dict(torch.load(args.weights, map_location="cpu"), strict=False)

    model.to(device)
    model.eval()

    dataset = COCOSegmentDataset(args.test_data_dir, split="test")

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=lambda b: collate_fn_api(
            b, dict_key="input", with_seg_masks=True
        ),
    )

    inference(model, loader, dataset, device, args.output_dir)

    print("Running evaluation...")
    evaluate(args.output_dir, args.json_out)

    print("✅ Done")


if __name__ == "__main__":
    main()