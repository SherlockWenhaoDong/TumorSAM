#!/usr/bin/env python3
"""
SAM3 + LoRA Inference Script

Based on official SAM3 batched inference patterns.
Supports text prompts and visual prompts with LoRA fine-tuned weights.

Usage:
    # Text prompt inference
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" \
        --output output.png

    # Multiple prompts
    python3 infer_sam.py \
        --config configs/full_lora_config.yaml \
        --image path/to/image.jpg \
        --prompt "crack" "defect" "damage" \
        --output output.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import yaml
from torchvision.ops import nms

# SAM3 imports
from sam3.model_builder import build_sam3_image_model
from sam3.train.data.sam3_image_dataset import (
    Datapoint,
    Image as SAMImage,
    FindQueryLoaded,
    InferenceMetadata
)
from sam3.train.data.collator import collate_fn_api
from sam3.model.utils.misc import copy_data_to_device
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
)
from sam3.eval.postprocessors import PostProcessImage

# LoRA imports
from lora_layers import LoRAConfig, apply_lora_to_model, load_lora_weights

import cv2
import json

from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
from pycocotools.cocoeval import COCOeval

def compute_dsc(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()

    denom = pred_mask.sum() + gt_mask.sum()

    if denom == 0:
        return None

    return 2.0 * intersection / denom


def compute_iou(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    intersection = np.logical_and(pred_mask, gt_mask).sum()

    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        return None

    return intersection / union

def compute_cgf1(pred_mask, gt_mask):
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    tp = np.logical_and(pred_mask, gt_mask).sum()

    precision = tp / pred_mask.sum() if pred_mask.sum() > 0 else 0

    recall = tp / gt_mask.sum() if gt_mask.sum() > 0 else 0

    if precision + recall == 0:
        return 0

    return 2 * precision * recall / (precision + recall)

def merge_masks(masks):
    if masks is None:
        return None

    if len(masks) == 0:
        return None

    merged = np.zeros_like(masks[0], dtype=np.uint8)

    for m in masks:
        merged = np.logical_or(merged, m)

    return merged.astype(np.uint8)


class CocoMetricAccumulator:

    def __init__(self):
        self.images = []
        self.annotations = []
        self.predictions = []

        self.image_id = 1
        self.annotation_id = 1

    def add_sample(
            self,
            gt_mask,
            pred_masks,
            pred_scores
    ):
        h, w = gt_mask.shape

        self.images.append(
            {
                "id": self.image_id,
                "width": w,
                "height": h
            }
        )

        gt_rle = mask_utils.encode(
            np.asfortranarray(gt_mask.astype(np.uint8))
        )

        gt_rle["counts"] = gt_rle["counts"].decode()

        self.annotations.append(
            {
                "id": self.annotation_id,
                "image_id": self.image_id,
                "category_id": 1,
                "segmentation": gt_rle,
                "area": float(gt_mask.sum()),
                "bbox": mask_utils.toBbox(gt_rle).tolist(),
                "iscrowd": 0
            }
        )

        if pred_masks is not None:

            for mask, score in zip(pred_masks, pred_scores):
                rle = mask_utils.encode(
                    np.asfortranarray(mask.astype(np.uint8))
                )

                rle["counts"] = rle["counts"].decode()

                self.predictions.append(
                    {
                        "image_id": self.image_id,
                        "category_id": 1,
                        "segmentation": rle,
                        "score": float(score)
                    }
                )
        self.image_id += 1
        self.annotation_id += 1

    def evaluate(self):

        if len(self.annotations) == 0:
            return 0
        gt_dataset = {
            "images": self.images,
            "annotations": self.annotations,
            "categories": [
                {
                    "id": 1,
                    "name": "target"
                }
            ]
        }
        coco_gt = COCO()
        coco_gt.dataset = gt_dataset
        coco_gt.createIndex()

        coco_dt = coco_gt.loadRes(self.predictions)

        coco_eval = COCOeval(
            coco_gt,
            coco_dt,
            "segm"
        )

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats[0]




class SAM3LoRAInference:
    """SAM3 model with LoRA for inference."""

    def __init__(
            self,
            config_path: str,
            weights_path: Optional[str] = None,
            resolution: int = 1008,
            detection_threshold: float = 0.5,
            nms_iou_threshold: float = 0.5,
            keep_top1: bool = False,
            device: str = "cuda"
    ):
        """
        Initialize SAM3 with LoRA.

        Args:
            config_path: Path to training config YAML
            weights_path: Path to LoRA weights (optional, auto-detected from config)
            resolution: Input image resolution (default: 1008)
            detection_threshold: Confidence threshold for detections (default: 0.5)
            nms_iou_threshold: IoU threshold for NMS (default: 0.5)
            device: Device to run on (default: "cuda")
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.keep_top1 = keep_top1

        # Auto-detect weights if not provided
        if weights_path is None:
            output_dir = self.config.get('output', {}).get('output_dir', 'outputs/sam3_lora_full')
            weights_path = os.path.join(output_dir, 'best_lora_weights.pt')
            print(f"ℹ️  Auto-detected weights: {weights_path}")

        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"LoRA weights not found: {weights_path}")

        self.weights_path = weights_path
        self.resolution = resolution
        self.detection_threshold = detection_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        print(f"🔧 Initializing SAM3 + LoRA...")
        print(f"   Device: {self.device}")
        print(f"   Resolution: {resolution}x{resolution}")
        print(f"   Confidence threshold: {detection_threshold}")
        print(f"   NMS IoU threshold: {nms_iou_threshold}")

        # Build base model
        print("\n📦 Building SAM3 model...")
        self.model = build_sam3_image_model(
            device=self.device.type,
            compile=False,
            load_from_HF=True,
            bpe_path="sam3/assets/bpe_simple_vocab_16e6.txt.gz",
            eval_mode=True
        )

        # Apply LoRA configuration
        print("🔗 Applying LoRA configuration...")
        lora_cfg = self.config["lora"]
        lora_config = LoRAConfig(
            rank=lora_cfg["rank"],
            alpha=lora_cfg["alpha"],
            dropout=0.0,  # No dropout during inference
            target_modules=lora_cfg["target_modules"],
            apply_to_vision_encoder=lora_cfg["apply_to_vision_encoder"],
            apply_to_text_encoder=lora_cfg["apply_to_text_encoder"],
            apply_to_geometry_encoder=lora_cfg["apply_to_geometry_encoder"],
            apply_to_detr_encoder=lora_cfg["apply_to_detr_encoder"],
            apply_to_detr_decoder=lora_cfg["apply_to_detr_decoder"],
            apply_to_mask_decoder=lora_cfg["apply_to_mask_decoder"],
        )
        self.model = apply_lora_to_model(self.model, lora_config)

        # Load LoRA weights
        print(f"💾 Loading LoRA weights from {weights_path}...")
        load_lora_weights(self.model, weights_path)

        self.model.to(self.device)
        self.model.eval()

        # Setup transforms (official SAM3 pattern)
        self.transform = ComposeAPI(
            transforms=[
                RandomResizeAPI(
                    sizes=resolution,
                    max_size=resolution,
                    square=True,
                    consistent_transform=False
                ),
                ToTensorAPI(),
                NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        # Setup postprocessor
        # Note: Using simpler manual postprocessing instead of PostProcessImage
        # because PostProcessImage may have additional filtering logic
        self.use_manual_postprocess = True

        print("✅ SAM3 + LoRA ready for inference!\n")

    def create_datapoint(self, pil_image: PILImage.Image, text_prompts: List[str]) -> Datapoint:
        """
        Create a SAM3 datapoint from image and text prompts.

        Args:
            pil_image: PIL Image
            text_prompts: List of text queries

        Returns:
            Datapoint with image and queries
        """
        w, h = pil_image.size

        # Create SAM Image
        sam_image = SAMImage(
            data=pil_image,
            objects=[],
            size=[h, w]
        )

        # Create queries for each text prompt
        queries = []
        for idx, text_query in enumerate(text_prompts):
            query = FindQueryLoaded(
                query_text=text_query,
                image_id=0,
                object_ids_output=[],
                is_exhaustive=True,
                query_processing_order=idx,
                inference_metadata=InferenceMetadata(
                    coco_image_id=idx,
                    original_image_id=idx,
                    original_category_id=1,
                    original_size=[w, h],
                    object_id=0,
                    frame_index=0,
                )
            )
            queries.append(query)

        return Datapoint(
            find_queries=queries,
            images=[sam_image]
        )

    @torch.no_grad()
    def predict(
        self,
        image_path: str,
        text_prompts: List[str]
    ) -> dict:
        """
        Run inference on an image with text prompts.

        Args:
            image_path: Path to input image
            text_prompts: List of text queries (e.g., ["crack", "defect"])

        Returns:
            Dictionary mapping prompt index to predictions:
            {
                0: {'boxes': [...], 'scores': [...], 'masks': [...]},
                1: {'boxes': [...], 'scores': [...], 'masks': [...]}
            }
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        pil_image = PILImage.open(image_path).convert("RGB")
        print(f"📷 Loaded image: {image_path}")
        print(f"   Size: {pil_image.size}")
        print(f"   Prompts: {text_prompts}")

        print("\n🔮 Running inference...")

        results = {}

        # Process each prompt separately (SAM3 expects one query per forward pass)
        for query_idx, prompt in enumerate(text_prompts):
            # Create datapoint with single prompt
            datapoint = self.create_datapoint(pil_image, [prompt])

            # Apply transforms
            datapoint = self.transform(datapoint)

            # Collate into batch
            batch = collate_fn_api([datapoint], dict_key="input")["input"]

            # Move to device
            batch = copy_data_to_device(batch, self.device, non_blocking=True)

            # Forward pass
            outputs = self.model(batch)

            # Manual post-processing
            last_output = outputs[-1]
            pred_logits = last_output['pred_logits']  # [batch, num_queries, num_classes]
            pred_boxes = last_output['pred_boxes']    # [batch, num_queries, 4]
            pred_masks = last_output.get('pred_masks', None)  # [batch, num_queries, H, W]

            # Get probabilities
            out_probs = pred_logits.sigmoid()  # [batch, num_queries, num_classes]

            # Get scores for this query
            scores = out_probs[0, :, :].max(dim=-1)[0]  # [num_queries]

            # Filter by threshold
            keep = scores > self.detection_threshold
            num_keep = keep.sum().item()

            if num_keep > 0:
                # Get boxes and convert from cxcywh to xyxy
                boxes_cxcywh = pred_boxes[0, keep]  # [num_keep, 4]
                kept_scores = scores[keep]
                cx, cy, w, h = boxes_cxcywh.unbind(-1)

                # Convert to xyxy and scale to original image size
                orig_w, orig_h = pil_image.size
                x1 = (cx - w / 2) * orig_w
                y1 = (cy - h / 2) * orig_h
                x2 = (cx + w / 2) * orig_w
                y2 = (cy + h / 2) * orig_h

                boxes_xyxy = torch.stack([x1, y1, x2, y2], dim=-1)

                # Apply NMS to remove overlapping boxes
                keep_nms = nms(
                    boxes_xyxy,
                    kept_scores,
                    self.nms_iou_threshold
                )

                boxes_xyxy = boxes_xyxy[keep_nms]
                kept_scores = kept_scores[keep_nms]
                #
                # Keep only highest confidence prediction
                #
                if self.keep_top1 and len(kept_scores) > 0:
                    best_idx = torch.argmax(kept_scores)

                    boxes_xyxy = boxes_xyxy[best_idx].unsqueeze(0)
                    kept_scores = kept_scores[best_idx].unsqueeze(0)

                    keep_nms = keep_nms[best_idx].unsqueeze(0)

                # Get masks and resize to original size
                if pred_masks is not None:
                    # Apply NMS filtering to masks too
                    masks_small = pred_masks[0, keep][keep_nms].sigmoid() > 0.5  # [num_keep_nms, H, W]

                    # Resize masks to original image size
                    import torch.nn.functional as F
                    masks_resized = F.interpolate(
                        masks_small.unsqueeze(0).float(),
                        size=(orig_h, orig_w),
                        mode='bilinear',
                        align_corners=False
                    ).squeeze(0) > 0.5

                    masks_np = masks_resized.cpu().numpy()
                else:
                    masks_np = None

                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': boxes_xyxy.cpu().numpy(),
                    'scores': kept_scores.cpu().numpy(),
                    'masks': masks_np,
                    'num_detections': num_keep
                }
                print(f"   '{prompt}': {num_keep} detections after NMS (max score: {kept_scores.max().item():.3f})")
            else:
                results[query_idx] = {
                    'prompt': prompt,
                    'boxes': None,
                    'scores': None,
                    'masks': None,
                    'num_detections': 0
                }
                print(f"   '{prompt}': 0 detections")

        # Store original image for visualization
        results['_image'] = pil_image

        return results

    def visualize(
        self,
        results: dict,
        output_path: str,
        show_boxes: bool = True,
        show_masks: bool = True
    ):
        """
        Visualize predictions on image.

        Args:
            results: Results from predict()
            output_path: Where to save visualization
            show_boxes: Whether to show bounding boxes
            show_masks: Whether to show segmentation masks
        """
        pil_image = results['_image']

        # Create figure
        fig, ax = plt.subplots(1, figsize=(12, 8))
        ax.imshow(pil_image)

        # Colors for different prompts
        colors = ['red', 'blue', 'green', 'yellow', 'cyan', 'magenta']

        total_detections = 0

        # Draw results for each prompt
        for idx in sorted([k for k in results.keys() if k != '_image']):
            result = results[idx]
            prompt = result['prompt']
            color = colors[idx % len(colors)]

            if result['num_detections'] == 0:
                continue

            total_detections += result['num_detections']

            boxes = result['boxes']
            scores = result['scores']
            masks = result['masks']

            for i in range(result['num_detections']):
                # Draw mask
                if show_masks and masks is not None:
                    mask = masks[i]
                    colored_mask = np.zeros((*mask.shape, 4))
                    # Use different colors for different prompts
                    if color == 'red':
                        colored_mask[mask] = [1, 0, 0, 0.4]
                    elif color == 'blue':
                        colored_mask[mask] = [0, 0, 1, 0.4]
                    elif color == 'green':
                        colored_mask[mask] = [0, 1, 0, 0.4]
                    else:
                        colored_mask[mask] = [1, 1, 0, 0.4]
                    ax.imshow(colored_mask)

                # Draw box
                if show_boxes and boxes is not None:
                    box = boxes[i]  # [x1, y1, x2, y2]
                    x1, y1, x2, y2 = box

                    # Clamp to image bounds
                    img_w, img_h = pil_image.size
                    x1 = max(0, min(img_w, x1))
                    y1 = max(0, min(img_h, y1))
                    x2 = max(0, min(img_w, x2))
                    y2 = max(0, min(img_h, y2))

                    width = x2 - x1
                    height = y2 - y1

                    # Draw rectangle
                    rect = patches.Rectangle(
                        (x1, y1), width, height,
                        linewidth=2,
                        edgecolor=color,
                        facecolor='none'
                    )
                    ax.add_patch(rect)

                    # Add label
                    score = scores[i] if scores is not None else 0
                    label = f"{prompt}: {score:.2f}"
                    ax.text(
                        x1, y1 - 5,
                        label,
                        bbox=dict(facecolor=color, alpha=0.5),
                        fontsize=10,
                        color='white'
                    )

        ax.axis('off')

        # Add title with all prompts
        prompts_str = ", ".join([f'"{results[k]["prompt"]}"' for k in sorted([k for k in results.keys() if k != '_image'])])
        plt.suptitle(f'Text Prompts: {prompts_str}', fontsize=12, y=0.98)

        plt.tight_layout()
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        plt.close()

        print(f"\n✅ Saved visualization to {output_path}")
        print(f"   Total detections: {total_detections}")

    def evaluate_dataset(
            self,
            image_dir,
            coco_json,
            prompt,
            save_dir
    ):
        """
        Evaluate SAM3 LoRA on a COCO-format dataset.

        Args:
            image_dir: directory of images
            coco_json: COCO annotation json
            prompt: category name / text prompt
            save_dir: directory to save visualizations
        """

        os.makedirs(save_dir, exist_ok=True)

        coco = COCO(coco_json)

        category_id = None

        for cat in coco.dataset["categories"]:
            if cat["name"].lower() == prompt.lower():
                category_id = cat["id"]
                break

        if category_id is None:
            raise ValueError(
                f"Category '{prompt}' not found in {coco_json}"
            )

        print(f"\nEvaluating category: {prompt}")
        print(f"Category ID: {category_id}")

        dscs = []
        ious = []
        cgf1s = []

        coco_metric = CocoMetricAccumulator()

        valid_images = 0
        skipped_images = 0

        images = coco.dataset["images"]

        for image_info in images:

            image_id = image_info["id"]

            file_name = image_info["file_name"]

            image_path = os.path.join(
                image_dir,
                file_name
            )

            if not os.path.exists(image_path):
                print(
                    f"Skip {file_name}: image not found"
                )

                continue

            ann_ids = coco.getAnnIds(
                imgIds=[image_id],
                catIds=[category_id]
            )

            anns = coco.loadAnns(ann_ids)

            #
            # GT中不存在对应类别
            #
            if len(anns) == 0:
                skipped_images += 1

                print(
                    f"Skip {file_name}: no '{prompt}' in GT"
                )

                continue

            h = image_info["height"]
            w = image_info["width"]

            #
            # Merge all GT instances
            #
            gt_mask = np.zeros(
                (h, w),
                dtype=np.uint8
            )

            for ann in anns:

                try:

                    gt_instance = coco.annToMask(ann)

                    gt_mask = np.logical_or(
                        gt_mask,
                        gt_instance
                    )

                except Exception as e:

                    print(
                        f"Warning: failed to parse GT annotation in {file_name}: {e}"
                    )

            gt_mask = gt_mask.astype(np.uint8)

            #
            # Safety check
            #
            if gt_mask.sum() == 0:
                skipped_images += 1

                print(
                    f"Skip {file_name}: empty GT mask"
                )

                continue

            try:

                #
                # Inference
                #
                results = self.predict(
                    image_path,
                    [prompt]
                )

            except Exception as e:

                print(
                    f"Error during inference on {file_name}: {e}"
                )

                continue

            pred_masks = results[0]["masks"]
            pred_scores = results[0]["scores"]

            #
            # Merge predicted masks
            #
            pred_mask = merge_masks(pred_masks)

            if pred_mask is None:

                pred_mask = np.zeros(
                    (h, w),
                    dtype=np.uint8
                )

            else:

                pred_mask = pred_mask.astype(np.uint8)

            #
            # Metrics
            #
            dsc = compute_dsc(
                pred_mask,
                gt_mask
            )

            iou = compute_iou(
                pred_mask,
                gt_mask
            )

            cgf1 = compute_cgf1(
                pred_mask,
                gt_mask
            )

            if dsc is not None:
                dscs.append(dsc)

            if iou is not None:
                ious.append(iou)

            if cgf1 is not None:
                cgf1s.append(cgf1)

            #
            # COCO mAP accumulation
            #
            try:

                coco_metric.add_sample(
                    gt_mask,
                    pred_masks,
                    pred_scores
                )

            except Exception as e:

                print(
                    f"Warning: COCO metric accumulation failed on {file_name}: {e}"
                )

            #
            # Save visualization
            #
            try:

                output_file = os.path.join(
                    save_dir,
                    Path(file_name).stem + "_pred.png"
                )

                self.visualize(
                    results,
                    output_file,
                    show_boxes=False,
                    show_masks=True
                )

            except Exception as e:

                print(
                    f"Warning: visualization failed for {file_name}: {e}"
                )

            valid_images += 1

            print(
                f"[{valid_images}] "
                f"{file_name} "
                f"DSC={dsc:.4f} "
                f"IoU={iou:.4f} "
                f"cgF1={cgf1:.4f}"
            )

        #
        # Final statistics
        #
        if valid_images == 0:
            print(
                "\nNo valid images found for evaluation."
            )

            return

        try:

            map5095 = coco_metric.evaluate()

        except Exception as e:

            print(
                f"COCO evaluation failed: {e}"
            )

            map5095 = 0.0

        print("\n" + "=" * 60)
        print("Evaluation Summary")
        print("=" * 60)

        print(
            f"Prompt Category      : {prompt}"
        )

        print(
            f"Valid Images         : {valid_images}"
        )

        print(
            f"Skipped Images       : {skipped_images}"
        )

        print(
            f"Mean DSC             : {np.mean(dscs):.4f}"
        )

        print(
            f"Mean IoU             : {np.mean(ious):.4f}"
        )

        print(
            f"Mean cgF1            : {np.mean(cgf1s):.4f}"
        )

        print(
            f"COCO mAP@[0.5:0.95]  : {map5095:.4f}"
        )

        print("=" * 60)

        return {
            "valid_images": valid_images,
            "skipped_images": skipped_images,
            "mean_dsc": float(np.mean(dscs)),
            "mean_iou": float(np.mean(ious)),
            "mean_cgf1": float(np.mean(cgf1s)),
            "coco_map": float(map5095)
        }




def main():
    parser = argparse.ArgumentParser(description="SAM3 + LoRA Inference")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training config YAML"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to LoRA weights (auto-detected if not provided)"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs='+',
        default=["object"],
        help='Text prompt(s) to guide segmentation (e.g., "crack" or "crack" "defect")'
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1008,
        help="Input resolution (default: 1008)"
    )
    parser.add_argument(
        "--boundingbox",
        type=lambda x: x.lower() in ('true', '1', 'yes'),
        default=False,
        help="Show bounding boxes: True or False (default: False)"
    )
    parser.add_argument(
        "--no-masks",
        action="store_true",
        help="Don't show segmentation masks"
    )
    parser.add_argument(
        "--nms-iou",
        type=float,
        default=0.5,
        help="NMS IoU threshold (default: 0.5, lower = fewer overlapping boxes)"
    )
    parser.add_argument(
        "--eval-image-dir",
        type=str,
        default=None
    )

    parser.add_argument(
        "--coco-json",
        type=str,
        default=None
    )

    parser.add_argument(
        "--save-dir",
        type=str,
        default="eval_results"
    )

    parser.add_argument(
        "--top1",
        type=bool,
        default=False
    )


    args = parser.parse_args()

    # Initialize model
    inferencer = SAM3LoRAInference(
        config_path=args.config,
        weights_path=args.weights,
        resolution=args.resolution,
        detection_threshold=args.threshold,
        nms_iou_threshold=args.nms_iou,
        keep_top1=args.top1
    )

    if args.eval_image_dir is not None:
        inferencer.evaluate_dataset(
            image_dir=args.eval_image_dir,
            coco_json=args.coco_json,
            prompt=" ".join(args.prompt),
            save_dir=args.save_dir
        )



if __name__ == "__main__":
    main()
