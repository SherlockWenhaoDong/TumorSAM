#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from pycocotools.coco import COCO


def visualize_gt(
        image_path,
        gt_mask,
        save_path,
        alpha=0.4
):
    """
    Save image + GT overlay
    """

    image = Image.open(image_path).convert("RGB")

    fig, ax = plt.subplots(
        1,
        figsize=(12, 8)
    )

    ax.imshow(image)

    overlay = np.zeros(
        (*gt_mask.shape, 4),
        dtype=np.float32
    )

    #
    # Red mask
    #
    overlay[gt_mask > 0] = [
        1.0,  # R
        0.0,  # G
        0.0,  # B
        alpha
    ]

    ax.imshow(overlay)

    ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        bbox_inches="tight",
        pad_inches=0,
        dpi=150
    )

    plt.close()


def export_gt_visualization(
        image_dir,
        coco_json,
        category_name,
        save_dir
):

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    coco = COCO(coco_json)

    #
    # find category id
    #
    category_id = None

    for cat in coco.dataset["categories"]:

        if cat["name"].lower() == category_name.lower():

            category_id = cat["id"]
            break

    if category_id is None:

        raise ValueError(
            f"Cannot find category: {category_name}"
        )

    print(
        f"Category: {category_name} "
        f"(id={category_id})"
    )

    saved = 0

    for image_info in coco.dataset["images"]:

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

        h = image_info["height"]
        w = image_info["width"]

        ann_ids = coco.getAnnIds(
            imgIds=[image_id],
            catIds=[category_id]
        )

        anns = coco.loadAnns(
            ann_ids
        )

        if len(anns) == 0:
            continue

        #
        # merge all instances
        #
        gt_mask = np.zeros(
            (h, w),
            dtype=np.uint8
        )

        for ann in anns:

            try:

                gt_mask = np.logical_or(
                    gt_mask,
                    coco.annToMask(ann)
                )

            except Exception as e:

                print(
                    f"Warning {file_name}: {e}"
                )

        gt_mask = gt_mask.astype(
            np.uint8
        )

        if gt_mask.sum() == 0:
            continue

        save_path = os.path.join(
            save_dir,
            Path(file_name).stem + "_gt.png"
        )

        visualize_gt(
            image_path=image_path,
            gt_mask=gt_mask,
            save_path=save_path
        )

        saved += 1

        print(
            f"[{saved}] Saved: "
            f"{save_path}"
        )

    print("\nDone!")
    print(f"Saved {saved} images.")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image-dir",
        required=True
    )

    parser.add_argument(
        "--coco-json",
        required=True
    )

    parser.add_argument(
        "--category",
        required=True
    )

    parser.add_argument(
        "--save-dir",
        default="gt_visualization"
    )

    args = parser.parse_args()

    export_gt_visualization(
        image_dir=args.image_dir,
        coco_json=args.coco_json,
        category_name=args.category,
        save_dir=args.save_dir
    )


if __name__ == "__main__":
    main()