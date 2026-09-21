import os
import json
import numpy as np
from PIL import Image
import cv2
import random


# ---------------------------------------------------------
# mask -> COCO annotation
# ---------------------------------------------------------
def mask_to_ann(mask, image_id, ann_id, category_id):

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anns = []

    for cnt in contours:

        # ✅ 过滤无效 contour
        if len(cnt) < 6:
            continue

        area = cv2.contourArea(cnt)

        # ✅ 去掉太小区域（噪声）
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        # ✅ 防止 whole image bbox（关键修复）
        H, W = mask.shape
        if w * h > 0.9 * H * W:
            continue

        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(area),
            "segmentation": [cnt.flatten().astype(float).tolist()],
            "iscrowd": 0
        })

        ann_id += 1

    return anns, ann_id


# ---------------------------------------------------------
# split dataset
# ---------------------------------------------------------
def split_list(data, train=0.7, val=0.15):

    random.shuffle(data)

    n = len(data)
    n_train = int(n * train)
    n_val = int(n * val)

    return (
        data[:n_train],
        data[n_train:n_train+n_val],
        data[n_train+n_val:]
    )


# ---------------------------------------------------------
# 主处理流程
# ---------------------------------------------------------
def process_dataset(root_dir, output_dir):

    categories = []
    category_map = {}
    cid = 1

    disease_samples = {}

    # -----------------------
    # collect samples
    # -----------------------
    for disease in sorted(os.listdir(root_dir)):

        ddir = os.path.join(root_dir, disease)
        if not os.path.isdir(ddir):
            continue

        category_map[disease] = cid
        categories.append({"id": cid, "name": disease})

        samples = []

        for f in os.listdir(ddir):
            if f.endswith(".png") and "_mask" not in f:

                img_path = os.path.join(ddir, f)
                mask_path = os.path.join(ddir, f.replace(".png", "_mask.png"))

                if os.path.exists(mask_path):
                    samples.append((disease, img_path, mask_path))

        disease_samples[disease] = samples
        print(f"{disease}: {len(samples)} samples")

        cid += 1

    # -----------------------
    # balance classes
    # -----------------------
    min_n = min(len(v) for v in disease_samples.values())
    print(f"\n✅ Balancing each class to {min_n} samples")

    balanced_samples = []
    for disease, samples in disease_samples.items():
        selected = random.sample(samples, min_n)
        balanced_samples.extend(selected)

    # -----------------------
    # split
    # -----------------------
    train, val, test = split_list(balanced_samples)

    split_map = {
        "train": train,
        "val": val,
        "test": test
    }

    # -----------------------
    # generate COCO
    # -----------------------
    for split_name, dataset in split_map.items():

        split_dir = os.path.join(output_dir, split_name)
        img_dir = os.path.join(split_dir, "images")

        os.makedirs(img_dir, exist_ok=True)

        coco = {
            "images": [],
            "annotations": [],
            "categories": categories
        }

        image_id = 0
        ann_id = 0

        for disease, img_path, mask_path in dataset:

            img = Image.open(img_path).convert("RGB")

            # ✅ 正确读取 mask（关键修复）
            mask = np.array(Image.open(mask_path))

            # ✅ 如果是RGB mask
            if mask.ndim == 3:
                mask = mask[:, :, 0]

            # ✅ 正确阈值
            mask = (mask > 127).astype(np.uint8)

            H, W = mask.shape

            # ✅ Debug（可选）
            # print("mask unique:", np.unique(mask), "sum:", mask.sum())

            # ✅ 过滤全图mask（致命问题修复）
            if mask.sum() > 0.9 * H * W:
                continue

            # ✅ 生成 annotation
            anns, ann_id = mask_to_ann(
                mask, image_id, ann_id, category_map[disease]
            )

            # ✅ 没有有效目标 → 丢弃
            if len(anns) == 0:
                continue

            # ✅ 保存 image
            name = f"{disease}_{os.path.basename(img_path)}"
            img.save(os.path.join(img_dir, name))

            coco["images"].append({
                "id": image_id,
                "file_name": f"images/{name}",
                "height": H,
                "width": W
            })

            coco["annotations"].extend(anns)

            image_id += 1

        # save
        with open(os.path.join(split_dir, "_annotations.coco.json"), "w") as f:
            json.dump(coco, f, indent=2)

        print(f"✅ {split_name} done | images: {len(coco['images'])}, anns: {len(coco['annotations'])}")


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    process_dataset(args.input, args.output)