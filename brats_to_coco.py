import os
import json
import numpy as np
import SimpleITK as sitk
import cv2
from PIL import Image
import random


# -----------------------------
# Load .mha volume
# -----------------------------
def load_mha(path):
    img = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(img)  # (Z, H, W)


# -----------------------------
# Find modality inside case
# -----------------------------
def find_modality(case_dir, keyword):
    for sub in os.listdir(case_dir):
        if keyword.lower() in sub.lower():
            sub_path = os.path.join(case_dir, sub)
            for f in os.listdir(sub_path):
                if f.endswith(".mha"):
                    return os.path.join(sub_path, f)
    return None


# -----------------------------
# Convert mask → COCO ann
# -----------------------------
def mask_to_ann(mask, image_id, ann_id):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    anns = []
    for cnt in contours:
        if len(cnt) < 6:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        anns.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(cv2.contourArea(cnt)),
            "segmentation": [cnt.flatten().astype(float).tolist()],
            "iscrowd": 0
        })

        ann_id += 1

    return anns, ann_id


# -----------------------------
# Collect training cases
# -----------------------------
def collect_cases(root):

    cases = []
    train_root = os.path.join(root, "BRATS2015_Training")

    for grade in ["HGG", "LGG"]:
        grade_dir = os.path.join(train_root, grade)
        for case in os.listdir(grade_dir):
            cases.append(os.path.join(grade_dir, case))

    return cases


# -----------------------------
# Split (NO data leakage)
# -----------------------------
def split_cases(cases):

    random.shuffle(cases)
    n = len(cases)

    train = cases[:int(n * 0.7)]
    val   = cases[int(n * 0.7):int(n * 0.85)]
    test  = cases[int(n * 0.85):]

    return train, val, test


# -----------------------------
# Normalize MRI (important!)
# -----------------------------
def normalize(img):
    img = img.astype(np.float32)
    return (img - img.mean()) / (img.std() + 1e-8)


# -----------------------------
# Process one case
# -----------------------------
def process_case(case_dir, img_dir, coco, image_id, ann_id,
                 skip_empty=True):

    flair_path = find_modality(case_dir, "Flair")
    label_path = find_modality(case_dir, "OT")

    if flair_path is None or label_path is None:
        print(f"[WARN] skip {case_dir}")
        return image_id, ann_id

    flair = load_mha(flair_path)
    label = load_mha(label_path)

    Z = flair.shape[0]

    for z in range(Z):

        img = flair[z]
        mask = label[z]

        # ✅ convert multi-class → single tumor
        mask = (mask > 0).astype(np.uint8)

        if skip_empty and mask.sum() == 0:
            continue  # 🔥 huge speedup

        # ✅ MRI normalization（对 SAM3 非常重要）
        img = normalize(img)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        img_rgb = np.stack([img] * 3, axis=-1)

        name = os.path.basename(case_dir) + f"_{z:04d}.png"
        Image.fromarray(img_rgb).save(os.path.join(img_dir, name))

        coco["images"].append({
            "id": image_id,
            "file_name": f"images/{name}",
            "height": img.shape[0],
            "width": img.shape[1]
        })

        anns, ann_id = mask_to_ann(mask, image_id, ann_id)
        coco["annotations"].extend(anns)

        image_id += 1

    return image_id, ann_id


# -----------------------------
# Main pipeline
# -----------------------------
def process_brats(root, output):

    cases = collect_cases(root)

    train, val, test = split_cases(cases)

    split_map = {
        "train": train,
        "val": val,
        "test": test
    }

    print("Split summary:")
    print(f"train: {len(train)} cases")
    print(f"val:   {len(val)} cases")
    print(f"test:  {len(test)} cases")

    for split, case_list in split_map.items():

        print(f"\n=== Processing {split} ===")

        split_dir = os.path.join(output, split)
        img_dir = os.path.join(split_dir, "images")

        os.makedirs(img_dir, exist_ok=True)

        coco = {
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "tumor"}]
        }

        image_id = 0
        ann_id = 0

        for case_dir in case_list:
            print(f"processing {os.path.basename(case_dir)}")

            image_id, ann_id = process_case(
                case_dir,
                img_dir,
                coco,
                image_id,
                ann_id,
                skip_empty=True   # ✅ 强烈推荐
            )

        json_path = os.path.join(split_dir, "_annotations.coco.json")

        with open(json_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"✅ {split} saved: {json_path}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    process_brats(args.input, args.output)