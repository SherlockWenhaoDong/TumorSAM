import os
import json
import numpy as np
import nibabel as nib
import pydicom
from PIL import Image
import cv2
import random


# ---------------------------------------------------------
# Load a DICOM series and sort by InstanceNumber
# Output shape: (Z, H, W)
# ---------------------------------------------------------
def load_dicom_series(dicom_dir):
    dicom_files = []

    for fname in os.listdir(dicom_dir):
        if fname.endswith(".dcm"):
            path = os.path.join(dicom_dir, fname)
            try:
                dcm = pydicom.dcmread(path)
                dicom_files.append(dcm)
            except Exception as e:
                print(f"[Warning] Failed to read DICOM: {path}")

    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    dicom_files = sorted(dicom_files, key=lambda x: int(x.InstanceNumber))

    volume = np.stack([d.pixel_array for d in dicom_files], axis=0)
    return volume


# ---------------------------------------------------------
# Load NIfTI label and convert to (Z, H, W)
# ---------------------------------------------------------
def load_label(label_path):
    nii = nib.load(label_path)
    mask = nii.get_fdata()

    # transpose from (H, W, Z) -> (Z, H, W)
    return np.transpose(mask, (2, 0, 1))


# ---------------------------------------------------------
# Convert binary mask -> COCO annotation (polygon)
# ---------------------------------------------------------
def mask_to_coco_annotations(mask, image_id, ann_id):

    mask = mask.astype(np.uint8)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    annotations = []

    for cnt in contours:
        if len(cnt) < 6:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        segmentation = cnt.flatten().astype(float).tolist()

        annotation = {
            "id": ann_id,
            "image_id": image_id,
            "category_id": 1,  # tumor class
            "bbox": [float(x), float(y), float(w), float(h)],
            "area": float(cv2.contourArea(cnt)),
            "segmentation": [segmentation],
            "iscrowd": 0
        }

        annotations.append(annotation)
        ann_id += 1

    return annotations, ann_id


# ---------------------------------------------------------
# Initialize empty COCO structure
# ---------------------------------------------------------
def create_coco():
    return {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "tumor"}
        ]
    }


# ---------------------------------------------------------
# Process a single case (DICOM + Label)
# ---------------------------------------------------------
def process_case(case, dicom_dir, label_path, split_dir,
                 coco, image_id, ann_id):

    img_dir = os.path.join(split_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    volume = load_dicom_series(dicom_dir)
    mask = load_label(label_path)

    if volume.shape != mask.shape:
        raise ValueError(f"[Error] Shape mismatch in case: {case}")

    Z, H, W = volume.shape

    for z in range(Z):

        image_slice = volume[z]
        mask_slice = mask[z]

        anns, ann_id = mask_to_coco_annotations(mask_slice, image_id, ann_id)

        if len(anns) == 0:
            continue

        image_slice = (image_slice - image_slice.min()) / (
                image_slice.max() - image_slice.min() + 1e-8
        )
        image_slice = (image_slice * 255).astype(np.uint8)

        image_rgb = np.stack([image_slice] * 3, axis=-1)

        filename = f"{case}_{z:04d}.png"
        filepath = os.path.join(img_dir, filename)

        Image.fromarray(image_rgb).save(filepath)

        coco["images"].append({
            "id": image_id,
            "file_name": f"images/{filename}",
            "height": H,
            "width": W
        })

        coco["annotations"].extend(anns)

        image_id += 1

    return image_id, ann_id


# ---------------------------------------------------------
# Split dataset into train / valid / test (by cases)
# ---------------------------------------------------------
def split_cases(cases, train_ratio, val_ratio, seed=42):

    random.seed(seed)
    random.shuffle(cases)

    n = len(cases)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_cases = cases[:n_train]
    val_cases = cases[n_train:n_train + n_val]
    test_cases = cases[n_train + n_val:]

    return train_cases, val_cases, test_cases


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------
def convert_dataset(input_root, label_root, output_root,
                    train_ratio=0.7, val_ratio=0.2):

    cases = [
        c for c in os.listdir(input_root)
        if os.path.isdir(os.path.join(input_root, c))
    ]

    train_cases, val_cases, test_cases = split_cases(
        cases, train_ratio, val_ratio
    )

    splits = {
        "train": train_cases,
        "valid": val_cases,
        "test": test_cases
    }

    print("\nDataset Split Summary:")
    for split_name, case_list in splits.items():
        print(f"  {split_name}: {len(case_list)} cases")

    # Process each split
    for split_name, case_list in splits.items():

        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)

        coco = create_coco()

        image_id = 0
        ann_id = 0

        for case in case_list:

            print(f"[{split_name}] Processing case: {case}")

            dicom_dir = os.path.join(input_root, case)
            label_path = os.path.join(label_root, case + ".nii.gz")

            if not os.path.exists(label_path):
                print(f"[Warning] Missing label for case: {case}")
                continue

            try:
                image_id, ann_id = process_case(
                    case, dicom_dir, label_path,
                    split_dir, coco, image_id, ann_id
                )
            except Exception as e:
                print(f"[Error] Failed processing {case}: {e}")

        json_path = os.path.join(split_dir, "_annotations.coco.json")

        with open(json_path, "w") as f:
            json.dump(coco, f, indent=2)

        print(f"✅ Saved {split_name} annotations -> {json_path}")


# ---------------------------------------------------------
# CLI entry
# ---------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert DICOM + NIfTI dataset to COCO format")
    parser.add_argument("--input", required=True, help="Path to DICOM cases")
    parser.add_argument("--label", required=True, help="Path to NIfTI labels")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)

    args = parser.parse_args()

    convert_dataset(
        args.input,
        args.label,
        args.output,
        args.train_ratio,
        args.val_ratio
    )
