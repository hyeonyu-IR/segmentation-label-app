import argparse
import json
import shutil
from pathlib import Path


def get_case_key(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        name = name[:-7]
    elif name.endswith(".nii"):
        name = name[:-4]
    for sfx in ("-image", "-label"):
        if name.endswith(sfx):
            name = name[: -len(sfx)]
            break
    return name


def collect_pairs(source_dir: Path):
    image_map = {}
    label_map = {}
    for p in sorted(source_dir.glob("*.nii.gz")):
        key = get_case_key(p)
        if p.name.endswith("-image.nii.gz"):
            image_map[key] = p
        elif p.name.endswith("-label.nii.gz"):
            label_map[key] = p

    keys = sorted(set(image_map.keys()) | set(label_map.keys()))
    pairs = []
    missing_images = []
    missing_labels = []
    for k in keys:
        img = image_map.get(k)
        lbl = label_map.get(k)
        if img is None:
            missing_images.append(k)
            continue
        if lbl is None:
            missing_labels.append(k)
            continue
        pairs.append((k, img, lbl))
    return pairs, missing_images, missing_labels


def write_dataset_json(dataset_root: Path, dataset_name: str, num_training: int):
    ds = {
        "name": dataset_name,
        "description": "L3 skeletal muscle multi-class segmentation (2D slices)",
        "channel_names": {"0": "CT"},
        "labels": {
            "background": 0,
            "Psoas": 1,
            "Paraspinal": 2,
            "Abdominal_Wall": 3,
        },
        "numTraining": int(num_training),
        "file_ending": ".nii.gz",
    }
    out_path = dataset_root / "dataset.json"
    out_path.write_text(json.dumps(ds, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Prepare a 2D nnUNetv2 dataset from paired L3 slice files."
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Folder containing paired '*-image.nii.gz' and '*-label.nii.gz' files.",
    )
    parser.add_argument(
        "--nnunet-raw-dir",
        type=Path,
        required=True,
        help="Path to nnUNet_raw root.",
    )
    parser.add_argument(
        "--dataset-id",
        type=int,
        default=711,
        help="nnUNet dataset ID (default: 711).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="L3SM",
        help="Dataset short name (default: L3SM).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in imagesTr/labelsTr.",
    )
    args = parser.parse_args()

    source_dir = args.source_dir
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    pairs, missing_images, missing_labels = collect_pairs(source_dir)
    print(f"pairs_found: {len(pairs)}")
    print(f"missing_images: {len(missing_images)}")
    print(f"missing_labels: {len(missing_labels)}")
    if missing_images:
        print("sample_missing_images:", missing_images[:10])
    if missing_labels:
        print("sample_missing_labels:", missing_labels[:10])
    if missing_images or missing_labels:
        raise RuntimeError("Pairing incomplete. Fix missing image/label files first.")

    dataset_folder = f"Dataset{int(args.dataset_id):03d}_{args.dataset_name}"
    dataset_root = args.nnunet_raw_dir / dataset_folder
    images_tr = dataset_root / "imagesTr"
    labels_tr = dataset_root / "labelsTr"
    images_tr.mkdir(parents=True, exist_ok=True)
    labels_tr.mkdir(parents=True, exist_ok=True)

    copied = 0
    for case_key, img_path, lbl_path in pairs:
        out_img = images_tr / f"{case_key}_0000.nii.gz"
        out_lbl = labels_tr / f"{case_key}.nii.gz"
        if (out_img.exists() or out_lbl.exists()) and not args.overwrite:
            raise FileExistsError(
                f"Output already exists for case '{case_key}'. Use --overwrite to replace."
            )
        shutil.copy2(img_path, out_img)
        shutil.copy2(lbl_path, out_lbl)
        copied += 1

    ds_json = write_dataset_json(dataset_root, args.dataset_name, copied)
    print(f"dataset_root: {dataset_root}")
    print(f"dataset_json: {ds_json}")
    print(f"cases_copied: {copied}")
    print("done: dataset prepared for nnUNetv2 2D.")


if __name__ == "__main__":
    main()
