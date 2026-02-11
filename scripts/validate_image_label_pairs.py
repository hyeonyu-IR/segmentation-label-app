import argparse
import csv
import json
import re
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


def _strip_known_suffixes(name: str) -> str:
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return name


def _key_from_name(path: Path, image_prefix: str, label_prefix: str) -> str:
    name = _strip_known_suffixes(path.name)
    for pfx in (image_prefix, label_prefix):
        if pfx and name.startswith(pfx):
            name = name[len(pfx) :]
    # Common cleanup for exported names from the Streamlit app
    name = name.replace("image_slice_", "")
    name = name.replace("image_slice", "")
    return name


def _read_pairs_from_csv(pairs_csv: Path):
    pairs = []
    with pairs_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img = row.get("image")
            lbl = row.get("label")
            if not img or not lbl:
                continue
            pairs.append((Path(img), Path(lbl)))
    return pairs


def _collect_files(folder: Path):
    files = sorted(folder.rglob("*.nii.gz"))
    files += sorted(folder.rglob("*.nii"))
    # Keep unique while preserving order
    seen = set()
    out = []
    for f in files:
        if f.resolve() in seen:
            continue
        seen.add(f.resolve())
        out.append(f)
    return out


def _build_auto_pairs(images_dir: Path, labels_dir: Path, image_prefix: str, label_prefix: str):
    images = _collect_files(images_dir)
    labels = _collect_files(labels_dir)
    img_map = {}
    lbl_map = {}
    for p in images:
        img_map[_key_from_name(p, image_prefix=image_prefix, label_prefix=label_prefix)] = p
    for p in labels:
        lbl_map[_key_from_name(p, image_prefix=image_prefix, label_prefix=label_prefix)] = p

    keys = sorted(set(img_map.keys()) | set(lbl_map.keys()))
    pairs = []
    missing_images = []
    missing_labels = []
    for k in keys:
        img = img_map.get(k)
        lbl = lbl_map.get(k)
        if img is None:
            missing_images.append((k, lbl))
            continue
        if lbl is None:
            missing_labels.append((k, img))
            continue
        pairs.append((img, lbl))
    return pairs, missing_images, missing_labels


def _parse_allowed_labels(s: str):
    vals = []
    for token in s.split(","):
        token = token.strip()
        if token == "":
            continue
        vals.append(int(token))
    return set(vals)


def _check_pair(image_path: Path, label_path: Path, allowed_labels: set, affine_tol: float):
    issues = []
    try:
        img_nii = nib.load(str(image_path))
        lbl_nii = nib.load(str(label_path))
    except Exception as e:
        return [f"load_error: {e}"]

    img = img_nii.get_fdata(dtype=np.float32)
    lbl = lbl_nii.get_fdata(dtype=np.float32)

    if img.shape != lbl.shape:
        issues.append(f"shape_mismatch: image{img.shape} vs label{lbl.shape}")

    if img_nii.affine.shape != lbl_nii.affine.shape:
        issues.append("affine_shape_mismatch")
    else:
        if not np.allclose(img_nii.affine, lbl_nii.affine, atol=affine_tol, rtol=0.0):
            issues.append("affine_mismatch")

    lbl_rounded = np.rint(lbl).astype(np.int64)
    if not np.allclose(lbl, lbl_rounded, atol=1e-6, rtol=0.0):
        issues.append("label_not_integer_valued")

    unique_vals = set(np.unique(lbl_rounded).tolist())
    bad_vals = sorted(v for v in unique_vals if v not in allowed_labels)
    if bad_vals:
        issues.append(f"label_values_out_of_range: {bad_vals}")

    fg_count = int(np.sum(lbl_rounded > 0))
    if fg_count == 0:
        issues.append("empty_label")

    return issues


def main():
    parser = argparse.ArgumentParser(
        description="Validate image/label NIfTI pairs for training readiness."
    )
    parser.add_argument("--images-dir", type=Path, help="Directory containing baseline images.")
    parser.add_argument("--labels-dir", type=Path, help="Directory containing label maps.")
    parser.add_argument(
        "--pairs-csv",
        type=Path,
        help="Optional CSV with columns: image,label. Absolute or relative paths.",
    )
    parser.add_argument(
        "--image-prefix",
        type=str,
        default="",
        help="Prefix to strip from image filenames during automatic pairing.",
    )
    parser.add_argument(
        "--label-prefix",
        type=str,
        default="label_map_",
        help="Prefix to strip from label filenames during automatic pairing.",
    )
    parser.add_argument(
        "--allowed-labels",
        type=str,
        default="0,1,2,3",
        help="Comma-separated allowed label IDs. Example: 0,1,2,3",
    )
    parser.add_argument(
        "--affine-tol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for affine comparison.",
    )
    parser.add_argument(
        "--report-json",
        type=Path,
        help="Optional path to save full validation report JSON.",
    )
    args = parser.parse_args()

    if args.pairs_csv is None:
        if args.images_dir is None or args.labels_dir is None:
            print("ERROR: Provide --pairs-csv or both --images-dir and --labels-dir")
            sys.exit(2)
    else:
        if not args.pairs_csv.exists():
            print(f"ERROR: pairs CSV not found: {args.pairs_csv}")
            sys.exit(2)

    allowed_labels = _parse_allowed_labels(args.allowed_labels)

    missing_images = []
    missing_labels = []
    if args.pairs_csv is not None:
        pairs = _read_pairs_from_csv(args.pairs_csv)
    else:
        if not args.images_dir.exists() or not args.labels_dir.exists():
            print("ERROR: images or labels directory does not exist.")
            sys.exit(2)
        pairs, missing_images, missing_labels = _build_auto_pairs(
            args.images_dir,
            args.labels_dir,
            image_prefix=args.image_prefix,
            label_prefix=args.label_prefix,
        )

    results = []
    total_issues = 0
    for image_path, label_path in pairs:
        issues = _check_pair(image_path, label_path, allowed_labels, args.affine_tol)
        total_issues += len(issues)
        results.append(
            {
                "image": str(image_path),
                "label": str(label_path),
                "status": "ok" if len(issues) == 0 else "fail",
                "issues": issues,
            }
        )

    print(f"pairs_checked: {len(pairs)}")
    if args.pairs_csv is None:
        print(f"missing_images: {len(missing_images)}")
        print(f"missing_labels: {len(missing_labels)}")
    print(f"pairs_failed: {sum(1 for r in results if r['status'] == 'fail')}")
    print(f"total_issues: {total_issues}")

    if missing_images:
        print("sample_missing_images:")
        for k, lbl in missing_images[:5]:
            print(f"  key={k} label={lbl}")
    if missing_labels:
        print("sample_missing_labels:")
        for k, img in missing_labels[:5]:
            print(f"  key={k} image={img}")

    failed = [r for r in results if r["status"] == "fail"]
    if failed:
        print("sample_failures:")
        for r in failed[:10]:
            print(f"  image={r['image']}")
            print(f"  label={r['label']}")
            print(f"  issues={'; '.join(r['issues'])}")

    if args.report_json:
        report = {
            "summary": {
                "pairs_checked": len(pairs),
                "missing_images": len(missing_images),
                "missing_labels": len(missing_labels),
                "pairs_failed": len(failed),
                "total_issues": total_issues,
                "allowed_labels": sorted(allowed_labels),
                "affine_tol": args.affine_tol,
            },
            "missing_images": [
                {"key": k, "label": str(lbl)} for k, lbl in missing_images
            ],
            "missing_labels": [
                {"key": k, "image": str(img)} for k, img in missing_labels
            ],
            "results": results,
        }
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report_json: {args.report_json}")

    # Non-zero exit on any problem
    if total_issues > 0 or len(missing_images) > 0 or len(missing_labels) > 0:
        sys.exit(1)

    print("validation: PASS")
    sys.exit(0)


if __name__ == "__main__":
    main()
