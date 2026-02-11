import os
import shutil
import tempfile
import subprocess
import numpy as np
import nibabel as nib


def _save_nifti_single_slice(image_hu, spacing, out_path):
    # image_hu: HxW
    vol = image_hu.astype(np.float32)[:, :, None]
    # PixelSpacing is (row, col) in mm. Use (col, row, z) for affine diag.
    affine = np.diag([float(spacing[1]), float(spacing[0]), 1.0, 1.0])
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, out_path)


def predict_mask(
    image_hu,
    spacing,
    use_nnunet=False,
    label_id=1,
    dataset_id="701",
    config="3d_fullres",
    trainer="nnUNetTrainer",
    folds="all",
    device="cpu",
):
    if not use_nnunet:
        return np.zeros_like(image_hu, dtype=np.uint8)

    if shutil.which("nnUNetv2_predict") is None:
        raise RuntimeError("nnUNetv2_predict not found in PATH")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        input_path = os.path.join(input_dir, "case_0000.nii.gz")
        _save_nifti_single_slice(image_hu, spacing, input_path)

        cmd = [
            "nnUNetv2_predict",
            "-i",
            input_dir,
            "-o",
            output_dir,
            "-d",
            str(dataset_id),
            "-c",
            str(config),
            "-tr",
            str(trainer),
            "-f",
            str(folds),
            "--disable_tta",
        ]

        if device == "cpu":
            cmd += ["-device", "cpu"]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nnUNetv2_predict failed")

        pred_files = [f for f in os.listdir(output_dir) if f.endswith(".nii.gz")]
        if not pred_files:
            raise RuntimeError("No prediction output found")

        pred_path = os.path.join(output_dir, pred_files[0])
        seg = nib.load(pred_path).get_fdata()
        seg = np.squeeze(seg).astype(np.int32)

        mask = (seg == int(label_id)).astype(np.uint8)
        return mask
