import os
import shutil
import tempfile
import subprocess
from pathlib import Path
import numpy as np
import nibabel as nib


def _save_nifti_single_slice(image_hu, spacing, out_path):
    # image_hu: HxW
    vol = image_hu.astype(np.float32)[:, :, None]
    # PixelSpacing is (row, col) in mm. Use (col, row, z) for affine diag.
    affine = np.diag([float(spacing[1]), float(spacing[0]), 1.0, 1.0])
    img = nib.Nifti1Image(vol, affine)
    nib.save(img, out_path)


def _resolve_nnunet_predict_exe():
    # 1) explicit override
    env_exe = os.environ.get("NNUNETV2_PREDICT_EXE", "").strip().strip('"')
    if env_exe and os.path.isfile(env_exe):
        return env_exe

    # 2) PATH lookup
    exe = shutil.which("nnUNetv2_predict")
    if exe:
        return exe

    # 3) common Windows conda location fallback (user's medimg env)
    fallback = r"C:\Users\hyeon\miniconda3\envs\medimg\Scripts\nnUNetv2_predict.exe"
    if os.path.isfile(fallback):
        return fallback

    raise RuntimeError(
        "nnUNetv2_predict not found. Activate the medimg environment or set "
        "NNUNETV2_PREDICT_EXE to the full executable path."
    )


def _resolve_nnunet_env():
    env = os.environ.copy()
    # Keep existing settings if already defined.
    raw = env.get("nnUNet_raw")
    pre = env.get("nnUNet_preprocessed")
    res = env.get("nnUNet_results")
    if raw and pre and res:
        return env

    # Fallback to this workspace's standard locations.
    root = Path(__file__).resolve().parents[2]  # .../miniconda_medimg_env
    fallback_raw = root / "data" / "nnUNet_raw"
    fallback_pre = root / "data" / "nnUNet_preprocessed"
    fallback_res = root / "data" / "nnUNet_results"
    if fallback_raw.exists() and fallback_pre.exists() and fallback_res.exists():
        env["nnUNet_raw"] = str(fallback_raw)
        env["nnUNet_preprocessed"] = str(fallback_pre)
        env["nnUNet_results"] = str(fallback_res)
    return env


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
    return_labelmap=False,
):
    if not use_nnunet:
        return np.zeros_like(image_hu, dtype=np.uint8)

    nnunet_predict_exe = _resolve_nnunet_predict_exe()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_dir = os.path.join(tmpdir, "inputs")
        output_dir = os.path.join(tmpdir, "outputs")
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        input_path = os.path.join(input_dir, "case_0000.nii.gz")
        _save_nifti_single_slice(image_hu, spacing, input_path)

        cmd = [
            nnunet_predict_exe,
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

        result = subprocess.run(cmd, capture_output=True, text=True, env=_resolve_nnunet_env())
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or "nnUNetv2_predict failed")

        pred_files = [f for f in os.listdir(output_dir) if f.endswith(".nii.gz")]
        if not pred_files:
            raise RuntimeError("No prediction output found")

        pred_path = os.path.join(output_dir, pred_files[0])
        seg = nib.load(pred_path).get_fdata()
        seg = np.squeeze(seg).astype(np.int32)

        if return_labelmap:
            return seg.astype(np.uint8)

        mask = (seg == int(label_id)).astype(np.uint8)
        return mask
