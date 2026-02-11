import numpy as np
import pydicom
from PIL import Image
import nibabel as nib
import tempfile
import os


def load_dicom(path):
    try:
        ds = pydicom.dcmread(path)
    except UnicodeDecodeError:
        try:
            # Some DICOMs contain non-UTF8 metadata. Read minimal tags only.
            ds = pydicom.dcmread(
                path,
                force=True,
                specific_tags=["PixelData", "PixelSpacing", "RescaleSlope", "RescaleIntercept"],
            )
        except UnicodeDecodeError:
            # Last-resort: relax validation to avoid charset decode errors
            prev_allow = getattr(pydicom.config.settings, "allow_invalid_values", None)
            prev_mode = getattr(pydicom.config.settings, "reading_validation_mode", None)
            try:
                if prev_allow is not None:
                    pydicom.config.settings.allow_invalid_values = True
                if prev_mode is not None and hasattr(pydicom.config, "IGNORE"):
                    pydicom.config.settings.reading_validation_mode = pydicom.config.IGNORE

                ds = pydicom.dcmread(
                    path,
                    force=True,
                    specific_tags=["PixelData", "PixelSpacing", "RescaleSlope", "RescaleIntercept"],
                )
            finally:
                if prev_allow is not None:
                    pydicom.config.settings.allow_invalid_values = prev_allow
                if prev_mode is not None:
                    pydicom.config.settings.reading_validation_mode = prev_mode

    img = ds.pixel_array.astype(np.float32)

    # Apply rescale slope/intercept if present
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img * slope + intercept

    # Pixel spacing: (row, col) in mm
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None:
        spacing = (1.0, 1.0)
    else:
        spacing = (float(spacing[0]), float(spacing[1]))

    return img, spacing, ds


def window_to_uint8(img, center=50.0, width=400.0):
    # Standard CT soft tissue window
    low = center - width / 2.0
    high = center + width / 2.0
    img = np.clip(img, low, high)
    img = (img - low) / (high - low + 1e-8)
    img = (img * 255.0).astype(np.uint8)
    return img


def compute_area_cm2(mask, spacing):
    # mask: boolean or 0/1
    mask = mask.astype(np.uint8)
    area_mm2 = float(mask.sum()) * float(spacing[0]) * float(spacing[1])
    area_cm2 = area_mm2 / 100.0
    return area_cm2


def overlay_mask(image_uint8, mask, color=(255, 0, 0), alpha=0.4):
    # image_uint8: HxW uint8
    # mask: HxW bool or 0/1
    base = Image.fromarray(image_uint8).convert("RGB")
    overlay = Image.new("RGB", base.size, color)

    mask_img = (mask.astype(np.uint8) * int(255 * alpha)).astype(np.uint8)
    mask_pil = Image.fromarray(mask_img, mode="L")

    out = Image.composite(overlay, base, mask_pil)
    return out


def mask_to_pil(mask):
    mask_img = (mask.astype(np.uint8) * 255).astype(np.uint8)
    return Image.fromarray(mask_img, mode="L")


def mask_to_nifti_bytes(mask, spacing):
    vol = mask.astype(np.uint8)[:, :, None]
    affine = np.diag([float(spacing[1]), float(spacing[0]), 1.0, 1.0])
    img = nib.Nifti1Image(vol, affine)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()

    os.remove(tmp_path)
    return data
