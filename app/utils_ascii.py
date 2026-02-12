import numpy as np
import pydicom
from PIL import Image
import nibabel as nib
import tempfile
import os
import io


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


def overlay_label_map(image_uint8, label_map, color_map, alpha=0.4):
    # image_uint8: HxW uint8
    # label_map: HxW integer labels
    base = Image.fromarray(image_uint8).convert("RGB")
    out = base.copy()
    arr = np.array(out, dtype=np.uint8)
    lm = label_map.astype(np.int32)

    for label_id, color in color_map.items():
        region = lm == int(label_id)
        if not np.any(region):
            continue
        c = np.array(color, dtype=np.float32)
        a = float(alpha)
        arr_region = arr[region].astype(np.float32)
        arr[region] = np.clip((1.0 - a) * arr_region + a * c, 0, 255).astype(np.uint8)

    return Image.fromarray(arr, mode="RGB")


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


def labelmap_to_nifti_bytes(label_map, spacing):
    vol = label_map.astype(np.uint8)[:, :, None]
    affine = np.diag([float(spacing[1]), float(spacing[0]), 1.0, 1.0])
    img = nib.Nifti1Image(vol, affine)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()

    os.remove(tmp_path)
    return data


def image_to_nifti_bytes(image, spacing):
    vol = image.astype(np.float32)[:, :, None]
    affine = np.diag([float(spacing[1]), float(spacing[0]), 1.0, 1.0])
    img = nib.Nifti1Image(vol, affine)

    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()

    os.remove(tmp_path)
    return data


def load_nifti_volume(file_bytes):
    # Read NIfTI from uploaded bytes and return canonical (X, Y, Z) volume.
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        nii = nib.load(tmp_path)
        nii = nib.as_closest_canonical(nii)
        data = nii.get_fdata(dtype=np.float32)
        affine = nii.affine.copy()
        zooms = nii.header.get_zooms()
    finally:
        os.remove(tmp_path)

    if data.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI volume, got shape {data.shape}")

    # For axial slices data[:, :, k], pixel spacing is (row_mm, col_mm) = (Y, X)
    spacing_row_col = (float(zooms[1]), float(zooms[0]))
    return data, spacing_row_col, affine


def labelmap_volume_to_nifti_bytes(label_volume, affine):
    img = nib.Nifti1Image(label_volume.astype(np.uint8), affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
        nib.save(img, tmp.name)
        tmp_path = tmp.name

    with open(tmp_path, "rb") as f:
        data = f.read()

    os.remove(tmp_path)
    return data
