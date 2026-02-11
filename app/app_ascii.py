import io
import os
import hashlib
import csv
import zipfile
from pathlib import Path
from datetime import datetime
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from utils_ascii import (
    load_dicom,
    load_nifti_volume,
    window_to_uint8,
    compute_area_cm2,
    overlay_mask,
    mask_to_pil,
    mask_to_nifti_bytes,
    labelmap_to_nifti_bytes,
    labelmap_volume_to_nifti_bytes,
    image_to_nifti_bytes,
)
from infer_ascii import predict_mask


AREA_LOG_FILE = Path(__file__).resolve().parents[1] / "data" / "area_log.csv"


def make_case_name():
    return datetime.now().strftime("L3-seg-%Y%m%d-%H%M%S")


def get_or_create_case_name(source_key):
    if st.session_state.get("current_source_key") != source_key:
        st.session_state["current_source_key"] = source_key
        st.session_state["current_case_name"] = make_case_name()
    return st.session_state.get("current_case_name", make_case_name())


def append_area_log_row(row_dict):
    AREA_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "timestamp",
        "case_id",
        "input_type",
        "source_file",
        "slice_index_z",
        "total_area_cm2",
        "muscle1_area_cm2",
        "muscle2_area_cm2",
        "muscle3_area_cm2",
        "total_mean_hu",
        "total_std_hu",
        "psoas_mean_hu",
        "psoas_std_hu",
        "paraspinal_mean_hu",
        "paraspinal_std_hu",
        "abdominal_wall_mean_hu",
        "abdominal_wall_std_hu",
        "window_center",
        "window_width",
        "rotation_deg",
        "rl_flip",
    ]
    if AREA_LOG_FILE.exists():
        # Upgrade old CSV schema in-place so headers always include latest columns.
        with AREA_LOG_FILE.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fieldnames = reader.fieldnames or []
            existing_rows = list(reader)
        if old_fieldnames != fieldnames:
            with AREA_LOG_FILE.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for r in existing_rows:
                    new_row = {k: r.get(k, "") for k in fieldnames}
                    writer.writerow(new_row)
    else:
        with AREA_LOG_FILE.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    with AREA_LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row_dict)


def compute_hu_stats(image_hu, binary_mask):
    vals = image_hu[binary_mask > 0]
    if vals.size == 0:
        return None, None
    return float(np.mean(vals)), float(np.std(vals))


def build_area_log_row(
    case_name,
    input_type,
    source_file_name,
    volume_slice_idx,
    area_cm2,
    muscle1_area,
    muscle2_area,
    muscle3_area,
    total_mean_hu,
    total_std_hu,
    psoas_mean_hu,
    psoas_std_hu,
    paraspinal_mean_hu,
    paraspinal_std_hu,
    abdominal_wall_mean_hu,
    abdominal_wall_std_hu,
    window_center,
    window_width,
    rotation_deg,
    flip_rl,
):
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "case_id": case_name,
        "input_type": input_type,
        "source_file": source_file_name,
        "slice_index_z": "" if volume_slice_idx is None else int(volume_slice_idx),
        "total_area_cm2": f"{float(area_cm2):.4f}",
        "muscle1_area_cm2": f"{muscle1_area:.4f}",
        "muscle2_area_cm2": f"{muscle2_area:.4f}",
        "muscle3_area_cm2": f"{muscle3_area:.4f}",
        "total_mean_hu": "" if total_mean_hu is None else f"{total_mean_hu:.4f}",
        "total_std_hu": "" if total_std_hu is None else f"{total_std_hu:.4f}",
        "psoas_mean_hu": "" if psoas_mean_hu is None else f"{psoas_mean_hu:.4f}",
        "psoas_std_hu": "" if psoas_std_hu is None else f"{psoas_std_hu:.4f}",
        "paraspinal_mean_hu": "" if paraspinal_mean_hu is None else f"{paraspinal_mean_hu:.4f}",
        "paraspinal_std_hu": "" if paraspinal_std_hu is None else f"{paraspinal_std_hu:.4f}",
        "abdominal_wall_mean_hu": "" if abdominal_wall_mean_hu is None else f"{abdominal_wall_mean_hu:.4f}",
        "abdominal_wall_std_hu": "" if abdominal_wall_std_hu is None else f"{abdominal_wall_std_hu:.4f}",
        "window_center": int(window_center),
        "window_width": int(window_width),
        "rotation_deg": int(rotation_deg),
        "rl_flip": bool(flip_rl),
    }


def make_zip_payload(file_entries):
    # file_entries: list of (filename, bytes)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, payload in file_entries:
            zf.writestr(name, payload)
    return buf.getvalue()


def unrotate_2d(arr, k):
    # Inverse of np.rot90(arr, k=k)
    return np.rot90(arr, k=(-k) % 4)


def unflip_lr_2d(arr, flipped):
    if flipped:
        return np.fliplr(arr)
    return arr


st.set_page_config(page_title="L3 Muscle Segmentation", layout="wide")

st.title("L3 Skeletal Muscle Segmentation (MVP)")

st.sidebar.header("Display")
window_center = st.sidebar.slider("Window center", min_value=-1000, max_value=1000, value=50)
window_width = st.sidebar.slider("Window width", min_value=50, max_value=2000, value=400)
rotation_deg = st.sidebar.selectbox("Rotate view", [0, 90, 180, 270], index=0)
flip_rl = st.sidebar.checkbox("R-L flip", value=False)

st.sidebar.header("Mode")
mode = st.sidebar.radio("Segmentation mode", ["Manual annotation", "nnUNet inference"])
input_type = st.sidebar.radio("Input type", ["DICOM slice", "NIfTI volume (.nii.gz)"], index=1)

if input_type == "DICOM slice":
    uploaded = st.file_uploader("Upload L3 axial CT DICOM (extension not required)", type=None)
else:
    uploaded = st.file_uploader("Upload AMOS22 volume (.nii.gz)", type=["nii", "gz"])

if uploaded is not None:
    bytes_data = uploaded.getvalue()
    volume_data = None
    volume_affine = None
    volume_slice_idx = None
    case_name = make_case_name()
    source_file_name = uploaded.name

    if input_type == "DICOM slice":
        dicom_path = io.BytesIO(bytes_data)
        image_hu, spacing, ds = load_dicom(dicom_path)
        source_key = f"DICOM::{hashlib.sha1(bytes_data).hexdigest()[:16]}"
        case_name = get_or_create_case_name(source_key)
        source_id = f"DICOM::{uploaded.name}"
    else:
        volume_key = hashlib.sha1(bytes_data).hexdigest()[:16]
        source_key = f"NIFTI::{volume_key}"
        case_name = get_or_create_case_name(source_key)

        volume_data, spacing, volume_affine = load_nifti_volume(bytes_data)
        max_slice = int(volume_data.shape[2] - 1)
        volume_sig = f"{uploaded.name}:{uploaded.size}:{volume_data.shape}"
        if st.session_state.get("nifti_sig") != volume_sig:
            st.session_state["nifti_sig"] = volume_sig
            st.session_state["nifti_z_idx"] = max_slice // 2
        st.subheader("Volume navigation")
        nav_c1, nav_c2, nav_c3 = st.columns([1, 1, 6])
        with nav_c1:
            if st.button("Prev slice"):
                st.session_state["nifti_z_idx"] = min(max_slice, int(st.session_state["nifti_z_idx"]) + 1)
        with nav_c2:
            if st.button("Next slice"):
                st.session_state["nifti_z_idx"] = max(0, int(st.session_state["nifti_z_idx"]) - 1)
        volume_slice_idx = st.slider(
            "Axial slice index (Z)",
            min_value=0,
            max_value=max_slice,
            key="nifti_z_idx",
        )
        image_hu = volume_data[:, :, volume_slice_idx]
        source_id = f"NIFTI::{uploaded.name}::Z{volume_slice_idx}"
        st.caption(f"Case ID: {case_name}")

    rot_k = int(rotation_deg // 90)
    if rot_k:
        image_hu = np.rot90(image_hu, k=rot_k)
    if flip_rl:
        image_hu = np.fliplr(image_hu)

    image_u8 = window_to_uint8(image_hu, center=window_center, width=window_width)
    image_pil = Image.fromarray(image_u8).convert("L")

    mask = None

    if mode == "Manual annotation":
        st.sidebar.header("Annotation")
        brush_size = st.sidebar.slider("Brush size", min_value=2, max_value=50, value=12)
        brush_alpha = st.sidebar.slider("Brush transparency", min_value=0.05, max_value=0.9, value=0.25)
        canvas_max = 900
        tool = st.sidebar.selectbox("Tool", ["Draw", "Erase"], index=0)
        update_mode = st.sidebar.selectbox("Preview update", ["Realtime", "On submit"], index=1)
        label_options = [
            ("Psoas", 1, (255, 0, 0)),
            ("Paraspinal", 2, (0, 255, 0)),
            ("Abdominal_Wall", 3, (0, 128, 255)),
        ]
        label_name = st.sidebar.selectbox("Active label", [x[0] for x in label_options], index=0)
        label_id = next(x[1] for x in label_options if x[0] == label_name)
        rgb = next(x[2] for x in label_options if x[0] == label_name)
        if tool == "Erase":
            stroke_color = f"rgba(255, 255, 0, {brush_alpha})"
            fill_color = f"rgba(255, 255, 0, {brush_alpha})"
        else:
            stroke_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {brush_alpha})"
            fill_color = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {brush_alpha})"

        st.subheader("Annotation")
        # Downscale large images for canvas stability
        h, w = image_u8.shape
        scale = float(canvas_max) / float(max(h, w))
        scale = min(scale, 1.5)
        disp_w = max(1, int(round(w * scale)))
        disp_h = max(1, int(round(h * scale)))
        bg_img = Image.fromarray(image_u8).convert("RGB").resize((disp_w, disp_h), Image.BILINEAR)
        st.caption(f"Canvas display size: {disp_w}x{disp_h} (scale {scale:.2f}x)")

        if (
            "label_masks" not in st.session_state
            or st.session_state.get("label_shape") != (h, w)
            or st.session_state.get("source_id") != source_id
        ):
            st.session_state["label_masks"] = {
                1: np.zeros((h, w), dtype=np.uint8),
                2: np.zeros((h, w), dtype=np.uint8),
                3: np.zeros((h, w), dtype=np.uint8),
            }
            st.session_state["label_shape"] = (h, w)
            st.session_state["source_id"] = source_id
            st.session_state["canvas_key"] = 0
            st.session_state["last_canvas_hash"] = None
            st.session_state["pending_draw_small_by_label"] = {
                1: np.zeros((disp_h, disp_w), dtype=np.uint8),
                2: np.zeros((disp_h, disp_w), dtype=np.uint8),
                3: np.zeros((disp_h, disp_w), dtype=np.uint8),
            }
            st.session_state["pending_erase_small_by_label"] = {
                1: np.zeros((disp_h, disp_w), dtype=np.uint8),
                2: np.zeros((disp_h, disp_w), dtype=np.uint8),
                3: np.zeros((disp_h, disp_w), dtype=np.uint8),
            }
            st.session_state["pending_shape"] = (disp_h, disp_w)
        elif st.session_state.get("pending_shape") != (disp_h, disp_w):
            st.session_state["pending_draw_small_by_label"] = {
                1: np.zeros((disp_h, disp_w), dtype=np.uint8),
                2: np.zeros((disp_h, disp_w), dtype=np.uint8),
                3: np.zeros((disp_h, disp_w), dtype=np.uint8),
            }
            st.session_state["pending_erase_small_by_label"] = {
                1: np.zeros((disp_h, disp_w), dtype=np.uint8),
                2: np.zeros((disp_h, disp_w), dtype=np.uint8),
                3: np.zeros((disp_h, disp_w), dtype=np.uint8),
            }
            st.session_state["pending_shape"] = (disp_h, disp_w)
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=brush_size,
            stroke_color=stroke_color,
            background_color="rgba(0, 0, 0, 0)",
            background_image=bg_img,
            update_streamlit=True,
            height=disp_h,
            width=disp_w,
            drawing_mode="freedraw",
            key=f"canvas_{st.session_state['canvas_key']}",
        )

        if canvas_result.image_data is not None:
            # Alpha channel > 0 means drawn
            mask_small = (canvas_result.image_data[:, :, 3] > 0).astype(np.uint8)
            if update_mode == "Realtime":
                if scale != 1.0:
                    mask = np.array(
                        Image.fromarray(mask_small).resize((w, h), Image.NEAREST)
                    ).astype(np.uint8)
                else:
                    mask = mask_small
                cur_hash = hash(mask.tobytes())
                if cur_hash != st.session_state.get("last_canvas_hash"):
                    if tool == "Draw":
                        st.session_state["label_masks"][label_id] = np.maximum(
                            st.session_state["label_masks"][label_id], mask
                        ).astype(np.uint8)
                    else:
                        m = st.session_state["label_masks"][label_id]
                        m[mask > 0] = 0
                        st.session_state["label_masks"][label_id] = m
                    st.session_state["last_canvas_hash"] = cur_hash
                    # Clear canvas after applying stroke
                    st.session_state["canvas_key"] += 1
            else:
                if tool == "Draw":
                    st.session_state["pending_draw_small_by_label"][label_id] = np.maximum(
                        st.session_state["pending_draw_small_by_label"][label_id], mask_small
                    ).astype(np.uint8)
                else:
                    st.session_state["pending_erase_small_by_label"][label_id] = np.maximum(
                        st.session_state["pending_erase_small_by_label"][label_id], mask_small
                    ).astype(np.uint8)
        else:
            mask = np.zeros_like(image_u8, dtype=np.uint8)

        if update_mode == "On submit":
            if st.button("Apply annotations"):
                for lid in [1, 2, 3]:
                    pm_draw = st.session_state["pending_draw_small_by_label"][lid]
                    pm_erase = st.session_state["pending_erase_small_by_label"][lid]
                    if scale != 1.0:
                        mask_draw = np.array(
                            Image.fromarray(pm_draw).resize((w, h), Image.NEAREST)
                        ).astype(np.uint8)
                        mask_erase = np.array(
                            Image.fromarray(pm_erase).resize((w, h), Image.NEAREST)
                        ).astype(np.uint8)
                    else:
                        mask_draw = pm_draw
                        mask_erase = pm_erase

                    if mask_draw.sum() > 0:
                        st.session_state["label_masks"][lid] = np.maximum(
                            st.session_state["label_masks"][lid], mask_draw
                        ).astype(np.uint8)
                    if mask_erase.sum() > 0:
                        m = st.session_state["label_masks"][lid]
                        m[mask_erase > 0] = 0
                        st.session_state["label_masks"][lid] = m

                st.session_state["pending_draw_small_by_label"] = {
                    1: np.zeros((disp_h, disp_w), dtype=np.uint8),
                    2: np.zeros((disp_h, disp_w), dtype=np.uint8),
                    3: np.zeros((disp_h, disp_w), dtype=np.uint8),
                }
                st.session_state["pending_erase_small_by_label"] = {
                    1: np.zeros((disp_h, disp_w), dtype=np.uint8),
                    2: np.zeros((disp_h, disp_w), dtype=np.uint8),
                    3: np.zeros((disp_h, disp_w), dtype=np.uint8),
                }
                st.session_state["last_canvas_hash"] = None
                st.session_state["canvas_key"] += 1

        label_map = np.zeros((h, w), dtype=np.uint8)
        for lid, m in st.session_state["label_masks"].items():
            label_map[m > 0] = lid
        mask = (label_map > 0).astype(np.uint8)

    else:
        st.sidebar.header("nnUNet")
        st.sidebar.caption("Uses your nnUNet results folder; muscle class is not present in AMOS labels.")
        dataset_id = st.sidebar.text_input("Dataset ID", value="701")
        config = st.sidebar.text_input("Config", value="3d_fullres")
        trainer = st.sidebar.text_input("Trainer", value="nnUNetTrainer")
        folds = st.sidebar.text_input("Folds", value="all")
        label_id = st.sidebar.number_input("Label ID", min_value=0, max_value=200, value=1)
        device = st.sidebar.selectbox("Device", ["cpu", "cuda"], index=0)

        if "nnunet_mask" not in st.session_state:
            st.session_state["nnunet_mask"] = None

        if st.button("Run nnUNet inference"):
            with st.spinner("Running nnUNet inference..."):
                try:
                    mask = predict_mask(
                        image_hu,
                        spacing,
                        use_nnunet=True,
                        label_id=int(label_id),
                        dataset_id=str(dataset_id),
                        config=str(config),
                        trainer=str(trainer),
                        folds=str(folds),
                        device=str(device),
                    )
                    st.session_state["nnunet_mask"] = mask
                except Exception as e:
                    st.error(f"Inference failed: {e}")

        if st.session_state["nnunet_mask"] is None:
            mask = np.zeros_like(image_u8, dtype=np.uint8)
        else:
            mask = st.session_state["nnunet_mask"]

    area_cm2 = compute_area_cm2(mask, spacing)
    overlay = overlay_mask(image_u8, mask)

    st.subheader("Preview")
    col1, col2 = st.columns(2)
    with col1:
        st.caption("CT slice")
        st.image(image_pil)
    with col2:
        st.caption("Overlay (mask)")
        st.image(overlay)

    st.subheader("Results")
    st.write(f"Total muscle area: {area_cm2:.2f} cm^2")
    muscle1_area = 0.0
    muscle2_area = 0.0
    muscle3_area = 0.0
    total_mean_hu = None
    total_std_hu = None
    psoas_mean_hu = None
    psoas_std_hu = None
    paraspinal_mean_hu = None
    paraspinal_std_hu = None
    abdominal_wall_mean_hu = None
    abdominal_wall_std_hu = None
    if mode == "Manual annotation":
        areas = []
        for lid in [1, 2, 3]:
            m = st.session_state["label_masks"].get(lid, np.zeros_like(mask))
            areas.append((lid, compute_area_cm2(m, spacing)))
        muscle1_area = float(areas[0][1])
        muscle2_area = float(areas[1][1])
        muscle3_area = float(areas[2][1])
        st.write(f"Psoas area: {muscle1_area:.2f} cm^2")
        st.write(f"Paraspinal area: {muscle2_area:.2f} cm^2")
        st.write(f"Abdominal_Wall area: {muscle3_area:.2f} cm^2")
        total_mean_hu, total_std_hu = compute_hu_stats(image_hu, mask)
        psoas_mean_hu, psoas_std_hu = compute_hu_stats(image_hu, st.session_state["label_masks"].get(1, np.zeros_like(mask)))
        paraspinal_mean_hu, paraspinal_std_hu = compute_hu_stats(image_hu, st.session_state["label_masks"].get(2, np.zeros_like(mask)))
        abdominal_wall_mean_hu, abdominal_wall_std_hu = compute_hu_stats(image_hu, st.session_state["label_masks"].get(3, np.zeros_like(mask)))

        st.caption(f"Area log file: {AREA_LOG_FILE}")
        st.caption("Use the primary Finalize button in Downloads to save CSV + download pair together.")

        if AREA_LOG_FILE.exists():
            st.download_button(
                "Download area log CSV",
                data=AREA_LOG_FILE.read_bytes(),
                file_name="area_log.csv",
                mime="text/csv",
            )

    st.subheader("Downloads")
    image_nifti = image_to_nifti_bytes(image_hu, spacing)
    image_file_name = "image_slice.nii.gz"
    label_file_name = "label_map.nii.gz"
    volume_label_file_name = "label_map_volume.nii.gz"
    if input_type == "NIfTI volume (.nii.gz)":
        image_file_name = f"{case_name}-image.nii.gz"
        label_file_name = f"{case_name}-label.nii.gz"
        volume_label_file_name = f"{case_name}-label-volume.nii.gz"

    pair_label_nifti = None
    if mode == "Manual annotation":
        label_map = np.zeros_like(mask, dtype=np.uint8)
        for lid, m in st.session_state["label_masks"].items():
            label_map[m > 0] = lid
        pair_label_nifti = labelmap_to_nifti_bytes(label_map, spacing)
    else:
        # Fallback for nnUNet mode: paired export uses current binary mask.
        pair_label_nifti = mask_to_nifti_bytes(mask, spacing)

    pair_zip = make_zip_payload(
        [
            (image_file_name, image_nifti),
            (label_file_name, pair_label_nifti),
        ]
    )
    st.markdown(
        "<p style='color:#c92a2a; font-weight:700;'>Primary export (image + label pair)</p>",
        unsafe_allow_html=True,
    )
    if mode == "Manual annotation":
        area_row = build_area_log_row(
            case_name=case_name,
            input_type=input_type,
            source_file_name=source_file_name,
            volume_slice_idx=volume_slice_idx,
            area_cm2=area_cm2,
            muscle1_area=muscle1_area,
            muscle2_area=muscle2_area,
            muscle3_area=muscle3_area,
            total_mean_hu=total_mean_hu,
            total_std_hu=total_std_hu,
            psoas_mean_hu=psoas_mean_hu,
            psoas_std_hu=psoas_std_hu,
            paraspinal_mean_hu=paraspinal_mean_hu,
            paraspinal_std_hu=paraspinal_std_hu,
            abdominal_wall_mean_hu=abdominal_wall_mean_hu,
            abdominal_wall_std_hu=abdominal_wall_std_hu,
            window_center=window_center,
            window_width=window_width,
            rotation_deg=rotation_deg,
            flip_rl=flip_rl,
        )
        st.download_button(
            "Finalize case: save CSV + download pair ZIP",
            data=pair_zip,
            file_name=f"{case_name}-pair.zip" if input_type == "NIfTI volume (.nii.gz)" else "image_label_pair.zip",
            mime="application/zip",
            on_click=append_area_log_row,
            args=(area_row,),
        )
    else:
        st.download_button(
            "Download baseline slice + label map (ZIP)",
            data=pair_zip,
            file_name=f"{case_name}-pair.zip" if input_type == "NIfTI volume (.nii.gz)" else "image_label_pair.zip",
            mime="application/zip",
        )

    st.download_button(
        "Download baseline slice (NIfTI)",
        data=image_nifti,
        file_name=image_file_name,
        mime="application/gzip",
    )

    mask_pil = mask_to_pil(mask)

    buf_mask = io.BytesIO()
    mask_pil.save(buf_mask, format="PNG")
    st.download_button("Download mask (PNG)", data=buf_mask.getvalue(), file_name="mask.png", mime="image/png")

    nifti_bytes = mask_to_nifti_bytes(mask, spacing)
    st.download_button("Download mask (NIfTI)", data=nifti_bytes, file_name="mask.nii.gz", mime="application/gzip")

    if mode == "Manual annotation":
        label_nifti = pair_label_nifti
        if "export_idx" not in st.session_state:
            st.session_state["export_idx"] = 1
        idx = st.session_state["export_idx"]
        def _inc_export():
            st.session_state["export_idx"] += 1
        st.download_button(
            "Download label map (NIfTI)",
            data=label_nifti,
            file_name=label_file_name if input_type == "NIfTI volume (.nii.gz)" else f"label_map_{idx:05d}.nii.gz",
            mime="application/gzip",
            on_click=_inc_export,
        )
        if input_type == "NIfTI volume (.nii.gz)" and volume_data is not None and volume_affine is not None:
            full_label_volume = np.zeros(volume_data.shape, dtype=np.uint8)
            label_map_for_volume = label_map.astype(np.uint8)
            label_map_for_volume = unflip_lr_2d(label_map_for_volume, flip_rl)
            if rot_k:
                label_map_for_volume = unrotate_2d(label_map_for_volume, rot_k)
            full_label_volume[:, :, int(volume_slice_idx)] = label_map_for_volume
            full_label_nifti = labelmap_volume_to_nifti_bytes(full_label_volume, volume_affine)
            st.download_button(
                "Download full-volume label map (NIfTI)",
                data=full_label_nifti,
                file_name=volume_label_file_name,
                mime="application/gzip",
            )

    buf_overlay = io.BytesIO()
    overlay.save(buf_overlay, format="PNG")
    st.download_button("Download overlay (PNG)", data=buf_overlay.getvalue(), file_name="overlay.png", mime="image/png")

else:
    st.info("Upload a DICOM file to start.")
