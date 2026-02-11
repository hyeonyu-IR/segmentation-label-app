import io
import os
import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from app.utils_main import (
    load_dicom,
    window_to_uint8,
    compute_area_cm2,
    overlay_mask,
    mask_to_pil,
    mask_to_nifti_bytes,
)
from app.infer_main import predict_mask


st.set_page_config(page_title="L3 Muscle Segmentation", layout="centered")

st.title("L3 Skeletal Muscle Segmentation (MVP)")

st.sidebar.header("Display")
window_center = st.sidebar.number_input("Window center", value=50.0)
window_width = st.sidebar.number_input("Window width", value=400.0)

st.sidebar.header("Mode")
mode = st.sidebar.radio("Segmentation mode", ["Manual annotation", "nnUNet inference"])

uploaded = st.file_uploader("Upload L3 axial CT DICOM", type=["dcm"])

if uploaded is not None:
    bytes_data = uploaded.getvalue()
    dicom_path = io.BytesIO(bytes_data)

    image_hu, spacing, ds = load_dicom(dicom_path)
    image_u8 = window_to_uint8(image_hu, center=window_center, width=window_width)

    mask = None

    if mode == "Manual annotation":
        st.sidebar.header("Annotation")
        brush_size = st.sidebar.slider("Brush size", min_value=2, max_value=50, value=12)

        st.subheader("Annotation")
        canvas_result = st_canvas(
            fill_color="rgba(255, 0, 0, 0.3)",
            stroke_width=brush_size,
            stroke_color="rgba(255, 0, 0, 1)",
            background_image=Image.fromarray(image_u8),
            update_streamlit=True,
            height=image_u8.shape[0],
            width=image_u8.shape[1],
            drawing_mode="freedraw",
            key="canvas",
        )

        if canvas_result.image_data is not None:
            # Alpha channel > 0 means drawn
            mask = (canvas_result.image_data[:, :, 3] > 0).astype(np.uint8)
        else:
            mask = np.zeros_like(image_u8, dtype=np.uint8)

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
        st.image(image_u8, clamp=True)
    with col2:
        st.caption("Overlay (mask)")
        st.image(overlay)

    st.subheader("Results")
    st.write(f"Muscle area: {area_cm2:.2f} cm²")

    st.subheader("Downloads")
    mask_pil = mask_to_pil(mask)

    buf_mask = io.BytesIO()
    mask_pil.save(buf_mask, format="PNG")
    st.download_button("Download mask (PNG)", data=buf_mask.getvalue(), file_name="mask.png", mime="image/png")

    nifti_bytes = mask_to_nifti_bytes(mask, spacing)
    st.download_button("Download mask (NIfTI)", data=nifti_bytes, file_name="mask.nii.gz", mime="application/gzip")

    buf_overlay = io.BytesIO()
    overlay.save(buf_overlay, format="PNG")
    st.download_button("Download overlay (PNG)", data=buf_overlay.getvalue(), file_name="overlay.png", mime="image/png")

else:
    st.info("Upload a DICOM file to start.")
