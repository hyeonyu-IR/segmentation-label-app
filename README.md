# Segmentation Label App (Streamlit)

This app is for manual skeletal muscle annotation on single-slice DICOM CT images (L3 level), with area calculation and mask export.
It also supports AMOS22 `.nii.gz` volumes with axial slice selection.

## Current Entry Point

- Main app file: `app/app_ascii.py`
- Launch script: `run_app.ps1`

## Recommended Environment

Use a dedicated conda environment (`segapp`) to avoid dependency conflicts.

```powershell
conda create -y -n segapp python=3.10
conda activate segapp
conda install -y numpy=1.26
pip install streamlit==1.29.0 streamlit-drawable-canvas==0.9.3 pydicom pillow nibabel
```

## Run the App

From `C:\Users\hyeon\Documents\miniconda_medimg_env`:

```powershell
conda activate segapp
python -m streamlit run segmentation-label-app\app\app_ascii.py
```

Alternative (if you `cd segmentation-label-app` first):

```powershell
.\run_app.ps1
```

## Basic Workflow

1. Upload one DICOM slice (file extension is not required).
1. Choose input type in the sidebar:
- `DICOM slice`
- `NIfTI volume (.nii.gz)` for AMOS22 volumes.
1. If using NIfTI volume, use `Prev slice`, `Next slice`, or the `Axial slice index (Z)` slider in the app to choose a slice.
2. Set CT display with `Window center` and `Window width` sliders.
3. Optional: adjust view orientation:
- `Rotate view` (`0/90/180/270`)
- `R-L flip`
3. In `Manual annotation` mode:
- Select label: `Psoas`, `Paraspinal`, or `Abdominal_Wall`.
- Select tool: `Draw` or `Erase`.
- Set `Brush size` and `Brush transparency`.
- Choose `Preview update`:
  - `Realtime`: overlay updates as you annotate.
  - `On submit`: click `Apply annotations` to update overlay.
4. Download outputs:
- `mask.png`
- `mask.nii.gz` (binary mask)
- `label_map_00001.nii.gz`, `label_map_00002.nii.gz`, ... (multi-class label map)
- `label_map_volume_00001.nii.gz`, ... (full 3D label volume with your annotated slice at selected Z)

For `NIfTI volume (.nii.gz)` input, the app auto-assigns a case ID:
- Timestamp-based format: `L3-seg-YYYYMMDD-HHMMSS`
- Example: `L3-seg-20260211-113600`

Download names for that case:
- `L3-seg-20260211-113600-image.nii.gz`
- `L3-seg-20260211-113600-label.nii.gz`
- `L3-seg-20260211-113600-label-volume.nii.gz` (optional 3D export)

Area logging:
- Click `Finalize case: save CSV + download pair ZIP` in `Downloads`.
- Log file: `segmentation-label-app/data/area_log.csv`
- Each row stores case ID, source file, selected Z slice, total area, per-muscle areas, HU mean/std (total and per muscle), and view settings.

## Output Meaning

- `mask.nii.gz`: foreground vs background (binary).
- `label_map_XXXXX.nii.gz`: class IDs for future training.
  - `0` = background
  - `1` = Psoas
  - `2` = Paraspinal
  - `3` = Abdominal_Wall

For future nnUNet training, prefer using `label_map_XXXXX.nii.gz`.

## Known Notes

- Browser cannot force a custom save folder; download destination is controlled by browser settings.
- If launch fails due module mismatch, verify versions:

```powershell
python -c "import streamlit; print(streamlit.__version__)"
pip show streamlit-drawable-canvas
```

Expected:
- `streamlit==1.29.0`
- `streamlit-drawable-canvas==0.9.3`

## Troubleshooting

If PowerShell blocks script execution:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

If port `8501` is already in use:

```powershell
python -m streamlit run segmentation-label-app\app\app_ascii.py --server.port 8502
```

## Dataset Validator

Use this script to validate image/label training pairs before model training.

Script:
- `scripts/validate_image_label_pairs.py`

Checks:
- image/label pairing completeness
- shape match
- affine match
- integer label map values
- allowed class IDs
- empty labels

Example (automatic pairing by filename key):

```powershell
python segmentation-label-app\scripts\validate_image_label_pairs.py `
  --images-dir path\to\images `
  --labels-dir path\to\labels `
  --label-prefix label_map_ `
  --allowed-labels 0,1,2,3 `
  --report-json segmentation-label-app\reports\validation_report.json
```

Example (explicit pairs via CSV):

Create CSV with columns `image,label`, then run:

```powershell
python segmentation-label-app\scripts\validate_image_label_pairs.py `
  --pairs-csv path\to\pairs.csv `
  --allowed-labels 0,1,2,3 `
  --report-json segmentation-label-app\reports\validation_report.json
```

Exit code:
- `0` = all checks passed
- `1` = validation issues found
- `2` = invalid arguments/path setup
