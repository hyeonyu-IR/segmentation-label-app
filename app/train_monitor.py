import re
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components


def parse_training_log(log_path: Path) -> pd.DataFrame:
    epoch_re = re.compile(r"Epoch\s+(\d+)")
    lr_re = re.compile(r"Current learning rate:\s+([0-9.eE+-]+)")
    train_re = re.compile(r"train_loss\s+(-?[0-9.]+)")
    val_re = re.compile(r"val_loss\s+(-?[0-9.]+)")
    time_re = re.compile(r"Epoch time:\s+([0-9.]+)\s+s")
    dice_re = re.compile(
        r"Pseudo dice\s+\[np.float32\(([-0-9.]+)\),\s*np.float32\(([-0-9.]+)\),\s*np.float32\(([-0-9.]+)\)\]"
    )
    ema_re = re.compile(r"New best EMA pseudo Dice:\s+([0-9.]+)")

    rows = []
    cur = {"epoch": None, "lr": None, "train_loss": None, "val_loss": None, "epoch_time_s": None, "dice1": None, "dice2": None, "dice3": None, "best_ema": None}
    best_ema = None
    for line in log_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = epoch_re.search(line)
        if m:
            if cur["epoch"] is not None:
                cur["best_ema"] = best_ema
                rows.append(cur.copy())
            cur = {"epoch": int(m.group(1)), "lr": None, "train_loss": None, "val_loss": None, "epoch_time_s": None, "dice1": None, "dice2": None, "dice3": None, "best_ema": None}
            continue
        m = lr_re.search(line)
        if m:
            cur["lr"] = float(m.group(1))
            continue
        m = train_re.search(line)
        if m:
            cur["train_loss"] = float(m.group(1))
            continue
        m = val_re.search(line)
        if m:
            cur["val_loss"] = float(m.group(1))
            continue
        m = time_re.search(line)
        if m:
            cur["epoch_time_s"] = float(m.group(1))
            continue
        m = dice_re.search(line)
        if m:
            cur["dice1"] = float(m.group(1))
            cur["dice2"] = float(m.group(2))
            cur["dice3"] = float(m.group(3))
            continue
        m = ema_re.search(line)
        if m:
            best_ema = float(m.group(1))
            continue

    if cur["epoch"] is not None:
        cur["best_ema"] = best_ema
        rows.append(cur.copy())

    if not rows:
        return pd.DataFrame(columns=["epoch", "lr", "train_loss", "val_loss", "epoch_time_s", "dice1", "dice2", "dice3", "best_ema"])
    df = pd.DataFrame(rows).sort_values("epoch").drop_duplicates("epoch", keep="last")
    return df


st.set_page_config(page_title="nnUNet Training Monitor", layout="wide")
st.title("nnUNet Training Monitor")

default_results = Path(r"C:\Users\hyeon\Documents\miniconda_medimg_env\data\nnUNet_results")
results_root = Path(st.sidebar.text_input("nnUNet results root", value=str(default_results)))
dataset_id = st.sidebar.number_input("Dataset ID", min_value=1, max_value=9999, value=711, step=1)
dataset_name = st.sidebar.text_input("Dataset name suffix", value="L3SM")
config = st.sidebar.text_input("Config", value="2d")
trainer = st.sidebar.text_input("Trainer", value="nnUNetTrainer")
fold = st.sidebar.number_input("Fold", min_value=0, max_value=9, value=0, step=1)

auto_refresh = st.sidebar.checkbox("Auto refresh", value=True)
refresh_sec = st.sidebar.slider("Refresh interval (sec)", min_value=5, max_value=120, value=15, step=5)
if auto_refresh:
    ms = int(refresh_sec * 1000)
    components.html(
        f"""
        <script>
          setTimeout(function() {{
            window.parent.location.reload();
          }}, {ms});
        </script>
        """,
        height=0,
        width=0,
    )

dataset_folder = f"Dataset{int(dataset_id):03d}_{dataset_name}"
fold_dir = results_root / dataset_folder / f"{trainer}__nnUNetPlans__{config}" / f"fold_{int(fold)}"

st.caption(f"Monitoring folder: {fold_dir}")
if not fold_dir.exists():
    st.error("Fold directory does not exist yet.")
    st.stop()

logs = sorted(fold_dir.glob("training_log_*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
if not logs:
    st.warning("No training log found yet.")
    st.stop()

log_path = logs[0]
st.caption(f"Log file: {log_path.name}")
df = parse_training_log(log_path)
if df.empty:
    st.warning("No epoch metrics parsed yet. Training may still be initializing.")
    st.stop()

latest = df.iloc[-1]
total_epochs = 1000
epochs_done = int(latest["epoch"]) + 1
epochs_left = max(0, total_epochs - epochs_done)
avg_time = float(df["epoch_time_s"].dropna().tail(20).mean()) if df["epoch_time_s"].dropna().shape[0] > 0 else None
eta_min = (epochs_left * avg_time / 60.0) if avg_time is not None else None

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current epoch", int(latest["epoch"]))
c2.metric("Train loss", f"{latest['train_loss']:.4f}" if pd.notna(latest["train_loss"]) else "N/A")
c3.metric("Val loss", f"{latest['val_loss']:.4f}" if pd.notna(latest["val_loss"]) else "N/A")
c4.metric("Best EMA dice", f"{latest['best_ema']:.4f}" if pd.notna(latest["best_ema"]) else "N/A")

c5, c6, c7, c8 = st.columns(4)
c5.metric("Dice - Psoas", f"{latest['dice1']:.4f}" if pd.notna(latest["dice1"]) else "N/A")
c6.metric("Dice - Paraspinal", f"{latest['dice2']:.4f}" if pd.notna(latest["dice2"]) else "N/A")
c7.metric("Dice - Abdominal_Wall", f"{latest['dice3']:.4f}" if pd.notna(latest["dice3"]) else "N/A")
c8.metric("ETA (min)", f"{eta_min:.1f}" if eta_min is not None else "N/A")

st.subheader("Loss Curves")
loss_df = df[["epoch", "train_loss", "val_loss"]].dropna(subset=["epoch"]).set_index("epoch")
st.line_chart(loss_df)

st.subheader("Dice Curves")
dice_df = df[["epoch", "dice1", "dice2", "dice3"]].dropna(subset=["epoch"]).set_index("epoch")
st.line_chart(dice_df)

st.subheader("Recent Epochs")
st.dataframe(df.tail(15), use_container_width=True)
