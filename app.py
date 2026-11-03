import os
from pathlib import Path
import streamlit as st
import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from dataset import dataset_warning, inspect_dataset
from model import SatelliteRestormerSR

st.set_page_config(page_title="Satellite Image Super-Resolution", layout="wide")
st.title("🛰️ Low-Resolution Satellite Image Super-Resolution (4x)")
st.caption("Architecture: Channel-Transposed Hybrid Transformer (Restormer-SR) | Loss: Charbonnier + 2D FFT")

PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_PATH = PROJECT_ROOT / "restormer_weights.pth"
DATASET_PATH = PROJECT_ROOT / "data" / "train"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_sr_model(weights_path=str(WEIGHTS_PATH)):
    model = SatelliteRestormerSR(scale=4).to(device)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
        trained = True
    else:
        trained = False
    model.eval()
    return model, trained

model, is_trained = load_sr_model()
dataset_report = inspect_dataset(DATASET_PATH)
dataset_message = dataset_warning(dataset_report)

st.sidebar.header("Demo Settings")
if is_trained:
    st.sidebar.success("Loaded trained weights: `restormer_weights.pth`")
else:
    st.sidebar.warning("No `restormer_weights.pth` found yet. Running with untrained residual weights. Run `train.py` to train.")
if dataset_message:
    st.sidebar.warning(dataset_message)
else:
    st.sidebar.success(f"Training data checked: {dataset_report['count']} image(s)")

mode = st.sidebar.radio(
    "Select Demo Mode:",
    ["CA Evaluation Mode (Downsample HR -> Compare PSNR/SSIM)", "Direct 4x Upscale (LR Input Only)"]
)

uploaded_file = st.file_uploader("Upload a Satellite Image (.jpg, .png, .tif)", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    raw_img = Image.open(uploaded_file).convert("RGB")

    if mode.startswith("CA Evaluation Mode"):
        w, h = raw_img.size
        hr_w, hr_h = (w // 4) * 4, (h // 4) * 4
        hr_img = raw_img.resize((hr_w, hr_h), Image.Resampling.BICUBIC)
        lr_img = hr_img.resize((hr_w // 4, hr_h // 4), Image.Resampling.BICUBIC)
        bicubic_img = lr_img.resize((hr_w, hr_h), Image.Resampling.BICUBIC)

        lr_tensor = TF.to_tensor(lr_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = torch.clamp(model(lr_tensor), 0.0, 1.0)
        sr_img = TF.to_pil_image(sr_tensor.squeeze(0).cpu())

        hr_np = np.array(hr_img)
        bic_np = np.array(bicubic_img)
        sr_np = np.array(sr_img)

        psnr_bic = psnr(hr_np, bic_np, data_range=255)
        ssim_bic = ssim(hr_np, bic_np, channel_axis=2, data_range=255)
        psnr_sr = psnr(hr_np, sr_np, data_range=255)
        ssim_sr = ssim(hr_np, sr_np, channel_axis=2, data_range=255)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bicubic PSNR", f"{psnr_bic:.2f} dB")
        m2.metric("Restormer-SR PSNR", f"{psnr_sr:.2f} dB", f"{psnr_sr - psnr_bic:+.2f} dB")
        m3.metric("Bicubic SSIM", f"{ssim_bic:.4f}")
        m4.metric("Restormer-SR SSIM", f"{ssim_sr:.4f}", f"{ssim_sr - ssim_bic:+.4f}")

        st.divider()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.subheader(f"1. LR Input ({hr_w//4}×{hr_h//4})")
            st.image(lr_img, use_container_width=True)
        with c2:
            st.subheader("2. Bicubic (4x)")
            st.image(bicubic_img, use_container_width=True)
        with c3:
            st.subheader("3. Restormer-SR (4x)")
            st.image(sr_img, use_container_width=True)
        with c4:
            st.subheader(f"4. Ground Truth ({hr_w}×{hr_h})")
            st.image(hr_img, use_container_width=True)

    else:
        w, h = raw_img.size
        if max(w, h) > 256:
            st.info("Input image is large; resizing to 128×128 before 4x upscaling to prevent memory overflow.")
            raw_img = raw_img.resize((128, 128), Image.Resampling.BICUBIC)
            w, h = raw_img.size

        bicubic_img = raw_img.resize((w * 4, h * 4), Image.Resampling.BICUBIC)
        lr_tensor = TF.to_tensor(raw_img).unsqueeze(0).to(device)
        with torch.no_grad():
            sr_tensor = torch.clamp(model(lr_tensor), 0.0, 1.0)
        sr_img = TF.to_pil_image(sr_tensor.squeeze(0).cpu())

        c1, c2, c3 = st.columns(3)
        with c1:
            st.subheader(f"LR Input ({w}×{h})")
            st.image(raw_img, use_container_width=True)
        with c2:
            st.subheader(f"Bicubic 4x ({w*4}×{h*4})")
            st.image(bicubic_img, use_container_width=True)
        with c3:
            st.subheader(f"Restormer-SR 4x ({w*4}×{h*4})")
            st.image(sr_img, use_container_width=True)
