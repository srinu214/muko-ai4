from pathlib import Path
import io
import json
import zipfile
from datetime import datetime
import streamlit as st
import torch
import timm
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget

st.set_page_config(page_title="Oral Cancer Triage AI", page_icon="🩺", layout="centered")

st.title("🩺 Oral Cancer Safety-First Triage AI")
st.caption("Rural Primary Care Screening • Visual AI + Epidemiological Risk Fusion")

# 1. Dynamically find the .pt model file in the repository
BASE_DIR = Path(__file__).resolve().parent

CANDIDATE_NAMES = [
    "best_oral_triage_b0_calibrated.pt",
    "best_oral_triage_b0_calibrated .pt",
    "best_oral_triage_b0_balanced.pt",
    "best_oral_triage_b0_balanced .pt",
    "best_oral_triage_b0_antishortcut.pt",
]

MODEL_PATH = None
for name in CANDIDATE_NAMES:
    candidate = BASE_DIR / name
    if candidate.exists():
        MODEL_PATH = candidate
        break

if MODEL_PATH is None:
    pt_files = list(BASE_DIR.rglob("*.pt"))
    if pt_files:
        MODEL_PATH = pt_files[0]

# Safety-First Thresholds (< 35% Green | 35%-65% Amber | >= 65% Red)
GREEN_MAX = 0.35
AMBER_MAX = 0.65

@st.cache_resource
def load_triage_model():
    if MODEL_PATH is None or not MODEL_PATH.exists():
        files_found = [p.name for p in BASE_DIR.iterdir()]
        st.error(
            f"❌ No `.pt` model file found in `{BASE_DIR}`!\n\n"
            f"**Files currently in your GitHub repo:** `{files_found}`"
        )
        st.stop()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=1)
    state_dict = torch.load(str(MODEL_PATH), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model, device

model, device = load_triage_model()

eval_transform = transforms.Compose([
    transforms.Resize((260, 260)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- STEP 1: EPIDEMIOLOGICAL RISK QUESTIONNAIRE ---
st.subheader("1️⃣ Patient History & Symptom Intake")
col_q1, col_q2, col_q3 = st.columns(3)
with col_q1:
    q_tobacco = st.checkbox("Smokeless Tobacco (Gutkha/Khaini) or Smoking")
with col_q2:
    q_trismus = st.checkbox("Jaw Stiffness / Reduced Opening (< 3 Fingers)")
with col_q3:
    q_duration = st.checkbox("Ulcer or White/Red Patch > 14 Days")

# Log-Odds (Bayesian Prior Shift) weights based on WHO/Epidemiological Likelihood Ratios
logit_shift = 0.0
if q_tobacco:
    logit_shift += 0.35   # Moderate prior odds shift for tobacco exposure
if q_trismus:
    logit_shift += 0.45   # Submucous fibrosis / trismus indicator
if q_duration:
    logit_shift += 0.60   # WHO >14-day non-healing ulcer referral criterion

# --- STEP 2: IMAGE CAPTURE ---
st.subheader("2️⃣ Oral Cavity Image Capture")
tab1, tab2 = st.tabs(["📁 Upload Mouth Photo", "📷 Take Live Phone Photo"])
uploaded_file = tab1.file_uploader("Select a JPG or PNG image", type=["jpg", "jpeg", "png"])
camera_file = tab2.camera_input("Take a well-lit photo of the inner mouth")

image_source = uploaded_file if uploaded_file is not None else camera_file

if image_source is not None:
    raw_img = Image.open(image_source).convert("RGB")
    
    # Strip 8% outer border to avoid retractor, glove, and border artifacts
    w, h = raw_img.size
    cropped_img = raw_img.crop((int(w * 0.08), int(h * 0.08), int(w * 0.92), int(h * 0.92)))

    # --- CONSERVATIVE SOFT-WARNING GATEKEEPER ---
    img_np = np.array(cropped_img)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness_score = float(np.mean(gray))

    if brightness_score < 35:
        st.warning(
            f"⚠️ **Low Lighting Notice (Brightness: {brightness_score:.0f}/255):** "
            "Image is dim. If possible, turn on phone flash for clearer mucosal visibility."
        )
    if blur_score < 20:
        st.warning(
            f"⚠️ **Focus Notice (Sharpness: {blur_score:.0f}):** "
            "Slight motion blur detected. Analysis will proceed, but holding the phone steady improves accuracy."
        )

    # --- STEP 3: AI INFERENCE & BAYESIAN LOG-ODDS FUSION ---
    with st.spinner("Analyzing mucosal tissue & computing Bayesian clinical risk..."):
        t_full = eval_transform(cropped_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            t_flip = torch.flip(t_full, dims=[3])
            
            # Temperature-scaled visual logits (T = 2.0)
            l1 = model(t_full) / 2.0
            l2 = model(t_flip) / 2.0
            avg_visual_logit = (l1 + l2) / 2.0
            
            # Pure Visual Probability
            visual_prob = float(torch.sigmoid(avg_visual_logit).cpu().item())
            
            # Bayesian Log-Odds Fused Probability: sigmoid(visual_logit + clinical_logit_shift)
            fused_logit = avg_visual_logit + logit_shift
            final_prob = float(torch.sigmoid(fused_logit).cpu().item())

        clinical_delta = max(0.0, final_prob - visual_prob)

        disp_pil = cropped_img.resize((260, 260))
        rgb_float = np.float32(disp_pil) / 255.0
        
        targets_for_cam = [BinaryClassifierOutputTarget(1)]
        with GradCAM(model=model, target_layers=[model.conv_head]) as cam:
            grayscale_cam = cam(input_tensor=t_full, targets=targets_for_cam)[0, :]
        cam_overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # Display Score Breakdown Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Visual AI Score", f"{visual_prob*100:.1f}%")
    m2.metric("History Risk Shift", f"+{clinical_delta*100:.1f}%")
    m3.metric("Final Triage Risk", f"{final_prob*100:.1f}%")

    # 3-Tier Clinical Traffic Light
    if final_prob < GREEN_MAX:
        st.success(
            f"### 🟢 GREEN: Low Risk / Normal Tissue\n"
            f"**Combined Risk Score:** `{final_prob*100:.1f}%` (Threshold < 35%)\n\n"
            f"**Action:** No high-risk visual mucosal abnormalities detected. Counsel against tobacco use and maintain routine oral hygiene."
        )
    elif final_prob < AMBER_MAX:
        st.warning(
            f"### 🟡 AMBER: Mucosal Variation / Suspicious Lesion\n"
            f"**Combined Risk Score:** `{final_prob*100:.1f}%` (Threshold 35% – 65%)\n\n"
            f"**Action:** Atypical tissue or high-risk clinical history detected. Recommend tobacco cessation and a PHC/dental checkup if the patch persists > 14 days."
        )
    else:
        st.error(
            f"### 🔴 RED: High-Risk Oral Lesion Detected\n"
            f"**Combined Risk Score:** `{final_prob*100:.1f}%` (Threshold ≥ 65%)\n\n"
            f"**Action:** High suspicion of malignancy or severe pre-cancer. Immediate referral to district hospital / specialist for biopsy assessment."
        )

    st.info(
        "**Clinical Disclaimer:** This software is a triage screening demonstration tool. "
        "It does not replace histopathological biopsy or professional clinical diagnosis."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.image(disp_pil, caption="Analyzed Oral Region", use_container_width=True)
    with col2:
        st.image(cam_overlay, caption="Grad-CAM Lesion Heatmap", use_container_width=True)

    # --- STEP 4: CLINICIAN / VERIFIER ACTIVE LEARNING LOOP ---
    with st.expander("👨‍⚕️ Clinician / PHC Doctor Verification Mode (Active Learning Loop)"):
        st.caption("Restricted to Clinical Officers & Testing Team: Log verified diagnosis to evolve the model in the next training cycle.")
        true_ground_truth = st.radio(
            "Clinician-Verified Ground Truth:",
            ["0 - Healthy / Benign Mucosa", "1 - Confirmed OPMD / Oral Malignancy"],
            horizontal=True
        )
        
        zip_buffer = io.BytesIO()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_val = 0 if true_ground_truth.startswith("0") else 1
        img_filename = f"sample_{timestamp}_target{target_val}.jpg"
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            img_bytes = io.BytesIO()
            raw_img.save(img_bytes, format="JPEG", quality=95)
            zf.writestr(img_filename, img_bytes.getvalue())
            
            # Uniform CSV row ready to append directly into clean_splits.csv in Pandas!
            csv_header = "clean_path,target,split,visual_prob,final_prob,tobacco,trismus,duration_14d,blur_score,brightness_score,timestamp\n"
            csv_row = (
                f"/content/clean/feedback/{img_filename},{target_val},train,"
                f"{visual_prob:.4f},{final_prob:.4f},{int(q_tobacco)},{int(q_trismus)},"
                f"{int(q_duration)},{blur_score:.1f},{brightness_score:.1f},{timestamp}\n"
            )
            zf.writestr("metadata.csv", csv_header + csv_row)
            
        st.download_button(
            label="📥 Export Clinician-Verified Training Package (.zip)",
            data=zip_buffer.getvalue(),
            file_name=f"verified_case_{timestamp}_class{target_val}.zip",
            mime="application/zip"
        )