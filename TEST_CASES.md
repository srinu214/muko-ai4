# 🧪 MUKO v2.0 — Standardized Verification Test Cases

Judges can download the sample images in this folder and upload them directly into the live Streamlit portal to verify our 3-Tier Triage & Grad-CAM pipeline.

| Test ID | Image File | Clinical Intake Checkboxes | Expected Triage Tier | Expected Behavior |
| :--- | :--- | :--- | :--- | :--- |
| **TC-01** | `healthy_mouth_1.jpg` | All Unchecked | 🟢 **GREEN (<35%)** | Passes OpenCV quality check; low visual probability; no focal Grad-CAM lesion activation. |
| **TC-02** | `healthy_mouth_2.jpg` | Check *Smokeless Tobacco* + *>14 Days* | 🟡 **AMBER (35%–65%)** | Demonstrates Bayesian Log-Odds Multimodal Fusion escalating a borderline visual score. |
| **TC-03** | `cancer_lesion_1.jpg` | All Unchecked | 🔴 **RED (≥65%)** | High visual probability; Grad-CAM heatmap directly highlights the mucosal ulcer/tumor. |
