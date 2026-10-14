# 🩺 MUKO (v2.0): Multimodal Safety-First Oral Cancer Triage & Active-Learning System
> **"Giving every village the eyes to spot oral cancer before it catches them."**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

## 📌 1. Overview & Problem Context
India bears the world's highest burden of oral cavity cancer, recording **143,759 new cases** and **79,979 deaths annually** (1 death every ~6.5 minutes, *GLOBOCAN 2022*). Driven by a **21.4% smokeless tobacco prevalence** (*gutkha, khaini, betel quid*, *GATS-2 India*), over **70% of rural patients** are diagnosed at late stages (Stage III/IV), where 5-year survival drops below 30%.

**MUKO** is a safety-first, multimodal AI clinical triage system built for rural primary healthcare (ASHA workers, ANMs, and Community Health Officers at Ayushman Arogya Mandirs). It fuses **Calibrated Deep Computer Vision (`EfficientNet-B0`)**, **OpenCV Pre-Inference Quality Gatekeeping**, **Bayesian Log-Odds Epidemiological Risk Intake**, and **Grad-CAM Visual Explainability** into a 60-second smartphone workflow requiring zero external hardware attachments.

---

## 🏗️ 2. System Architecture & 5-Stage Pipeline

```text
[1. Epidemiological Intake] ──► [2. Camera & OpenCV Gatekeeper] ──► [3. Calibrated Vision Engine]
 (Tobacco, Trismus, >14d Ulcer)    (8% Border Strip + Blur/Light)      (EfficientNet-B0 + Flip TTA + T=2.0)
                                                                                     │
[5. Clinician Active-Learning Loop] ◄── [4. Traffic-Light Triage & Grad-CAM] ◄───────┘
 (PHC Doctor Ground-Truth Export)        (Bayesian Log-Odds Fusion: Green/Amber/Red)
