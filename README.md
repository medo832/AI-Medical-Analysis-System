# 🏥 AI Medical Analysis System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-red?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-CPU-orange?logo=tensorflow&logoColor=white)
![YOLO](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![Gemini](https://img.shields.io/badge/Google_Gemini-Flash-4285F4?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-Academic%20Use-green)

**A hospital-grade AI diagnostic platform combining CT kidney stone analysis and MRI brain tumor segmentation in a single professional interface.**

[Features](#-features) · [Architecture](#-system-architecture) · [Quick Start](#-quick-start) · [Usage](#-usage-guide) · [Models](#-ai-models) · [Troubleshooting](#-troubleshooting)

</div>

---

## 📌 Overview

The **AI Medical Analysis System** is a graduation project that demonstrates how deep learning can be integrated into a clinical decision-support workflow. It provides:

- **CT Scan Module** — Automated kidney stone detection, radiomics feature extraction, stone classification, and EAU guideline-based treatment recommendations.
- **MRI Scan Module** — Brain tumor segmentation using U-Net, WHO CNS5-based reporting, and radiomics analysis.
- **Bilingual AI Chatbot** — Arabic/English RAG-powered chatbot grounded in clinical guidelines for each module.
- **Patient Records** — Persistent patient database with search, filter, and export (Excel/CSV).
- **PDF Reports** — Professional hospital-style reports generated automatically after each scan.

> ⚠️ **Clinical Disclaimer:** This system is a decision-support tool only. All findings must be reviewed by a qualified medical professional. It is not a substitute for clinical diagnosis.

---

## ✨ Features

| Feature | Description |
|---|---|
| 🩻 **CT Kidney Stone Detection** | YOLOv8 model detects stones with bounding boxes and confidence scores |
| 📐 **Radiomics Analysis** | Extracts texture, shape, and intensity features from the detected region |
| 🔬 **Stone Classification** | KMeans clustering classifies stones into soft/dense phenotypes |
| 📋 **EAU Treatment Guidance** | RAG retrieval from EAU Urolithiasis Guidelines for treatment options |
| 🧠 **MRI Brain Tumor Segmentation** | U-Net model segments tumor regions with pixel-level precision |
| 🏷️ **Tumor Classification** | Identifies Glioma, Meningioma, Pituitary, or No Tumor |
| 📖 **WHO CNS5 Reporting** | Reports aligned with the latest WHO Brain Tumor classification |
| 💬 **Bilingual Chatbot** | Arabic & English medical Q&A powered by Gemini + ChromaDB |
| 👥 **Patient Database** | CSV-backed patient records with full search and export |
| 📄 **PDF Report Generation** | Hospital-grade PDF reports via ReportLab |

---

## 🏗️ System Architecture

```
AI-Medical-Analysis-System/
├── app.py                      # Streamlit entry point & UI
├── config.py                   # Global configuration & constants
├── setup.py                    # One-time initialization script
├── requirements.txt            # Python dependencies
│
├── modules/
│   ├── ct_pipeline.py          # CT: YOLO detection + Radiomics + KMeans
│   ├── mri_pipeline.py         # MRI: U-Net segmentation + Radiomics
│   ├── rag.py                  # RAG engine + bilingual chatbot (Gemini)
│   ├── patients_db.py          # Patient records management (CSV)
│   ├── report.py               # PDF report generation (ReportLab)
│   └── radiomics_lite.py       # Lightweight radiomics (no PyRadiomics dep)
│
├── models/
│   ├── yolo_kidney.pt          # YOLOv8 kidney stone detection model
│   ├── kmeans_kidney.pkl       # KMeans stone classifier
│   ├── scaler_kidney.pkl       # Feature scaler for KMeans
│   └── segmentation_model.h5   # U-Net brain tumor segmentation (372 MB, not in repo)
│
├── chroma_dbs/
│   ├── ct/                     # EAU Urolithiasis Guidelines vector store
│   └── mri/                    # WHO CNS5 Brain Tumor vector store
│
├── sample_images/
│   ├── ct/                     # Sample CT scans for demo
│   └── mri/                    # Sample MRI scans for demo
│
├── assets/                     # UI assets & branding
├── patients_data/              # Auto-created: patient records (CSV)
└── .streamlit/
    └── config.toml             # Streamlit theme configuration
```

---

## 🤖 AI Models

| Model | Task | Framework |
|---|---|---|
| **YOLOv8** (custom-trained) | Kidney stone detection in CT | Ultralytics |
| **U-Net** (custom-trained) | Brain tumor segmentation in MRI | TensorFlow / Keras |
| **KMeans** | Stone phenotype clustering (soft vs. dense) | scikit-learn |
| **all-MiniLM-L6-v2** | Text embeddings for RAG | sentence-transformers |
| **ms-marco-MiniLM-L-6-v2** | Cross-encoder reranking | sentence-transformers |
| **Gemini Flash** | Clinical report generation & chatbot | Google Generative AI |

> 📌 **Note:** `segmentation_model.h5` (~372 MB) is not included in the repository due to size. See [Quick Start](#-quick-start) for setup instructions.

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11** (required — TensorFlow CPU does not support 3.12+)
- **4 GB RAM** minimum (8 GB recommended)
- **~800 MB disk space** for all models and dependencies

### Step 1 — Clone the repository

```bash
git clone https://github.com/medo832/AI-Medical-Analysis-System.git
cd AI-Medical-Analysis-System
```

### Step 2 — Add the segmentation model

Download `segmentation_model.h5` separately and place it inside the `models/` folder:

```
models/
└── segmentation_model.h5   ← place it here (372 MB)
```

Without this file, the MRI tab will not function.

### Step 3 — Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

> ⏱️ First install takes 10–20 minutes (TensorFlow is large).

### Step 4 — Run one-time setup

```bash
py -3.11 setup.py
```

This script verifies models, rebuilds the ChromaDB vector stores with the correct dimensions, and downloads the embedding model (~90 MB). Takes ~5–10 minutes on first run only.

### Step 5 — Launch the app

```bash
py -3.11 -m streamlit run app.py
```

The browser will open automatically at `http://localhost:8501` 🎉

---

## 📖 Usage Guide

### 🏠 Dashboard
The home screen shows overall statistics and a summary of recent analyses.

### 🩻 CT Scan Tab
1. Enter patient information
2. Upload a CT scan image or choose a sample
3. Click **▶ Run CT Analysis**
4. View results across 4 sub-tabs:
   - **Detection** — YOLO bounding box + confidence
   - **Radiomics** — Extracted texture & shape features
   - **EAU Recommendation** — Guideline-based treatment options
   - **PDF Report** — Download the full clinical report
5. Use the **Chatbot** sidebar to ask questions about the findings

### 🧠 MRI Scan Tab
Same workflow as CT. The U-Net model produces a pixel-level segmentation mask, and results are classified according to WHO CNS5 (Glioma / Meningioma / Pituitary / No Tumor).

### 👥 Patients Tab
- Search by patient ID or name
- Filter by scan type (CT / MRI)
- Export records as **Excel** or **CSV**

---

## 💬 Bilingual Chatbot

Each module includes an integrated chatbot that automatically detects and responds in **Arabic or English**.

Answers are retrieved from the clinical guideline databases using RAG (Retrieval-Augmented Generation), then enhanced by Gemini Flash.

**Example questions:**

```
EN: What is the first-line treatment for a 6mm distal ureteric stone?
AR: ما هي مضاعفات تفتيت الحصوات بالموجات التصادمية؟

EN: Role of IDH mutation in glioma grading?
AR: ما هي درجات تصنيف الورم الدبقي؟
```

---

## 🔑 Gemini API Key

A default Gemini API key is included for demo purposes. To use your own key:

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Generate a free API key
3. Paste it into the **Custom Gemini Key** field in the app's sidebar

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `Python was not found` | Install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/) — check ✅ **Add Python to PATH** |
| `Brain segmentation model not found` | Copy `segmentation_model.h5` into the `models/` folder |
| `pip is not recognized` | Use `py -3.11 -m pip` instead of `pip` |
| TensorFlow install error | Confirm you are using Python 3.11, not 3.12 or 3.13 |
| ChromaDB dimension mismatch | Re-run `py -3.11 setup.py` |
| PDF not downloading | Try Microsoft Edge instead of Chrome, or check your Downloads folder |

---

## 🧰 Tech Stack

| Layer | Technology |
|---|---|
| **UI** | Streamlit |
| **CT Detection** | YOLOv8 (Ultralytics) |
| **MRI Segmentation** | U-Net (TensorFlow / Keras) |
| **Stone Classification** | KMeans (scikit-learn) |
| **Radiomics** | Custom lightweight radiomics (scikit-image) |
| **RAG / Vector Store** | ChromaDB + LangChain |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) |
| **LLM** | Google Gemini Flash |
| **PDF Generation** | ReportLab |
| **Patient Records** | Pandas + CSV + openpyxl |

---

## 📜 License

This project is released for **academic and research purposes only**.

It is not intended for clinical deployment without appropriate regulatory approval. All medical decisions must be made by qualified healthcare professionals.

---

## 👨‍💻 Author

**Mohamed** — Final Graduation Project  
📧 moel832005@gmail.com  
🔗 [GitHub](https://github.com/medo832)

---

<div align="center">
  <sub>Built with ❤️ as a graduation project · Not a substitute for professional medical diagnosis</sub>
</div>
