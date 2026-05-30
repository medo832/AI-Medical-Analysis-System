<div align="center">

# 🏥 AI Medical Analysis System

**Hospital-grade clinical decision-support platform powered by deep learning.**  
Combining CT kidney stone analysis and MRI brain tumor segmentation in one professional Streamlit interface.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/downloads/release/python-3119/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![TensorFlow](https://img.shields.io/badge/TensorFlow_CPU-2.15-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-5C3EE8?style=for-the-badge)](https://github.com/ultralytics/ultralytics)
[![Gemini](https://img.shields.io/badge/Google_Gemini-Flash-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://aistudio.google.com/)
[![License](https://img.shields.io/badge/License-Academic_Use-22C55E?style=for-the-badge)](LICENSE)

[Features](#-features) · [Architecture](#-project-structure) · [AI Models](#-ai-models) · [Quick Start](#-quick-start) · [Usage](#-usage-guide) · [Chatbot](#-bilingual-chatbot) · [Troubleshooting](#-troubleshooting)

---

> ⚕️ **Clinical Disclaimer:** This system is a **decision-support tool only**. All findings must be reviewed and confirmed by a qualified medical professional. It does not substitute for clinical diagnosis or specialist judgement.

</div>

---

## 📌 Overview

The **AI Medical Analysis System** is a graduation project demonstrating how state-of-the-art deep learning can be integrated into a clinical imaging workflow. The system runs entirely on CPU, requires no GPU, and produces structured outputs that follow real clinical guidelines.

**Two complete diagnostic pipelines in one app:**

- **CT Kidney Stone Module** — Detects stones using a custom-trained YOLOv8 model, extracts radiomics features, classifies stones into phenotypes (soft/dense), and retrieves evidence-based treatment options directly from the **2026 EAU Urolithiasis Guidelines**.
- **MRI Brain Tumor Module** — Segments tumor regions using a custom-trained U-Net, extracts radiomics features, classifies tumor type (Glioma / Meningioma / Pituitary / None), and generates reports aligned with the **WHO CNS5 Brain Tumor Classification**.

Both modules share a unified bilingual RAG chatbot (Arabic + English) backed by ChromaDB and powered by Google Gemini Flash.

---

## ✨ Features

### 🩻 CT — Kidney Stone Analysis
| Step | What Happens |
|---|---|
| **Detection** | Custom YOLOv8 model localizes stones with bounding boxes, confidence scores, and pixel dimensions |
| **Size Estimation** | Stone diameter estimated in mm using configurable mm/pixel calibration (`<5 mm`, `5-10 mm`, `10-20 mm`, `>20 mm`) |
| **Radiomics** | 93 features extracted (first-order, GLCM, GLRLM, GLSZM, GLDM, NGTDM) via a pure NumPy/scikit-image implementation — no PyRadiomics dependency |
| **Phenotyping** | KMeans classifier assigns each stone to a clinical phenotype: **Soft/Low-Density** (uric acid / struvite) or **Dense/Calcified** (calcium oxalate / phosphate) |
| **EAU Guidance** | RAG retrieval from the 2026 EAU Urolithiasis Guidelines returns treatment options (MET, SWL, URS, PCNL) matched to stone size and phenotype |
| **PDF Report** | ReportLab generates a hospital-style A4 PDF with patient info, annotated scan, findings, and recommendations |

### 🧠 MRI — Brain Tumor Analysis
| Step | What Happens |
|---|---|
| **Segmentation** | U-Net (256×256 grayscale input, binary mask output) produces pixel-level tumor delineation |
| **Tumor Area** | Computes tumor coverage as percentage of total scan area |
| **Radiomics** | Same 93-feature extraction applied to the segmented ROI |
| **Classification** | Heuristic classifier identifies tumor type: **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor** |
| **WHO CNS5 Reporting** | RAG retrieval from WHO CNS5 guidelines provides tumor-specific clinical context |
| **PDF Report** | Same hospital-grade PDF format with segmentation overlay and tumor profile |

### 🔧 Shared Infrastructure
- **Bilingual AI Chatbot** — Arabic/English auto-detection, RAG + Gemini Flash, one chatbot per module
- **Patient Database** — CSV-backed registry, auto-assigned record IDs (`REC-YYYYMMDD-NNNN`), full search & filter
- **Excel/CSV Export** — One-click patient record export via openpyxl
- **Progress Charts** — If a patient has ≥2 scans, a history progress chart is included in the PDF (page 2)
- **Hospital UI** — Custom Streamlit CSS: Inter font, deep navy + emerald + gold palette, stat cards, chat bubbles

---

## 📁 Project Structure

```
AI-Medical-Analysis-System/
│
├── app.py                          # Streamlit entry point — UI, tabs, CSS, layout
├── config.py                       # All paths, colors, model params, cluster profiles
├── setup.py                        # One-time init: verify files, rebuild ChromaDB, preload embeddings
├── requirements.txt                # All Python dependencies (pinned ranges)
│
├── modules/
│   ├── CT processing Pipline.py    # YOLO → Radiomics → KMeans → clinical translation
│   ├── MRI processing pipline.py   # U-Net → Radiomics → tumor classification
│   ├── Unified RAG Module.py       # ChromaDB retrieval + cross-encoder reranking + Gemini
│   ├── patients_db store.py        # CSV patient registry (add, search, export)
│   ├── Report Generation.py        # ReportLab A4 PDF generator (2-page with chart)
│   ├── radiomics_lite.py           # 93-feature radiomics — pure NumPy/scikit-image fallback
│   └── __init__.py
│
├── models/
│   ├── yolo_kidney.pt              # Custom-trained YOLOv8 — kidney stone detection ✅
│   ├── kmeans_kidney.pkl           # Trained KMeans stone phenotype classifier ✅
│   ├── scaler_kidney.pkl           # Feature scaler for KMeans input ✅
│   └── segmentation_model.h5       # Custom-trained U-Net — brain tumor segmentation ⚠️ (372 MB, not in repo)
│
├── chroma_dbs/
│   ├── ct/
│   │   ├── chroma.sqlite3          # EAU 2026 Urolithiasis Guidelines vector store (384-dim)
│   │   ├── STONE_Markdown .md      # Source guideline document (markdown)
│   │   └── build_chroma.py         # Script used to build the CT vector store
│   └── mri/
│       ├── chroma.sqlite3          # WHO CNS5 Brain Tumor guidelines vector store (384-dim)
│       └── MRI.pdf                 # Source guideline document (PDF)
│
├── Nootbooks/
│   ├── CT/
│   │   ├── YOLOv8L.ipynb           # YOLOv8-L training notebook
│   │   ├── YOLOv8m.ipynb           # YOLOv8-M training notebook
│   │   └── PYRadiomics and K-mean.ipynb  # Radiomics extraction + KMeans training
│   └── MRI/
│       ├── U-net Model.ipynb       # U-Net architecture + training
│       ├── Test model.ipynb        # Model evaluation notebook
│       └── PyRadiomics.ipynb       # MRI radiomics experiments
│
├── sample_images/
│   ├── ct/                         # 5 real CT scan samples for demo
│   └── mri/                        # 4 MRI samples (glioma, meningioma, pituitary, no_tumor)
│
├── assets/                         # UI assets and branding
│
├── patients_data/                  # Auto-created at runtime — patient CSV records
│
└── .streamlit/
    └── config.toml                 # Theme: navy primary, white bg, sans-serif font, 50MB upload limit
```

---

## 🤖 AI Models

| Model | Task | Framework | Notes |
|---|---|---|---|
| **YOLOv8** (custom) | Kidney stone detection | Ultralytics | Conf: 0.5, IoU: 0.45, CPU inference |
| **U-Net** (custom) | Brain tumor segmentation | TensorFlow/Keras CPU | 256×256 grayscale → binary mask |
| **KMeans** (trained) | Stone phenotype clustering | scikit-learn | 2 clusters: soft vs. dense |
| **Scaler** | KMeans feature normalization | scikit-learn | Fitted on training radiomics features |
| **all-MiniLM-L6-v2** | Text embeddings (RAG) | sentence-transformers | 384-dim vectors |
| **ms-marco-MiniLM-L-6-v2** | Cross-encoder reranking | sentence-transformers | Top-10 → Top-5 rerank |
| **Gemini Flash Lite** | Report generation + chatbot | Google Generative AI | Configurable API key |

### Stone Phenotype Profiles

| Cluster | Name | Stone Type | Typical Cause |
|---|---|---|---|
| **0** | Soft / Low-Density | Uric acid / Struvite | Infection or metabolic imbalance |
| **1** | Dense / Calcified | Calcium oxalate / Phosphate | Dehydration, hypercalciuria |

### Brain Tumor Profiles

| Class | Description |
|---|---|
| **Glioma** | Tumors from glial cells; ranges from low-grade to glioblastoma |
| **Meningioma** | Usually benign, slow-growing, arises from meninges |
| **Pituitary Tumor** | Usually benign adenoma; may cause hormonal imbalance |
| **No Tumor** | No suspicious mass detected by the segmentation model |

---

## ⚡ Quick Start

### Prerequisites

- **Python 3.11** — required. TensorFlow CPU does not support Python 3.12+.
- **~800 MB disk space** for models and dependencies
- **4 GB RAM** minimum (8 GB recommended for smooth performance)
- No GPU required — runs on CPU

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/medo832/AI-Medical-Analysis-System.git
cd AI-Medical-Analysis-System
```

---

### Step 2 — Add the U-Net segmentation model

The file `segmentation_model.h5` (~372 MB) is not stored in this repository due to GitHub's file size limits.

Download it separately and place it here:

```
models/
└── segmentation_model.h5   ← place here
```

> The CT module works without this file. The MRI tab will throw an error until it is present.

---

### Step 3 — Install Python 3.11

```bash
# Verify you have Python 3.11
py -3.11 --version
# Expected output: Python 3.11.x
```

If not installed, download from [python.org](https://www.python.org/downloads/release/python-3119/) — choose **Windows installer (64-bit)** and check ✅ **Add Python to PATH**.

---

### Step 4 — Install dependencies

```bash
py -3.11 -m pip install -r requirements.txt
```

> ⏱️ First install takes 10–20 minutes. TensorFlow CPU is large. Leave the terminal running.

---

### Step 5 — Run one-time setup

```bash
py -3.11 setup.py
```

This script:
1. Verifies all required model files are present
2. Rebuilds the CT ChromaDB vector store at 384-dim (the correct embedding dimension)
3. Pre-downloads the sentence-transformer embedding model (~90 MB)
4. Runs a quick YOLO inference test

> ⏱️ Takes 5–10 minutes on first run only. Must be run before launching the app.

---

### Step 6 — Launch the app

```bash
py -3.11 -m streamlit run app.py
```

The browser opens automatically at **http://localhost:8501** 🎉

---

## 📖 Usage Guide

### 🏠 Dashboard
The home screen displays live statistics (total patients, CT scans, MRI scans) and a summary of the most recent analyses.

---

### 🩻 CT Scan Tab

1. Fill in patient details (name, ID, age, gender, referring physician, clinical indication)
2. Upload a CT scan image **or** select one of the 5 included samples
3. Click **▶ Run CT Analysis**
4. Results appear across 4 sub-tabs:

| Sub-tab | Contents |
|---|---|
| **Detection** | Annotated image with YOLO bounding boxes, confidence, estimated stone size in mm |
| **Radiomics** | Table of 93 extracted features + stone phenotype cluster assignment |
| **EAU Recommendation** | Retrieved treatment options from 2026 EAU Urolithiasis Guidelines |
| **PDF Report** | Preview + download button for the full hospital-style report |

5. Use the **Chatbot** panel in the sidebar for any follow-up questions about the findings

---

### 🧠 MRI Scan Tab

Same workflow as CT. The U-Net produces a segmentation mask overlaid on the original scan. Results include tumor type classification, radiomics features, WHO CNS5 clinical context, and a PDF report.

Four sample MRI scans are included: glioma, meningioma, pituitary tumor, and a no-tumor case.

---

### 👥 Patients Tab

- **Search** by patient ID or name
- **Filter** by modality (CT / MRI)
- **Export** the full database as Excel (`.xlsx`) or CSV

All records are auto-saved after each analysis with a unique record ID in the format `REC-YYYYMMDD-NNNN`.

---

## 💬 Bilingual Chatbot

Each module includes a chatbot that **automatically detects whether you write in Arabic or English** and responds in the same language.

Answers are grounded in the clinical guideline databases via RAG:
- CT chatbot uses the **2026 EAU Urolithiasis Guidelines**
- MRI chatbot uses the **WHO CNS5 Brain Tumor Classification**

Results are retrieved with `all-MiniLM-L6-v2` embeddings, reranked with a cross-encoder, then passed to **Gemini Flash** for a fluent, patient-friendly response.

**Example questions:**

```
# English
"What is the first-line treatment for a 6mm distal ureteric stone?"
"When is PCNL preferred over SWL?"
"Role of IDH mutation in glioma grading?"
"What are the surgical options for meningioma?"

# Arabic
"ما هي مضاعفات تفتيت الحصوات بالموجات التصادمية؟"
"متى يُفضّل استخدام منظار الحالب على تفتيت الحصوات؟"
"ما هي درجات تصنيف الورم الدبقي وفق WHO؟"
"ما علاج ورم الغدة النخامية الحميد؟"
```

---

## 🔑 Gemini API Key

A default Gemini API key is bundled for demo purposes. To use your own free key:

1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a free API key
3. Paste it into the **Custom Gemini Key** field in the app sidebar

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---|---|
| `Python was not found` | Install Python 3.11 from [python.org](https://www.python.org/downloads/release/python-3119/) and check ✅ **Add Python to PATH** |
| `Brain segmentation model not found` | Copy `segmentation_model.h5` into the `models/` folder |
| `pip is not recognized` | Use `py -3.11 -m pip install` instead of `pip install` |
| TensorFlow install error | Confirm you are on Python 3.11 — TensorFlow CPU does not support 3.12 or 3.13 |
| ChromaDB dimension mismatch | Re-run `py -3.11 setup.py` to rebuild the vector store |
| PDF not downloading | Try Microsoft Edge instead of Chrome, or check your Downloads folder |
| App is slow on first run | Normal — models are loading into RAM for the first time. Second run is much faster |
| `ModuleNotFoundError` | Re-run `py -3.11 -m pip install -r requirements.txt` |

---

## 🧰 Full Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **UI** | Streamlit 1.32+ | Web interface, tabs, sidebar, file upload |
| **Styling** | Custom CSS (Inter font, CSS variables) | Hospital-grade look and feel |
| **CT Detection** | YOLOv8 (Ultralytics) | Bounding box detection of kidney stones |
| **MRI Segmentation** | U-Net (TensorFlow/Keras CPU) | Pixel-level brain tumor mask |
| **Stone Classification** | KMeans + Scaler (scikit-learn) | Phenotype clustering (soft vs. dense) |
| **Radiomics** | Custom `radiomics_lite.py` (NumPy + scikit-image) | 93 imaging features without PyRadiomics |
| **Vector Store** | ChromaDB + LangChain | Guideline document embeddings |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim semantic text embeddings |
| **Reranking** | sentence-transformers (ms-marco-MiniLM-L-6-v2) | Cross-encoder relevance reranking |
| **LLM** | Google Gemini Flash Lite | Report generation and chatbot responses |
| **PDF** | ReportLab (A4) | Hospital-style clinical reports |
| **Patient Records** | Pandas + CSV + openpyxl | Patient database and Excel export |
| **Image Processing** | OpenCV + Pillow | Scan loading, annotation, mask overlay |

---

## 📓 Training Notebooks

The `Nootbooks/` directory contains the original Jupyter notebooks used to train all models:

| Notebook | Description |
|---|---|
| `CT/YOLOv8L.ipynb` | YOLOv8-Large training on kidney stone dataset |
| `CT/YOLOv8m.ipynb` | YOLOv8-Medium training (comparison run) |
| `CT/PYRadiomics and K-mean.ipynb` | Radiomics feature extraction + KMeans phenotyping |
| `MRI/U-net Model.ipynb` | U-Net architecture definition and training |
| `MRI/Test model.ipynb` | Model evaluation and segmentation visualization |
| `MRI/PyRadiomics.ipynb` | MRI radiomics experiments |

---

## 📊 Clinical Knowledge Bases

| Module | Guideline Source | Embedding Dimensions |
|---|---|---|
| CT | EAU Urolithiasis Guidelines 2026 | 384-dim (all-MiniLM-L6-v2) |
| MRI | WHO CNS5 Brain Tumor Classification | 384-dim (all-MiniLM-L6-v2) |

Retrieval pipeline: `similarity_search (top-10)` → `cross-encoder rerank (top-5)` → `Gemini Flash synthesis`

---

## 📜 License

This project is released for **academic and research purposes only**.

It is not approved for clinical deployment. All medical decisions require review by a licensed healthcare professional.

---

## 👨‍💻 Contact us

**Team** — Computer Science / AI Graduation Project

📧 [mse90901@gmail.com](mailto:mse90901@gmail.com)  
🔗 [github.com/medo832](https://github.com/medo832)

---

<div align="center">

**Built as a Final Graduation Project**  
*Kidney Stone Detection · Brain Tumor Segmentation · Clinical Decision Support . Patient assistant*

</div>
