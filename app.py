"""
Medical AI Diagnostic System — Main App (v3)
=============================================
Premium hospital-grade UI:
- Refined colors (deep navy + emerald + subtle gold accents)
- Patient-friendly chatbot with auto-summary
- Simplified one-page PDF reports with progress charts
"""
from __future__ import annotations

import io
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import config
from modules import patients_db


# ====================================================================
st.set_page_config(
    page_title="Medical AI Diagnostic System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ====================================================================
# Premium hospital-grade CSS
# ====================================================================
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  #MainMenu, footer {{visibility: hidden;}}

  html, body, [class*="css"]  {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont,
                 'Segoe UI', Roboto, sans-serif !important;
  }}

  .stApp {{
    background: linear-gradient(180deg, {config.BG_LIGHT} 0%, #EEF2F7 100%);
  }}

  /* ============ Hospital header ============ */
  .hospital-header {{
    background: linear-gradient(135deg, {config.PRIMARY} 0%, {config.PRIMARY_DARK} 70%, #0E2F5C 100%);
    color: white;
    padding: 1.6rem 2rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 30px rgba(26, 77, 140, .22),
                0 2px 6px rgba(26, 77, 140, .15);
    display: flex; align-items: center; justify-content: space-between;
    position: relative; overflow: hidden;
  }}
  .hospital-header::after {{
    content: ""; position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {config.ACCENT} 0%, {config.GOLD} 100%);
  }}
  .hospital-left {{ display: flex; align-items: center; gap: 1.2rem; }}
  .hospital-logo {{
    width: 60px; height: 60px; border-radius: 14px;
    background: rgba(255,255,255,.12);
    display: grid; place-items: center; font-size: 32px;
    border: 1px solid rgba(255,255,255,.2);
    backdrop-filter: blur(10px);
  }}
  .hospital-title {{
    font-size: 1.55rem; font-weight: 800; letter-spacing: -.01em;
    line-height: 1.15; color: white;
  }}
  .hospital-sub {{ font-size: .85rem; opacity: .8; margin-top: 3px;
                   color: white; font-weight: 400; }}
  .hospital-right {{ text-align: right; font-size: .8rem;
                     opacity: .9; color: white; }}
  .hospital-right .dept {{ font-weight: 600; margin-bottom: 2px; }}

  /* ============ Section heading ============ */
  .sec-h {{
    display: flex; align-items: center; gap: .6rem;
    font-size: 1.05rem; font-weight: 700;
    color: {config.TEXT_DARK}; margin: .8rem 0 .8rem;
    letter-spacing: -.005em;
  }}
  .sec-h::before {{
    content: ""; width: 4px; height: 20px; border-radius: 3px;
    background: linear-gradient(180deg, {config.PRIMARY} 0%, {config.ACCENT} 100%);
  }}

  /* ============ Stat cards ============ */
  .stat-grid {{
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 1rem; margin-bottom: 1.2rem;
  }}
  .stat-card {{
    background: white;
    border: 1px solid {config.BORDER_LIGHT}; border-radius: 14px;
    padding: 1.1rem 1.2rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, .04);
    transition: all .2s ease;
    position: relative; overflow: hidden;
  }}
  .stat-card::before {{
    content: ""; position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, {config.PRIMARY} 0%, {config.ACCENT} 100%);
    opacity: .8;
  }}
  .stat-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 8px 20px rgba(26, 77, 140, .10);
  }}
  .stat-card .lbl {{
    font-size: .72rem; color: {config.TEXT_LIGHT};
    font-weight: 600; text-transform: uppercase;
    letter-spacing: .06em;
  }}
  .stat-card .val {{
    font-size: 2rem; font-weight: 800; color: {config.PRIMARY};
    margin-top: 6px; line-height: 1; letter-spacing: -.02em;
  }}
  .stat-card .delta {{
    font-size: .72rem; color: {config.TEXT_LIGHT};
    margin-top: 8px; font-weight: 500;
  }}

  /* ============ Module cards ============ */
  .mod-grid {{
    display: grid; grid-template-columns: repeat(2, 1fr);
    gap: 1.3rem; margin-top: 1rem;
  }}
  .mod-card {{
    background: white;
    border: 1px solid {config.BORDER_LIGHT}; border-radius: 16px;
    padding: 1.6rem;
    box-shadow: 0 2px 10px rgba(15, 23, 42, .05);
    transition: all .2s ease;
  }}
  .mod-card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 12px 28px rgba(26, 77, 140, .12);
    border-color: {config.ACCENT};
  }}
  .mod-card .icon {{
    font-size: 40px; margin-bottom: .5rem;
    background: linear-gradient(135deg, {config.PRIMARY} 0%, {config.ACCENT} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: inline-block;
  }}
  .mod-card .ttl {{
    font-size: 1.15rem; font-weight: 700; letter-spacing: -.01em;
    color: {config.TEXT_DARK}; margin-bottom: .4rem;
  }}
  .mod-card .desc {{
    font-size: .9rem; color: {config.TEXT_LIGHT}; line-height: 1.6;
  }}
  .mod-card .badge {{
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    background: #ECFDF5; color: {config.ACCENT_DARK};
    font-size: .72rem; font-weight: 600;
    margin-top: .7rem;
    border: 1px solid #BBF7D0;
  }}

  /* ============ Pills ============ */
  .pill {{
    display: inline-block; padding: 5px 14px; border-radius: 999px;
    font-size: .78rem; font-weight: 600;
    background: {config.BG_SOFT}; color: {config.TEXT_DARK};
    border: 1px solid {config.BORDER_LIGHT};
    margin-right: 6px; margin-bottom: 6px;
  }}
  .pill.good {{ background:#DCFCE7;color:#166534;border-color:#BBF7D0; }}
  .pill.danger {{ background:#FEE2E2;color:#991B1B;border-color:#FECACA; }}
  .pill.warn {{ background:#FEF3C7;color:#92400E;border-color:#FDE68A; }}
  .pill.info {{ background:#E0F2FE;color:#075985;border-color:#BAE6FD; }}

  /* ============ Patient summary card (NEW) ============ */
  .pt-summary {{
    background: linear-gradient(135deg, #FAFBFC 0%, #F0F9FF 100%);
    border: 1px solid {config.BORDER_LIGHT};
    border-left: 4px solid {config.ACCENT};
    border-radius: 14px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 2px 8px rgba(15, 23, 42, .04);
  }}
  .pt-summary .pt-head {{
    font-size: 1.15rem; font-weight: 700; color: {config.PRIMARY};
    margin-bottom: .5rem; letter-spacing: -.01em;
  }}
  .pt-summary .pt-short {{
    font-size: 1rem; color: {config.TEXT_DARK};
    font-weight: 500; margin-bottom: .8rem; line-height: 1.5;
  }}
  .pt-summary .pt-detail {{
    font-size: .9rem; color: {config.TEXT_DEFAULT};
    line-height: 1.7; white-space: pre-line;
  }}
  .pt-summary .pt-next {{
    margin-top: .8rem; padding: .7rem 1rem;
    background: rgba(12, 166, 120, 0.08);
    border-radius: 8px;
    font-size: .88rem;
    color: {config.ACCENT_DARK}; font-weight: 500;
    border-left: 3px solid {config.ACCENT};
  }}

  /* ============ Chat ============ */
  .chat-msg {{
    padding: 12px 16px; border-radius: 14px;
    margin-bottom: 10px; font-size: .93rem; line-height: 1.6;
    box-shadow: 0 1px 3px rgba(0,0,0,.04);
  }}
  .chat-user {{
    background: linear-gradient(135deg, {config.PRIMARY} 0%, {config.PRIMARY_DARK} 100%);
    color: white;
    margin-left: auto; max-width: 80%;
  }}
  .chat-bot  {{
    background: white;
    border: 1px solid {config.BORDER_LIGHT};
    color: {config.TEXT_DARK}; max-width: 90%;
  }}

  /* ============ Inputs ============ */
  .stTextInput input, .stNumberInput input, .stSelectbox > div > div,
  .stTextArea textarea {{
    border-radius: 10px !important;
    border: 1px solid {config.BORDER_LIGHT} !important;
  }}
  .stTextInput input:focus, .stNumberInput input:focus,
  .stTextArea textarea:focus {{
    border-color: {config.PRIMARY} !important;
    box-shadow: 0 0 0 3px rgba(26, 77, 140, .1) !important;
  }}

  /* ============ Buttons ============ */
  .stButton button {{
    border-radius: 10px !important;
    font-weight: 600 !important;
    border: 1px solid transparent !important;
    transition: all .15s ease !important;
  }}
  .stButton button[kind="primary"] {{
    background: linear-gradient(135deg, {config.PRIMARY} 0%, {config.PRIMARY_DARK} 100%) !important;
    border: none !important;
    box-shadow: 0 2px 6px rgba(26, 77, 140, .2) !important;
  }}
  .stButton button[kind="primary"]:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(26, 77, 140, .3) !important;
  }}
  .stDownloadButton button {{
    background: linear-gradient(135deg, {config.ACCENT} 0%, {config.ACCENT_DARK} 100%) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 0.7rem 1.2rem !important;
    box-shadow: 0 3px 10px rgba(12, 166, 120, .25) !important;
  }}
  .stDownloadButton button:hover {{
    transform: translateY(-1px);
    box-shadow: 0 6px 16px rgba(12, 166, 120, .35) !important;
  }}

  /* ============ Tabs ============ */
  div[data-baseweb="tab-list"] {{
    gap: 6px;
    background: white;
    padding: 6px;
    border-radius: 12px;
    border: 1px solid {config.BORDER_LIGHT};
    box-shadow: 0 1px 3px rgba(15, 23, 42, .03);
  }}
  button[data-baseweb="tab"] {{
    border-radius: 8px !important;
    padding: 10px 20px !important;
    font-weight: 600 !important;
    transition: all .15s ease !important;
  }}
  button[data-baseweb="tab"][aria-selected="true"] {{
    background: linear-gradient(135deg, {config.PRIMARY} 0%, {config.PRIMARY_DARK} 100%) !important;
    color: white !important;
    box-shadow: 0 2px 6px rgba(26, 77, 140, .25) !important;
  }}

  /* ============ Empty hero ============ */
  .empty-hero {{
    border: 2px dashed {config.BORDER};
    border-radius: 16px;
    padding: 3rem 1rem; text-align: center;
    color: {config.TEXT_LIGHT};
    background: white;
  }}
  .empty-hero h3 {{ color: {config.TEXT_DARK};
                    margin: .5rem 0; font-weight: 700; }}

  /* ============ Sidebar ============ */
  section[data-testid="stSidebar"] {{
    background: white;
    border-right: 1px solid {config.BORDER_LIGHT};
  }}

  /* ============ Dataframes ============ */
  .stDataFrame {{ border-radius: 12px; overflow: hidden;
                  border: 1px solid {config.BORDER_LIGHT}; }}

  /* ============ Expander ============ */
  .streamlit-expanderHeader {{
    background: white;
    border-radius: 10px;
    border: 1px solid {config.BORDER_LIGHT};
    font-weight: 600;
  }}

  /* ============ Big "Generate PDF" highlight ============ */
  div.pdf-action {{
    background: linear-gradient(135deg, #FFFBEB 0%, #FEF3C7 100%);
    border: 1px solid #FCD34D;
    border-radius: 14px;
    padding: 1.2rem; margin-top: 1rem; text-align: center;
  }}
</style>
""", unsafe_allow_html=True)

# ====================================================================
# Hospital header
# ====================================================================
now_str = datetime.now().strftime("%A, %d %B %Y · %H:%M")
st.markdown(f"""
<div class="hospital-header">
  <div class="hospital-left">
    <div class="hospital-logo">🏥</div>
    <div>
      <div class="hospital-title">{config.HOSPITAL_NAME}</div>
      <div class="hospital-sub">{config.HOSPITAL_SUBTITLE}</div>
    </div>
  </div>
  <div class="hospital-right">
    <div class="dept">{config.DEPARTMENT}</div>
    <div>{now_str}</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# Sidebar
# ====================================================================
with st.sidebar:
    st.markdown("### ⚙️ Global Settings")

    st.markdown("**LLM (Gemini)**")
    use_llm = st.toggle("Use Gemini for reports & chat", value=True)
    custom_key = st.text_input("Custom Gemini key (optional)", type="password")
    gemini_key = custom_key.strip() or (config.DEFAULT_GEMINI_KEY if use_llm else None)

    st.markdown("---")
    st.markdown("### 📊 Quick Stats")
    s = patients_db.stats()
    st.metric("Total records", s["total"])
    st.metric("Records today", s["today"])

    st.markdown("---")
    with st.expander("ℹ️ About"):
        st.markdown("""
**Modules:**
- 🩻 **CT** — Kidney stones (YOLO + radiomics + KMeans + EAU)
- 🧠 **MRI** — Brain tumors (U-Net + radiomics + WHO CNS5)
- 💬 **Chatbot** — Bilingual AR/EN, patient-friendly
- 👥 **Registry** — CSV/Excel export, progress tracking
        """)


# ====================================================================
# Main tabs
# ====================================================================
tab_dash, tab_ct, tab_mri, tab_patients = st.tabs([
    "🏠 Dashboard", "🩻 CT — Kidney", "🧠 MRI — Brain", "👥 Patients",
])


# ====================================================================
# Dashboard
# ====================================================================
with tab_dash:
    s = patients_db.stats()
    st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="lbl">Total records</div>
    <div class="val">{s["total"]}</div>
    <div class="delta">All-time</div>
  </div>
  <div class="stat-card">
    <div class="lbl">CT studies</div>
    <div class="val">{s["ct"]}</div>
    <div class="delta">Kidney calculi</div>
  </div>
  <div class="stat-card">
    <div class="lbl">MRI studies</div>
    <div class="val">{s["mri"]}</div>
    <div class="delta">Brain tumor</div>
  </div>
  <div class="stat-card">
    <div class="lbl">Records today</div>
    <div class="val">{s["today"]}</div>
    <div class="delta">{datetime.now().strftime("%Y-%m-%d")}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sec-h">Diagnostic Modules</div>',
                unsafe_allow_html=True)
    st.markdown("""
<div class="mod-grid">
  <div class="mod-card">
    <div class="icon">🩻</div>
    <div class="ttl">CT Imaging — Renal Calculi</div>
    <div class="desc">AI-assisted detection and phenotyping of kidney stones
    using YOLO, radiomic feature extraction, and K-Means clustering.
    Recommendations grounded in EAU Urolithiasis Guidelines.</div>
    <div class="badge">YOLO · Radiomics · K-Means · RAG</div>
  </div>
  <div class="mod-card">
    <div class="icon">🧠</div>
    <div class="ttl">MRI Imaging — Brain Tumor</div>
    <div class="desc">U-Net semantic segmentation of brain tumor regions
    with quantitative radiomic analysis and reports following WHO CNS5
    (2021) terminology.</div>
    <div class="badge">U-Net · Radiomics · WHO CNS5 · RAG</div>
  </div>
  <div class="mod-card">
    <div class="icon">💬</div>
    <div class="ttl">Patient-Facing Chatbot</div>
    <div class="desc">Each module embeds a friendly bilingual chatbot
    (Arabic / English) that explains the patient's scan in plain language
    and answers their questions.</div>
    <div class="badge">Arabic · English · Patient-friendly</div>
  </div>
  <div class="mod-card">
    <div class="icon">📋</div>
    <div class="ttl">Records & Progress</div>
    <div class="desc">Centralized registry of every analysis with progress
    charts over time. One-click hospital-style PDF report + Excel/CSV export.</div>
    <div class="badge">CSV · Excel · PDF · Charts</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="sec-h" style="margin-top:1.6rem">'
                'Recent Activity</div>', unsafe_allow_html=True)
    df = patients_db.load_all()
    if df.empty:
        st.info("No analyses recorded yet. Start by uploading an image in "
                "the CT or MRI tab.")
    else:
        recent = df.tail(8).iloc[::-1]
        st.dataframe(
            recent[["timestamp", "patient_id", "patient_name",
                    "modality", "key_findings"]],
            use_container_width=True, hide_index=True,
            column_config={
                "timestamp": "Date / Time", "patient_id": "Patient ID",
                "patient_name": "Name", "modality": "Modality",
                "key_findings": "Key Findings",
            },
        )


# ====================================================================
# Patient form helper
# ====================================================================
def patient_form(modality: str):
    with st.expander("👤 Patient Information", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            pid = st.text_input("Patient ID *", key=f"pid_{modality}",
                                placeholder="e.g. P-001")
            pname = st.text_input("Full name", key=f"pname_{modality}")
        with c2:
            page = st.text_input("Age", key=f"page_{modality}")
            pgender = st.selectbox("Gender",
                                    ["—", "Male", "Female", "Other"],
                                    key=f"pgender_{modality}")
        with c3:
            pcontact = st.text_input("Contact", key=f"pcontact_{modality}")
            pphys = st.text_input("Referring physician", key=f"pphys_{modality}")
        pind = st.text_area("Clinical indication",
                             key=f"pind_{modality}", height=68)
        return {
            "patient_id": pid, "patient_name": pname,
            "age": page, "gender": pgender if pgender != "—" else "",
            "contact": pcontact, "referring_physician": pphys,
            "clinical_indication": pind,
        }


def render_summary_card(summary: dict):
    """Show the patient-friendly summary card."""
    if not summary:
        return
    # Convert newlines in detailed text to <br> for HTML
    detailed_html = (summary.get("detailed", "")
                     .replace("\n", "<br>"))
    st.markdown(f"""
<div class="pt-summary">
  <div class="pt-head">{summary.get("headline", "")}</div>
  <div class="pt-short">{summary.get("short", "")}</div>
  <div class="pt-detail">{detailed_html}</div>
  <div class="pt-next">💡 <strong>Next:</strong> {summary.get("next_steps", "")}</div>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# CT tab
# ====================================================================
with tab_ct:
    ct_main, ct_chat = st.tabs(["🔬 Analysis", "💬 Ask About Your Results"])

    with ct_main:
        patient = patient_form("ct")

        c_left, c_right = st.columns([1.1, 1.4], gap="large")
        with c_left:
            st.markdown('<div class="sec-h">CT Image Input</div>',
                        unsafe_allow_html=True)
            upload = st.file_uploader(
                "Upload CT slice",
                type=["jpg", "jpeg", "png", "bmp", "tif"],
                label_visibility="collapsed", key="ct_upload",
            )
            samples = sorted(Path(config.CT_SAMPLES).glob("*.jpg"))
            sample_choice = st.selectbox(
                "Or pick a sample",
                ["— none —"] + [s.name for s in samples], key="ct_sample",
            )
            conf = st.slider("Detection confidence",
                             0.10, 0.95, 0.50, 0.05, key="ct_conf")
            mmpx = st.number_input("mm per pixel", 0.05, 2.0,
                                    config.DEFAULT_MM_PER_PIXEL, 0.05,
                                    key="ct_mmpx")

            img_bgr = None
            if upload is not None:
                pil = Image.open(upload).convert("RGB")
                img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
            elif sample_choice != "— none —":
                img_bgr = cv2.imread(str(Path(config.CT_SAMPLES) / sample_choice))

            if img_bgr is not None:
                st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                         caption="Input image", use_container_width=True)

        with c_right:
            st.markdown('<div class="sec-h">Run Analysis</div>',
                        unsafe_allow_html=True)
            if img_bgr is None:
                st.markdown("""
<div class="empty-hero">
  <div style="font-size:42px">🩻</div>
  <h3>Awaiting CT image</h3>
  <p>Upload a CT slice or pick a sample to begin.</p>
</div>
                """, unsafe_allow_html=True)
            else:
                if not patient["patient_id"]:
                    st.warning("⚠️ Please enter a Patient ID first.")
                run = st.button("▶ Run CT Analysis", type="primary",
                                use_container_width=True,
                                disabled=not patient["patient_id"])
                if run:
                    from modules import ct_pipeline
                    prog = st.progress(0, text="Starting...")
                    def _cb(stage, pct):
                        prog.progress(min(int(pct * 100), 100), text=stage)
                    try:
                        t0 = time.time()
                        stones = ct_pipeline.run_pipeline(
                            img_bgr, confidence=conf,
                            mm_per_pixel=mmpx, progress_cb=_cb,
                        )
                        st.session_state["ct_stones"] = stones
                        st.session_state["ct_img"] = img_bgr
                        st.session_state["ct_patient"] = patient
                        st.session_state["ct_elapsed"] = time.time() - t0
                        # Generate patient summary right away
                        from modules import rag
                        idx = ct_pipeline.index_stone(stones)
                        st.session_state["ct_summary_en"] = (
                            rag.generate_ct_patient_summary(stones, idx, "en"))
                        st.session_state["ct_summary_ar"] = (
                            rag.generate_ct_patient_summary(stones, idx, "ar"))
                        st.session_state.pop("ct_rag", None)
                        st.session_state.pop("ct_chat_history", None)
                    except Exception as e:
                        st.error(f"Pipeline failed: `{e}`")
                        prog.empty()
                        st.stop()
                    prog.empty()
                    st.success(f"✅ Detected {len(stones)} stone(s) "
                               f"in {st.session_state['ct_elapsed']:.1f}s")

        # ---- Results ----
        if "ct_stones" in st.session_state:
            from modules import ct_pipeline
            stones = st.session_state["ct_stones"]
            img = st.session_state["ct_img"]
            idx = ct_pipeline.index_stone(stones)

            # Show patient-friendly summary CARD first
            render_summary_card(st.session_state["ct_summary_en"])

            pills = [f'<span class="pill info">🔍 {len(stones)} stone(s)</span>']
            if idx:
                pills.append(
                    f'<span class="pill">Ø {idx.diameter_mm:.1f} mm · '
                    f'{idx.size_bucket}</span>')
                if idx.cluster_profile:
                    cls = "good" if idx.cluster == 0 else "danger"
                    pills.append(f'<span class="pill {cls}">'
                                 f'{idx.cluster_profile["name_en"]}</span>')
            st.markdown(" ".join(pills), unsafe_allow_html=True)

            t_det, t_rad, t_eau, t_pdf = st.tabs([
                "🔍 Detection", "📊 Radiomics",
                "📚 EAU Recommendation (Clinical)", "📄 Download PDF Report",
            ])

            with t_det:
                cA, cB = st.columns([1.3, 1])
                with cA:
                    ann = ct_pipeline.draw_annotations(img, stones)
                    st.image(cv2.cvtColor(ann, cv2.COLOR_BGR2RGB),
                             caption="AI annotated",
                             use_container_width=True)
                with cB:
                    df = ct_pipeline.stones_dataframe(stones)
                    st.dataframe(df, use_container_width=True, hide_index=True)

            with t_rad:
                if any(s.features for s in stones):
                    keys = sorted({k for s in stones
                                   for k in (s.features or {}).keys()})
                    data = {f"Stone #{s.idx}":
                            [s.features.get(k, np.nan) for k in keys]
                            for s in stones}
                    feat_df = pd.DataFrame(data, index=keys)
                    feat_df.index.name = "Feature"
                    cats = sorted({k.split("_")[1]
                                   for k in keys if "_" in k})
                    sel = st.multiselect("Filter feature classes",
                                          cats, default=cats[:3],
                                          key="ct_feat_cats")
                    if sel:
                        mask = feat_df.index.to_series().apply(
                            lambda k: any(f"_{c}_" in k for c in sel))
                        feat_df = feat_df[mask]
                    st.dataframe(feat_df.style.format("{:.4g}"),
                                 use_container_width=True, height=400)
                else:
                    st.info("No radiomic features extracted.")

            with t_eau:
                st.caption("Detailed clinical recommendation from EAU "
                           "Urolithiasis Guidelines.")
                if not idx or not idx.clinical:
                    st.info("No clinical profile available for retrieval.")
                else:
                    if st.button("🔎 Retrieve EAU Guidance",
                                  type="primary", key="ct_rag_btn"):
                        from modules import rag
                        with st.spinner("Retrieving evidence..."):
                            rep, docs, q = rag.generate_report(
                                "ct", idx.clinical,
                                n_stones=len(stones),
                                gemini_api_key=gemini_key)
                        st.session_state["ct_rag"] = (rep, docs, q)
                    if "ct_rag" in st.session_state:
                        rep, docs, q = st.session_state["ct_rag"]
                        st.markdown("---")
                        st.markdown(rep)
                        with st.expander(f"📑 Retrieved passages "
                                          f"({len(docs)})"):
                            for i, d in enumerate(docs, 1):
                                meta = getattr(d, "metadata", {}) or {}
                                hdr = " · ".join(v for v in meta.values()
                                                  if v) or f"Passage {i}"
                                st.markdown(f"**{i}. {hdr}**")
                                st.markdown(f"> {d.page_content[:800]}...")

            with t_pdf:
                st.markdown("""
<div class="pdf-action">
  <h3 style="margin-bottom:.4rem;color:#92400E">📄 Download Your Medical Report</h3>
  <p style="color:#78350F;font-size:.9rem">One-click PDF download · No external programs needed · 
  Includes patient info, scan image, summary, recommendation, and progress chart.</p>
</div>
                """, unsafe_allow_html=True)
                note = st.text_area("Additional notes (optional)",
                                     "", height=68, key="ct_note")
                if st.button("🖨️ Generate PDF Report", type="primary",
                             key="ct_pdf_btn", use_container_width=True):
                    from modules import report

                    rec = patients_db.build_ct_record(
                        st.session_state["ct_patient"], stones, idx,
                    )
                    rid = patients_db.add_record(rec)
                    st.session_state["ct_record_id"] = rid

                    history_df = patients_db.get_patient_history(
                        st.session_state["ct_patient"]["patient_id"], "CT")

                    ann = ct_pipeline.draw_annotations(img, stones)
                    with st.spinner("Building report..."):
                        pdf_bytes = report.build_ct_report(
                            ann, stones, idx,
                            st.session_state["ct_summary_en"],
                            st.session_state["ct_patient"], rid,
                            history_df=history_df, note=note,
                        )
                    st.session_state["ct_pdf"] = (pdf_bytes, rid)

                if "ct_pdf" in st.session_state:
                    pdf_bytes, rid = st.session_state["ct_pdf"]
                    st.success(f"✅ Report ready. Record ID: **{rid}**")
                    st.download_button(
                        "⬇ Download PDF Now", pdf_bytes,
                        file_name=f"CT_Report_{rid}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

    # ---------------- CT chatbot ----------------
    with ct_chat:
        if "ct_stones" not in st.session_state:
            st.info("👋 Run a CT analysis first in the **Analysis** tab. "
                    "Then come back here to discuss your results in plain "
                    "language (Arabic or English).")
        else:
            # Show the summary card FIRST
            st.markdown('<div class="sec-h">📋 Your Case Summary</div>',
                        unsafe_allow_html=True)

            lang_toggle = st.radio("Summary language",
                                    ["English", "العربية"],
                                    horizontal=True, key="ct_sum_lang",
                                    label_visibility="collapsed")
            summary_key = ("ct_summary_ar" if lang_toggle == "العربية"
                           else "ct_summary_en")
            render_summary_card(st.session_state[summary_key])

            st.markdown('<div class="sec-h" style="margin-top:1.2rem">'
                        '💬 Ask About Your Results</div>',
                        unsafe_allow_html=True)
            st.caption("Ask anything in Arabic or English. The assistant "
                       "uses EAU clinical guidelines to answer.")

            if "ct_chat_history" not in st.session_state:
                st.session_state["ct_chat_history"] = []

            for turn in st.session_state["ct_chat_history"]:
                role_class = "chat-user" if turn["role"] == "user" else "chat-bot"
                avatar = "👤" if turn["role"] == "user" else "🤖"
                st.markdown(
                    f'<div class="chat-msg {role_class}">{avatar} '
                    f'{turn["content"]}</div>',
                    unsafe_allow_html=True,
                )

            # Sample questions
            st.markdown("**💡 Try a sample question:**")
            sample_cols = st.columns(4)
            samples = [
                ("EN", "Is this serious?"),
                ("AR", "هل حالتي خطيرة؟"),
                ("EN", "What treatment do I need?"),
                ("AR", "ما العلاج المناسب لحالتي؟"),
            ]
            for i, (lang, q) in enumerate(samples):
                if sample_cols[i].button(f"{lang}: {q}",
                                           key=f"ct_sample_{i}",
                                           use_container_width=True):
                    st.session_state["ct_chat_pending"] = q

            user_q = st.chat_input("Ask in Arabic or English...",
                                    key="ct_chat_input")
            pending = st.session_state.pop("ct_chat_pending", None) or user_q

            if pending:
                from modules import rag
                from modules import ct_pipeline as ctp
                # Build rich patient context from current case
                stones = st.session_state["ct_stones"]
                idx = ctp.index_stone(stones)
                ctx_parts = [
                    f"Patient: {st.session_state['ct_patient'].get('patient_name', '?')}",
                    f"Age: {st.session_state['ct_patient'].get('age', '?')}",
                    f"Clinical indication: {st.session_state['ct_patient'].get('clinical_indication', '-')}",
                    f"Findings: {len(stones)} kidney stone(s) detected",
                ]
                if idx:
                    prof = idx.cluster_profile or {}
                    ctx_parts.extend([
                        f"Main stone: {idx.diameter_mm:.2f} mm ({idx.size_bucket})",
                        f"Stone type (preliminary): {prof.get('name_en', '?')}",
                        f"Density: {idx.clinical.get('density', '?')}",
                        f"Texture: {idx.clinical.get('texture', '?')}",
                    ])
                ctx = "\n".join(ctx_parts)

                st.session_state["ct_chat_history"].append(
                    {"role": "user", "content": pending})
                with st.spinner("Thinking..."):
                    reply, _ = rag.chat_answer(
                        pending, "ct",
                        st.session_state["ct_chat_history"],
                        patient_context=ctx,
                        gemini_api_key=gemini_key,
                    )
                st.session_state["ct_chat_history"].append(
                    {"role": "assistant", "content": reply})
                st.rerun()

            if st.session_state["ct_chat_history"]:
                if st.button("🗑️ Clear conversation", key="ct_clear_chat"):
                    st.session_state["ct_chat_history"] = []
                    st.rerun()


# ====================================================================
# MRI tab
# ====================================================================
with tab_mri:
    mri_main, mri_chat = st.tabs(["🔬 Analysis", "💬 Ask About Your Results"])

    with mri_main:
        if not config.UNET_BRAIN.exists():
            st.error(f"""
⚠️ **Brain segmentation model not found.**

Please copy `segmentation_model.h5` into the `models/` folder:
`{config.UNET_BRAIN}`

The MRI module won't work until the model is in place.
            """)
        else:
            patient = patient_form("mri")

            m_left, m_right = st.columns([1.1, 1.4], gap="large")
            with m_left:
                st.markdown('<div class="sec-h">MRI Image Input</div>',
                            unsafe_allow_html=True)
                upload = st.file_uploader(
                    "Upload MRI slice",
                    type=["jpg", "jpeg", "png", "bmp", "tif"],
                    label_visibility="collapsed", key="mri_upload",
                )
                samples = sorted(Path(config.MRI_SAMPLES).glob("*.jpg"))
                sample_choice = st.selectbox(
                    "Or pick a sample",
                    ["— none —"] + [s.name for s in samples],
                    key="mri_sample",
                )

                img_bgr = None
                if upload is not None:
                    pil = Image.open(upload).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                elif sample_choice != "— none —":
                    img_bgr = cv2.imread(str(Path(config.MRI_SAMPLES)
                                              / sample_choice))

                if img_bgr is not None:
                    st.image(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB),
                             caption="Input MRI", use_container_width=True)

            with m_right:
                st.markdown('<div class="sec-h">Run Analysis</div>',
                            unsafe_allow_html=True)
                if img_bgr is None:
                    st.markdown("""
<div class="empty-hero">
  <div style="font-size:42px">🧠</div>
  <h3>Awaiting MRI image</h3>
  <p>Upload an MRI slice or pick a sample to begin.</p>
</div>
                    """, unsafe_allow_html=True)
                else:
                    if not patient["patient_id"]:
                        st.warning("⚠️ Please enter a Patient ID first.")
                    run = st.button("▶ Run MRI Analysis", type="primary",
                                     use_container_width=True,
                                     disabled=not patient["patient_id"],
                                     key="mri_run_btn")
                    if run:
                        from modules import mri_pipeline
                        prog = st.progress(0, text="Starting...")
                        def _cb(stage, pct):
                            prog.progress(min(int(pct * 100), 100),
                                          text=stage)
                        try:
                            t0 = time.time()
                            analysis = mri_pipeline.run_pipeline(
                                img_bgr, progress_cb=_cb)
                            st.session_state["mri_analysis"] = analysis
                            st.session_state["mri_img"] = img_bgr
                            st.session_state["mri_patient"] = patient
                            st.session_state["mri_elapsed"] = time.time() - t0
                            from modules import rag
                            st.session_state["mri_summary_en"] = (
                                rag.generate_mri_patient_summary(analysis, "en"))
                            st.session_state["mri_summary_ar"] = (
                                rag.generate_mri_patient_summary(analysis, "ar"))
                            st.session_state.pop("mri_rag", None)
                            st.session_state.pop("mri_chat_history", None)
                        except Exception as e:
                            st.error(f"Pipeline failed: `{e}`")
                            prog.empty()
                            st.stop()
                        prog.empty()
                        if analysis.has_tumor:
                            st.success(
                                f"✅ Tumor segmented ({analysis.tumor_pct:.2f}% area) "
                                f"in {st.session_state['mri_elapsed']:.1f}s")
                        else:
                            st.info("ℹ️ No tumor detected.")

            if "mri_analysis" in st.session_state:
                from modules import mri_pipeline
                analysis = st.session_state["mri_analysis"]
                img = st.session_state["mri_img"]

                # Patient-friendly summary card
                render_summary_card(st.session_state["mri_summary_en"])

                pills = []
                if analysis.has_tumor:
                    prof = analysis.tumor_profile or {}
                    pills.append('<span class="pill danger">⚠ Tumor detected</span>')
                    pills.append(f'<span class="pill">'
                                 f'{analysis.tumor_pct:.2f}% area</span>')
                    pills.append(f'<span class="pill warn">{prof.get("name_en", "?")}</span>')
                else:
                    pills.append('<span class="pill good">✓ No tumor</span>')
                st.markdown(" ".join(pills), unsafe_allow_html=True)

                t_seg, t_rad, t_who, t_pdf = st.tabs([
                    "🎯 Segmentation", "📊 Radiomics",
                    "📚 WHO CNS5 (Clinical)", "📄 Download PDF Report",
                ])

                with t_seg:
                    cA, cB = st.columns([1.3, 1])
                    with cA:
                        ovr = mri_pipeline.draw_overlay(img, analysis)
                        st.image(cv2.cvtColor(ovr, cv2.COLOR_BGR2RGB),
                                 caption="Tumor segmentation overlay",
                                 use_container_width=True)
                    with cB:
                        if analysis.has_tumor:
                            prof = analysis.tumor_profile or {}
                            c = analysis.clinical
                            st.markdown(f"""
**Suspected:** {prof.get('name_en', '-')}

| Metric | Value |
|--------|-------|
| Tumor area | {c.get('area_pct', '-')} |
| Size | {c.get('size_desc', '-')} |
| Intensity | {c.get('intensity', '-')} |
| Homogeneity | {c.get('homogeneity', '-')} |
| Complexity | {c.get('complexity', '-')} |
                            """)
                        else:
                            st.success("No suspicious mass detected.")

                with t_rad:
                    feats = analysis.features
                    if feats:
                        feat_df = pd.DataFrame({"Value": feats})
                        feat_df.index.name = "Feature"
                        cats = sorted({k.split("_")[1] for k in feats if "_" in k})
                        sel = st.multiselect("Filter feature classes",
                                              cats, default=cats[:3],
                                              key="mri_feat_cats")
                        if sel:
                            mask = feat_df.index.to_series().apply(
                                lambda k: any(f"_{c}_" in k for c in sel))
                            feat_df = feat_df[mask]
                        st.dataframe(feat_df.style.format("{:.4g}"),
                                     use_container_width=True, height=400)
                    else:
                        st.info("No features extracted.")

                with t_who:
                    st.caption("Detailed clinical recommendation based on "
                               "WHO CNS5 (2021).")
                    if st.button("🔎 Retrieve WHO CNS5 Guidance",
                                  type="primary", key="mri_rag_btn"):
                        from modules import rag
                        with st.spinner("Retrieving evidence..."):
                            rep, docs, q = rag.generate_report(
                                "mri", analysis, gemini_api_key=gemini_key)
                        st.session_state["mri_rag"] = (rep, docs, q)
                    if "mri_rag" in st.session_state:
                        rep, docs, q = st.session_state["mri_rag"]
                        st.markdown("---")
                        st.markdown(rep)
                        with st.expander(f"📑 Retrieved passages "
                                          f"({len(docs)})"):
                            for i, d in enumerate(docs, 1):
                                meta = getattr(d, "metadata", {}) or {}
                                hdr = " · ".join(v for v in meta.values()
                                                  if v) or f"Passage {i}"
                                st.markdown(f"**{i}. {hdr}**")
                                st.markdown(f"> {d.page_content[:800]}...")

                with t_pdf:
                    st.markdown("""
<div class="pdf-action">
  <h3 style="margin-bottom:.4rem;color:#92400E">📄 Download Your Medical Report</h3>
  <p style="color:#78350F;font-size:.9rem">One-click PDF download · No external programs needed · 
  Includes patient info, MRI scan, summary, recommendation, and progress chart.</p>
</div>
                    """, unsafe_allow_html=True)
                    note = st.text_area("Additional notes (optional)",
                                         "", height=68, key="mri_note")
                    if st.button("🖨️ Generate PDF Report", type="primary",
                                  key="mri_pdf_btn", use_container_width=True):
                        from modules import report

                        rec = patients_db.build_mri_record(
                            st.session_state["mri_patient"], analysis,
                        )
                        rid = patients_db.add_record(rec)
                        st.session_state["mri_record_id"] = rid

                        history_df = patients_db.get_patient_history(
                            st.session_state["mri_patient"]["patient_id"], "MRI")

                        ovr = mri_pipeline.draw_overlay(img, analysis)
                        with st.spinner("Building report..."):
                            pdf_bytes = report.build_mri_report(
                                ovr, analysis,
                                st.session_state["mri_summary_en"],
                                st.session_state["mri_patient"], rid,
                                history_df=history_df, note=note,
                            )
                        st.session_state["mri_pdf"] = (pdf_bytes, rid)

                    if "mri_pdf" in st.session_state:
                        pdf_bytes, rid = st.session_state["mri_pdf"]
                        st.success(f"✅ Report ready. Record ID: **{rid}**")
                        st.download_button(
                            "⬇ Download PDF Now", pdf_bytes,
                            file_name=f"MRI_Report_{rid}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )

    # ---------------- MRI chatbot ----------------
    with mri_chat:
        if "mri_analysis" not in st.session_state:
            st.info("👋 Run an MRI analysis first in the **Analysis** tab. "
                    "Then come back here to discuss your results in plain "
                    "language (Arabic or English).")
        else:
            st.markdown('<div class="sec-h">📋 Your Case Summary</div>',
                        unsafe_allow_html=True)

            lang_toggle = st.radio("Summary language",
                                    ["English", "العربية"],
                                    horizontal=True, key="mri_sum_lang",
                                    label_visibility="collapsed")
            summary_key = ("mri_summary_ar" if lang_toggle == "العربية"
                           else "mri_summary_en")
            render_summary_card(st.session_state[summary_key])

            st.markdown('<div class="sec-h" style="margin-top:1.2rem">'
                        '💬 Ask About Your Results</div>',
                        unsafe_allow_html=True)
            st.caption("Ask anything in Arabic or English. The assistant "
                       "uses WHO CNS5 (2021) guidelines to answer.")

            if "mri_chat_history" not in st.session_state:
                st.session_state["mri_chat_history"] = []

            for turn in st.session_state["mri_chat_history"]:
                role_class = "chat-user" if turn["role"] == "user" else "chat-bot"
                avatar = "👤" if turn["role"] == "user" else "🤖"
                st.markdown(
                    f'<div class="chat-msg {role_class}">{avatar} '
                    f'{turn["content"]}</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("**💡 Try a sample question:**")
            sample_cols = st.columns(4)
            samples = [
                ("EN", "Is this dangerous?"),
                ("AR", "هل حالتي خطيرة؟"),
                ("EN", "What's the next step?"),
                ("AR", "إيه اللي لازم أعمله الفترة الجاية؟"),
            ]
            for i, (lang, q) in enumerate(samples):
                if sample_cols[i].button(f"{lang}: {q}",
                                           key=f"mri_sample_{i}",
                                           use_container_width=True):
                    st.session_state["mri_chat_pending"] = q

            user_q = st.chat_input("Ask in Arabic or English...",
                                    key="mri_chat_input")
            pending = st.session_state.pop("mri_chat_pending", None) or user_q

            if pending:
                from modules import rag
                a = st.session_state["mri_analysis"]
                ctx_parts = [
                    f"Patient: {st.session_state['mri_patient'].get('patient_name', '?')}",
                    f"Age: {st.session_state['mri_patient'].get('age', '?')}",
                    f"Clinical indication: {st.session_state['mri_patient'].get('clinical_indication', '-')}",
                ]
                if a.has_tumor:
                    prof = a.tumor_profile or {}
                    ctx_parts.extend([
                        f"Findings: tumor detected, {a.tumor_pct:.2f}% area",
                        f"Suspected type (preliminary): {prof.get('name_en', '?')}",
                        f"Intensity: {a.clinical.get('intensity', '?')}",
                        f"Homogeneity: {a.clinical.get('homogeneity', '?')}",
                    ])
                else:
                    ctx_parts.append("Findings: no tumor detected")
                ctx = "\n".join(ctx_parts)

                st.session_state["mri_chat_history"].append(
                    {"role": "user", "content": pending})
                with st.spinner("Thinking..."):
                    reply, _ = rag.chat_answer(
                        pending, "mri",
                        st.session_state["mri_chat_history"],
                        patient_context=ctx,
                        gemini_api_key=gemini_key,
                    )
                st.session_state["mri_chat_history"].append(
                    {"role": "assistant", "content": reply})
                st.rerun()

            if st.session_state["mri_chat_history"]:
                if st.button("🗑️ Clear conversation", key="mri_clear_chat"):
                    st.session_state["mri_chat_history"] = []
                    st.rerun()


# ====================================================================
# Patients tab
# ====================================================================
with tab_patients:
    st.markdown('<div class="sec-h">👥 Patient Registry</div>',
                unsafe_allow_html=True)

    s = patients_db.stats()
    st.markdown(f"""
<div class="stat-grid">
  <div class="stat-card">
    <div class="lbl">Total</div><div class="val">{s["total"]}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">CT</div><div class="val">{s["ct"]}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">MRI</div><div class="val">{s["mri"]}</div>
  </div>
  <div class="stat-card">
    <div class="lbl">With findings</div>
    <div class="val">{s["with_findings"]}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 1])
    with fc1:
        f_pid = st.text_input("Filter: Patient ID", key="f_pid")
    with fc2:
        f_pname = st.text_input("Filter: Name", key="f_pname")
    with fc3:
        f_modality = st.selectbox("Filter: Modality",
                                    ["All", "CT", "MRI"], key="f_modality")
    with fc4:
        st.markdown("&nbsp;", unsafe_allow_html=True)
        if st.button("🔄 Refresh", use_container_width=True):
            st.rerun()

    mod_filter = f_modality if f_modality != "All" else ""
    df = patients_db.search(f_pid, f_pname, mod_filter)

    if df.empty:
        st.info("No records found.")
    else:
        display_df = df[["record_id", "timestamp", "patient_id",
                          "patient_name", "age", "gender",
                          "modality", "key_findings"]]
        st.dataframe(display_df, use_container_width=True, hide_index=True,
                     column_config={
                         "record_id": "Record ID", "timestamp": "Date/Time",
                         "patient_id": "Patient ID", "patient_name": "Name",
                         "age": "Age", "gender": "Gender",
                         "modality": "Mod.", "key_findings": "Key Findings",
                     })

        ec1, ec2, ec3 = st.columns([1, 1, 3])
        with ec1:
            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("⬇ Export CSV", csv,
                                file_name=f"patients_{datetime.now():%Y%m%d}.csv",
                                mime="text/csv", use_container_width=True)
        with ec2:
            try:
                buf = io.BytesIO()
                df.to_excel(buf, index=False, engine="openpyxl")
                st.download_button(
                    "⬇ Export Excel", buf.getvalue(),
                    file_name=f"patients_{datetime.now():%Y%m%d}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument."
                          "spreadsheetml.sheet",
                    use_container_width=True,
                )
            except Exception:
                st.caption("Install openpyxl for Excel export")

        st.markdown("---")
        st.markdown('<div class="sec-h">Record Detail & Progress</div>',
                    unsafe_allow_html=True)
        rids = ["—"] + df["record_id"].tolist()
        chosen = st.selectbox("View record", rids, key="rec_view")
        if chosen != "—":
            row = df[df["record_id"] == chosen].iloc[0].to_dict()
            cD1, cD2 = st.columns(2)
            with cD1:
                st.markdown(f"""
**Record ID:** `{row['record_id']}`  
**Date / Time:** {row['timestamp']}  
**Patient ID:** {row['patient_id']}  
**Name:** {row['patient_name']}  
**Age:** {row['age']} &nbsp;&nbsp; **Gender:** {row['gender']}  
**Contact:** {row.get('contact', '-')}
                """)
            with cD2:
                st.markdown(f"""
**Modality:** {row['modality']}  
**Exam type:** {row['exam_type']}  
**Referring MD:** {row['referring_physician']}  
**Clinical indication:** {row['clinical_indication']}
                """)
            st.markdown(f"**Key findings:** {row['key_findings']}")

            # Show patient's progress chart inline
            patient_history = patients_db.get_patient_history(
                row["patient_id"], row["modality"])
            if len(patient_history) >= 1:
                st.markdown('<div class="sec-h" style="margin-top:1rem">'
                            'Progress Over Time</div>',
                            unsafe_allow_html=True)
                from modules import report
                png = report._build_progress_chart_png(
                    patient_history, row["modality"])
                if png:
                    st.image(png, use_container_width=True)


# ====================================================================
# Footer
# ====================================================================
st.markdown(f"""
<div style="margin-top:2.5rem; padding:1.2rem;
            border-top:1px solid {config.BORDER_LIGHT};
            color:{config.TEXT_LIGHT}; font-size:.78rem; text-align:center;">
  Decision-support tool · For research and educational use only ·
  Not a substitute for professional medical diagnosis.<br>
  © {datetime.now().year} {config.HOSPITAL_NAME} · All rights reserved.
</div>
""", unsafe_allow_html=True)
