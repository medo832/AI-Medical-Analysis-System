"""
Hospital-Style PDF Report Generator (v3 — simplified + progress chart)
=======================================================================
Layout:
- Page 1: Patient info → Scan image → Brief summary → Recommendation
- Page 2 (if history >= 2): Progress chart
"""
from __future__ import annotations

import io
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    Table, TableStyle, PageBreak, HRFlowable,
)

import config


# ============================================================
def _esc(t) -> str:
    if t is None:
        return ""
    return (str(t).replace("&", "&amp;")
                  .replace("<", "&lt;").replace(">", "&gt;"))


def _bold_md(t: str) -> str:
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)


def _inline(t: str) -> str:
    return _bold_md(_esc(t))


# ============================================================
def _styles():
    base = getSampleStyleSheet()
    return {
        "HospitalName": ParagraphStyle(
            "HospitalName", parent=base["Title"], fontSize=18,
            textColor=colors.HexColor(config.PRIMARY),
            alignment=0, spaceAfter=0, leading=22),
        "HospitalSub": ParagraphStyle(
            "HospitalSub", parent=base["Normal"], fontSize=9,
            textColor=colors.HexColor(config.TEXT_LIGHT),
            spaceAfter=2, leading=12),
        "Department": ParagraphStyle(
            "Department", parent=base["Normal"], fontSize=8.5,
            textColor=colors.HexColor(config.ACCENT),
            spaceAfter=4, leading=12),
        "ReportTitle": ParagraphStyle(
            "ReportTitle", parent=base["Heading1"], fontSize=14,
            textColor=colors.HexColor(config.TEXT_DARK),
            alignment=1, spaceBefore=8, spaceAfter=10),
        "SectionH": ParagraphStyle(
            "SectionH", parent=base["Heading2"], fontSize=11,
            textColor=colors.white,
            backColor=colors.HexColor(config.PRIMARY),
            borderPadding=(4, 6, 4, 6),
            spaceBefore=10, spaceAfter=6, leading=14),
        "Body": ParagraphStyle(
            "Body", parent=base["BodyText"], fontSize=10,
            textColor=colors.HexColor(config.TEXT_DARK),
            leading=14, spaceAfter=4),
        "Headline": ParagraphStyle(
            "Headline", parent=base["Heading2"], fontSize=13,
            textColor=colors.HexColor(config.PRIMARY),
            spaceBefore=4, spaceAfter=8, leading=16),
        "Small": ParagraphStyle(
            "Small", parent=base["Normal"], fontSize=8,
            textColor=colors.HexColor(config.TEXT_LIGHT), leading=10),
        "Disclaimer": ParagraphStyle(
            "Disclaimer", parent=base["Italic"], fontSize=8,
            textColor=colors.HexColor(config.TEXT_LIGHT),
            leading=11, spaceBefore=6, alignment=4),
        "RecTitle": ParagraphStyle(
            "RecTitle", parent=base["Heading3"], fontSize=11,
            textColor=colors.HexColor(config.ACCENT_DARK),
            spaceAfter=4, leading=14),
    }


# ============================================================
def _header(elems, styles):
    elems.append(Paragraph(_esc(config.HOSPITAL_NAME), styles["HospitalName"]))
    elems.append(Paragraph(_esc(config.HOSPITAL_SUBTITLE), styles["HospitalSub"]))
    elems.append(Paragraph(_esc(config.DEPARTMENT), styles["Department"]))
    elems.append(HRFlowable(width="100%", thickness=1.5,
                            color=colors.HexColor(config.PRIMARY),
                            spaceBefore=2, spaceAfter=2))
    elems.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor(config.ACCENT),
                            spaceBefore=0, spaceAfter=8))


def _patient_block(elems, styles, patient, modality, record_id):
    """Big, clean patient info block at the top."""
    elems.append(Paragraph("Patient Information", styles["SectionH"]))

    rows = [
        ["Record ID", _esc(record_id),
         "Date / Time", datetime.now().strftime("%Y-%m-%d %H:%M")],
        ["Patient Name", _esc(patient.get("patient_name", "-")),
         "Patient ID", _esc(patient.get("patient_id", "-"))],
        ["Age", _esc(patient.get("age", "-")),
         "Gender", _esc(patient.get("gender", "-"))],
        ["Contact", _esc(patient.get("contact", "-")),
         "Referring MD", _esc(patient.get("referring_physician", "-"))],
        ["Modality", _esc(modality),
         "Exam Type", _esc(patient.get("exam_type", "-"))],
        ["Clinical Indication", _esc(patient.get("clinical_indication", "-")),
         "", ""],
    ]
    t = Table(rows, colWidths=[35*mm, 60*mm, 35*mm, 45*mm])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor(config.BG_SOFT)),
        ("BACKGROUND", (2, 0), (2, -1), colors.HexColor(config.BG_SOFT)),
        ("FONT", (0, 0), (-1, -1), "Helvetica", 9),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 9),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 9),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(config.TEXT_DARK)),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor(config.BORDER)),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("SPAN", (1, -1), (3, -1)),
    ]))
    elems.append(t)


def _image_block(elems, styles, image_bgr, caption: str):
    elems.append(Spacer(1, 6))
    elems.append(Paragraph("Scan Image", styles["SectionH"]))
    if image_bgr is None:
        elems.append(Paragraph("No image available.", styles["Body"]))
        return
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    from PIL import Image
    Image.fromarray(rgb).save(buf, format="PNG")
    buf.seek(0)
    elems.append(RLImage(buf, width=130*mm, height=98*mm, kind="proportional"))
    elems.append(Paragraph(_esc(caption), styles["Small"]))


def _summary_block(elems, styles, summary: dict):
    """Brief patient-friendly summary - 4-6 lines max."""
    elems.append(Spacer(1, 8))
    elems.append(Paragraph("Summary of Findings", styles["SectionH"]))
    elems.append(Paragraph(_inline(summary.get("headline", "")),
                            styles["Headline"]))
    elems.append(Paragraph(_inline(summary.get("short", "")), styles["Body"]))
    # The 'detailed' is multi-line; render each line:
    for line in (summary.get("detailed", "") or "").split("\n"):
        if line.strip():
            elems.append(Paragraph(_inline(line), styles["Body"]))


def _recommendation_block(elems, styles, summary: dict):
    """Small, friendly recommendation card."""
    elems.append(Spacer(1, 6))
    rec_text = summary.get("next_steps", "")
    rec = Table(
        [[Paragraph("<b>Recommendation</b>", styles["RecTitle"])],
         [Paragraph(_inline(rec_text), styles["Body"])]],
        colWidths=[175*mm],
    )
    rec.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1),
         colors.HexColor("#F0FDF4")),  # very light emerald
        ("LINEABOVE", (0, 0), (-1, 0), 2,
         colors.HexColor(config.ACCENT)),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elems.append(rec)


# ============================================================
def _build_progress_chart_png(history_df, modality: str) -> Optional[bytes]:
    """
    Build a beautiful matplotlib progress chart.
    Returns PNG bytes, or None if no useful history.

    CT: number of stones and (separately) index stone size in mm over time
    MRI: tumor area % over time
    """
    if history_df is None or len(history_df) < 1:
        return None

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    import pandas as pd

    df = history_df.copy()
    df["dt"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["dt"]).sort_values("dt")
    if df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.6), dpi=140)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFBFC")

    primary = config.PRIMARY
    accent = config.ACCENT

    if modality.upper() == "CT":
        # Parse stones_count and index size (in mm if available, fallback to bucket midpoint)
        stones = pd.to_numeric(df["stones_count"], errors="coerce").fillna(0)
        # Approximate index size mm from the size bucket string
        size_map = {"<5 mm": 3, "5-10 mm": 7, "10-20 mm": 15, ">20 mm": 22, "-": 0}
        sizes = df["index_stone_size"].map(size_map).fillna(0)

        ax.plot(df["dt"], stones, marker="o", markersize=9,
                linewidth=2.5, color=primary,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=primary, label="Stones count", zorder=3)
        ax.fill_between(df["dt"], stones, alpha=0.08, color=primary)

        ax2 = ax.twinx()
        ax2.plot(df["dt"], sizes, marker="s", markersize=8,
                 linewidth=2, color=accent, linestyle="--",
                 markerfacecolor="white", markeredgewidth=2,
                 markeredgecolor=accent, label="Index size (mm)", zorder=3)

        ax.set_ylabel("Number of Stones", color=primary, fontsize=10,
                      fontweight="bold")
        ax2.set_ylabel("Index Stone Size (mm)", color=accent, fontsize=10,
                       fontweight="bold")
        ax.tick_params(axis="y", labelcolor=primary)
        ax2.tick_params(axis="y", labelcolor=accent)
        ax.set_ylim(bottom=0)
        ax2.set_ylim(bottom=0)
        title = "Kidney Stone Progress Over Time"
        # Combined legend
        lns1, lbls1 = ax.get_legend_handles_labels()
        lns2, lbls2 = ax2.get_legend_handles_labels()
        ax.legend(lns1 + lns2, lbls1 + lbls2, loc="upper left",
                  frameon=True, facecolor="white", edgecolor="#E5E9F0",
                  fontsize=9)
    else:
        pct = pd.to_numeric(df["tumor_area_pct"], errors="coerce").fillna(0)
        ax.plot(df["dt"], pct, marker="o", markersize=10,
                linewidth=2.8, color=primary,
                markerfacecolor="white", markeredgewidth=2.5,
                markeredgecolor=primary, zorder=3)
        ax.fill_between(df["dt"], pct, alpha=0.15, color=primary)
        ax.set_ylabel("Tumor Area (% of slice)", color=primary, fontsize=10,
                      fontweight="bold")
        ax.set_ylim(bottom=0)
        title = "Tumor Area Progress Over Time"
        # Annotate points
        for x, y in zip(df["dt"], pct):
            ax.annotate(f"{y:.2f}%", (x, y),
                        textcoords="offset points", xytext=(0, 12),
                        ha="center", fontsize=9,
                        color=config.TEXT_DARK, fontweight="bold")

    ax.set_title(title, fontsize=13, fontweight="bold",
                 color=config.TEXT_DARK, pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(config.BORDER)
    ax.spines["bottom"].set_color(config.BORDER)
    ax.grid(axis="y", linestyle="--", alpha=0.5, color=config.BORDER)

    # Date formatting
    if len(df) == 1:
        # Single point - show context
        ax.set_xlim(df["dt"].iloc[0] - pd.Timedelta(days=2),
                    df["dt"].iloc[0] + pd.Timedelta(days=2))
    ax.xaxis.set_major_formatter(DateFormatter("%b %d\n%Y"))
    plt.setp(ax.get_xticklabels(), rotation=0, fontsize=9,
             color=config.TEXT_LIGHT)

    plt.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _progress_block(elems, styles, history_df, modality: str):
    """Add the progress chart section."""
    elems.append(Spacer(1, 10))
    elems.append(Paragraph("Progress Over Time", styles["SectionH"]))

    n = len(history_df) if history_df is not None else 0
    if n == 0:
        elems.append(Paragraph(
            "<i>No previous records for this patient. This is the first scan; "
            "future scans will appear here to track progress.</i>",
            styles["Small"]))
        return

    png = _build_progress_chart_png(history_df, modality)
    if png is None:
        elems.append(Paragraph(
            "<i>Progress chart unavailable.</i>", styles["Small"]))
        return

    buf = io.BytesIO(png)
    elems.append(RLImage(buf, width=175*mm, height=70*mm, kind="proportional"))

    if n == 1:
        elems.append(Paragraph(
            "<i>This is the first recorded scan for this patient. "
            "More data points will be added on future visits.</i>",
            styles["Small"]))
    else:
        elems.append(Paragraph(
            f"<i>Chart based on {n} scan(s) recorded for this patient.</i>",
            styles["Small"]))


def _footer(elems, styles):
    elems.append(Spacer(1, 10))
    elems.append(HRFlowable(width="100%", thickness=0.5,
                            color=colors.HexColor(config.BORDER)))
    elems.append(Spacer(1, 4))
    sig = [
        ["Reported by (AI):",
         f"{config.HOSPITAL_NAME} - Automated Analysis",
         "Reviewed by:", "_______________________"],
        ["Generation date:",
         datetime.now().strftime("%Y-%m-%d %H:%M"),
         "Signature & date:", ""],
    ]
    t = Table(sig, colWidths=[35*mm, 65*mm, 30*mm, 50*mm])
    t.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
        ("FONT", (0, 0), (0, -1), "Helvetica-Bold", 8.5),
        ("FONT", (2, 0), (2, -1), "Helvetica-Bold", 8.5),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor(config.TEXT_DARK)),
        ("LINEBELOW", (3, 0), (3, 0), 0.4,
         colors.HexColor(config.TEXT_LIGHT)),
        ("LINEBELOW", (3, 1), (3, 1), 0.4,
         colors.HexColor(config.TEXT_LIGHT)),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
    ]))
    elems.append(t)
    elems.append(Spacer(1, 4))
    elems.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated by an AI-based "
        "decision-support system and is intended for use by qualified medical "
        "professionals only. It is not a substitute for clinical evaluation, "
        "histopathology, or specialist consultation. All findings and "
        "recommendations must be reviewed and validated by a licensed "
        "physician before any clinical action is taken.",
        styles["Disclaimer"]))


# ============================================================
def build_ct_report(image_bgr, stones, idx_stone, summary: dict,
                    patient: dict, record_id: str,
                    history_df=None, note: str = "") -> bytes:
    """Simplified CT report: patient → image → summary → recommendation → chart."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="Medical AI - CT Kidney Report",
    )
    styles = _styles()
    elems = []

    _header(elems, styles)
    elems.append(Paragraph(
        "CT Imaging Report — Renal Calculi Analysis",
        styles["ReportTitle"]))

    _patient_block(elems, styles, patient, "CT (Computed Tomography)", record_id)
    _image_block(elems, styles, image_bgr,
                 "AI-annotated CT slice. Bounding boxes mark detected stones.")
    _summary_block(elems, styles, summary)
    _recommendation_block(elems, styles, summary)

    # Page 2: Progress chart
    elems.append(PageBreak())
    _header(elems, styles)
    elems.append(Paragraph(
        f"Patient: {_esc(patient.get('patient_name', '-'))}  "
        f"({_esc(patient.get('patient_id', '-'))})",
        styles["ReportTitle"]))
    _progress_block(elems, styles, history_df, "CT")

    if note:
        elems.append(Spacer(1, 8))
        elems.append(Paragraph("Additional Notes", styles["SectionH"]))
        elems.append(Paragraph(_esc(note), styles["Body"]))

    _footer(elems, styles)
    doc.build(elems)
    return buf.getvalue()


def build_mri_report(image_bgr, analysis, summary: dict,
                     patient: dict, record_id: str,
                     history_df=None, note: str = "") -> bytes:
    """Simplified MRI report."""
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=15*mm, rightMargin=15*mm,
        topMargin=14*mm, bottomMargin=14*mm,
        title="Medical AI - MRI Brain Report",
    )
    styles = _styles()
    elems = []

    _header(elems, styles)
    elems.append(Paragraph(
        "MRI Imaging Report — Brain Tumor Analysis",
        styles["ReportTitle"]))

    _patient_block(elems, styles, patient, "MRI (Magnetic Resonance)", record_id)
    _image_block(elems, styles, image_bgr,
                 "AI-segmented MRI slice. Red overlay marks any detected tumor "
                 "region; yellow box shows its extent.")
    _summary_block(elems, styles, summary)
    _recommendation_block(elems, styles, summary)

    # Page 2: Progress chart
    elems.append(PageBreak())
    _header(elems, styles)
    elems.append(Paragraph(
        f"Patient: {_esc(patient.get('patient_name', '-'))}  "
        f"({_esc(patient.get('patient_id', '-'))})",
        styles["ReportTitle"]))
    _progress_block(elems, styles, history_df, "MRI")

    if note:
        elems.append(Spacer(1, 8))
        elems.append(Paragraph("Additional Notes", styles["SectionH"]))
        elems.append(Paragraph(_esc(note), styles["Body"]))

    _footer(elems, styles)
    doc.build(elems)
    return buf.getvalue()
