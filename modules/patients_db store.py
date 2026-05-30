"""
Patients Database Manager
=========================
CSV-backed patient registry with history tracking for progress charts.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

import config


COLUMNS = [
    "record_id", "timestamp", "patient_id", "patient_name", "age", "gender",
    "contact", "modality", "exam_type", "referring_physician",
    "clinical_indication", "stones_count", "index_stone_size",
    "kidney_phenotype", "tumor_detected", "tumor_area_pct",
    "tumor_type_guess", "key_findings", "recommendation_summary", "notes",
]


def _ensure_csv():
    config.PATIENTS_DIR.mkdir(parents=True, exist_ok=True)
    if not config.PATIENTS_CSV.exists():
        pd.DataFrame(columns=COLUMNS).to_csv(config.PATIENTS_CSV, index=False)


def load_all() -> pd.DataFrame:
    _ensure_csv()
    df = pd.read_csv(config.PATIENTS_CSV, dtype=str).fillna("")
    return df


def _new_record_id() -> str:
    df = load_all()
    n = len(df) + 1
    return f"REC-{datetime.now().strftime('%Y%m%d')}-{n:04d}"


def add_record(record: dict) -> str:
    _ensure_csv()
    record_id = record.get("record_id") or _new_record_id()
    record["record_id"] = record_id
    record.setdefault("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    df = load_all()
    row = {c: str(record.get(c, "")) for c in COLUMNS}
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(config.PATIENTS_CSV, index=False)
    return record_id


def search(patient_id: str = "", patient_name: str = "",
           modality: str = "") -> pd.DataFrame:
    df = load_all()
    if patient_id:
        df = df[df["patient_id"].str.contains(patient_id, case=False, na=False)]
    if patient_name:
        df = df[df["patient_name"].str.contains(patient_name, case=False, na=False)]
    if modality:
        df = df[df["modality"].str.upper() == modality.upper()]
    return df


def get_patient_history(patient_id: str, modality: str = "") -> pd.DataFrame:
    """All records for one patient (optionally per modality), sorted by date."""
    df = load_all()
    df = df[df["patient_id"].str.strip().str.lower() ==
            str(patient_id).strip().lower()]
    if modality:
        df = df[df["modality"].str.upper() == modality.upper()]
    if not df.empty:
        df = df.sort_values("timestamp")
    return df


def stats() -> dict:
    df = load_all()
    if df.empty:
        return {"total": 0, "ct": 0, "mri": 0, "today": 0, "with_findings": 0}
    today = datetime.now().strftime("%Y-%m-%d")
    return {
        "total": len(df),
        "ct": int((df["modality"].str.upper() == "CT").sum()),
        "mri": int((df["modality"].str.upper() == "MRI").sum()),
        "today": int(df["timestamp"].str.startswith(today).sum()),
        "with_findings": int(
            ((df["stones_count"].fillna("0").astype(str) != "0") &
             (df["stones_count"] != ""))
            .sum()
            + (df["tumor_detected"].str.upper() == "YES").sum()
        ),
    }


def build_ct_record(patient: dict, stones, idx_stone) -> dict:
    if idx_stone:
        prof = idx_stone.cluster_profile or {}
        phenotype = prof.get("name_en", "-")
        key_findings = (
            f"{len(stones)} stone(s); index {idx_stone.diameter_mm:.2f}mm "
            f"({idx_stone.size_bucket}); phenotype: {phenotype}"
        )
        idx_size = idx_stone.size_bucket
    else:
        phenotype = "-"
        key_findings = "No stones detected"
        idx_size = "-"
    return {
        **patient,
        "modality": "CT", "exam_type": "CT KUB / Abdomen",
        "stones_count": str(len(stones)),
        "index_stone_size": idx_size,
        "kidney_phenotype": phenotype,
        "tumor_detected": "", "tumor_area_pct": "", "tumor_type_guess": "",
        "key_findings": key_findings,
    }


def build_mri_record(patient: dict, analysis) -> dict:
    if analysis.has_tumor:
        prof = analysis.tumor_profile or {}
        key_findings = (
            f"Tumor: {prof.get('name_en', '?')}; "
            f"{analysis.tumor_pct:.2f}% area; "
            f"{analysis.clinical.get('intensity', '-')}"
        )
    else:
        key_findings = "No tumor detected"
    return {
        **patient,
        "modality": "MRI", "exam_type": "MRI Brain",
        "stones_count": "", "index_stone_size": "", "kidney_phenotype": "",
        "tumor_detected": "Yes" if analysis.has_tumor else "No",
        "tumor_area_pct": f"{analysis.tumor_pct:.2f}",
        "tumor_type_guess": analysis.tumor_type_guess,
        "key_findings": key_findings,
    }
