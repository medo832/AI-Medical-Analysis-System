"""
CT Kidney Stone Pipeline
========================
YOLO detection -> Radiomics -> KMeans phenotyping -> clinical translation.
Uses the radiomics_lite fallback (no PyRadiomics dependency).
"""
from __future__ import annotations

import math
import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

import config
from modules import radiomics_lite


# ====================================================================
@dataclass
class KidneyStone:
    idx: int
    bbox: tuple
    confidence: float
    area_px: int
    diameter_mm: float
    size_bucket: str
    features: Dict[str, float] = field(default_factory=dict)
    cluster: Optional[int] = None
    cluster_profile: Optional[dict] = None
    clinical: Dict[str, str] = field(default_factory=dict)


# ====================================================================
_cache = {"yolo": None, "scaler": None, "kmeans": None}


def load_yolo():
    if _cache["yolo"] is None:
        from ultralytics import YOLO
        _cache["yolo"] = YOLO(str(config.YOLO_KIDNEY))
        _cache["yolo"].to("cpu")
    return _cache["yolo"]


def load_scaler():
    if _cache["scaler"] is None:
        with open(config.SCALER_KIDNEY, "rb") as f:
            _cache["scaler"] = pickle.load(f)
    return _cache["scaler"]


def load_kmeans():
    if _cache["kmeans"] is None:
        with open(config.KMEANS_KIDNEY, "rb") as f:
            _cache["kmeans"] = pickle.load(f)
    return _cache["kmeans"]


# ====================================================================
def estimate_size(area_px: int, mm_per_pixel: float):
    dia_px = 1.1284 * math.sqrt(max(area_px, 0))
    dia_mm = dia_px * mm_per_pixel
    if dia_mm < 5:
        bucket = "<5 mm"
    elif dia_mm < 10:
        bucket = "5-10 mm"
    elif dia_mm < 20:
        bucket = "10-20 mm"
    else:
        bucket = ">20 mm"
    return dia_mm, bucket


def radiomics_to_clinical(feats, size_bucket):
    mean = feats.get("original_firstorder_Mean", 0)
    entropy = feats.get("original_firstorder_Entropy", 0)
    contrast = feats.get("original_glcm_Contrast", 0)
    if mean < 80:
        density = "very low (uric acid suspected)"
    elif mean < 120:
        density = "low"
    else:
        density = "high (calcium suspected)"
    texture = "homogeneous" if contrast < 1 else "heterogeneous"
    complexity = "low" if entropy < 5 else "high"
    return {"size": size_bucket, "density": density,
            "texture": texture, "complexity": complexity}


# ====================================================================
def detect_stones(image_bgr, conf=config.KIDNEY_CONF, iou=config.KIDNEY_IOU):
    yolo = load_yolo()
    results = yolo.predict(image_bgr, conf=conf, iou=iou, verbose=False)
    out = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            c = float(box.conf[0])
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if area > 0:
                out.append({"bbox": (x1, y1, x2, y2), "confidence": c, "area_px": area})
    return out


def extract_features(image_bgr, bbox):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = bbox
    roi = gray[y1:y2, x1:x2]
    if roi.size == 0:
        return {}
    try:
        return radiomics_lite.extract_all_features(roi)
    except Exception:
        return {}


def classify_stones(stones: List[KidneyStone]):
    classifiable = [s for s in stones if s.features]
    if not classifiable:
        return
    scaler = load_scaler()
    kmeans = load_kmeans()
    cols = list(scaler.feature_names_in_)
    X = np.asarray([[s.features.get(c, 0.0) for c in cols] for s in classifiable])
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    labels = kmeans.predict(scaler.transform(X))
    for s, lab in zip(classifiable, labels):
        s.cluster = int(lab)
        s.cluster_profile = config.KIDNEY_CLUSTERS.get(int(lab))


# ====================================================================
def run_pipeline(image_bgr, confidence=config.KIDNEY_CONF,
                 mm_per_pixel=config.DEFAULT_MM_PER_PIXEL, progress_cb=None):
    def p(s, v):
        if progress_cb: progress_cb(s, v)

    p("Detecting stones (YOLO)...", 0.10)
    dets = detect_stones(image_bgr, conf=confidence)
    if not dets:
        p("No stones detected", 1.0)
        return []

    stones = []
    for i, d in enumerate(dets):
        p(f"Extracting radiomics ({i+1}/{len(dets)})...", 0.10 + 0.6*(i+1)/len(dets))
        dia_mm, bucket = estimate_size(d["area_px"], mm_per_pixel)
        feats = extract_features(image_bgr, d["bbox"])
        s = KidneyStone(
            idx=i+1, bbox=d["bbox"], confidence=d["confidence"],
            area_px=d["area_px"], diameter_mm=dia_mm, size_bucket=bucket,
            features=feats,
        )
        s.clinical = radiomics_to_clinical(feats, bucket)
        stones.append(s)

    p("Classifying phenotypes (K-Means)...", 0.85)
    classify_stones(stones)
    p("Done", 1.0)
    return stones


def draw_annotations(image_bgr, stones):
    out = image_bgr.copy()
    for s in stones:
        x1, y1, x2, y2 = s.bbox
        color = s.cluster_profile["color_bgr"] if s.cluster_profile else (255, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"#{s.idx} C{s.cluster if s.cluster is not None else '?'} {s.diameter_mm:.1f}mm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, max(0, y1-th-6)), (x1+tw+6, y1), color, -1)
        cv2.putText(out, label, (x1+3, y1-4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def index_stone(stones):
    return max(stones, key=lambda s: s.area_px) if stones else None


def stones_dataframe(stones):
    if not stones:
        return pd.DataFrame()
    rows = []
    for s in stones:
        prof = s.cluster_profile or {}
        rows.append({
            "Stone": f"#{s.idx}",
            "Conf": round(s.confidence, 3),
            "Diameter (mm)": round(s.diameter_mm, 2),
            "Size": s.size_bucket,
            "Cluster": s.cluster if s.cluster is not None else "-",
            "Phenotype": prof.get("name_en", "-"),
            "Density": s.clinical.get("density", "-"),
            "Texture": s.clinical.get("texture", "-"),
        })
    return pd.DataFrame(rows)
