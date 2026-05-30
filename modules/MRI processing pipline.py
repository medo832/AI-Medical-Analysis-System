"""
MRI Brain Tumor Pipeline
========================
U-Net segmentation -> Radiomics on segmented ROI -> Clinical descriptors -> Tumor-type heuristic.
Uses radiomics_lite fallback (no PyRadiomics dependency).
The trained model expects 256x256 grayscale input and outputs a binary tumor mask.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import cv2
import numpy as np

import config
from modules import radiomics_lite


@dataclass
class BrainAnalysis:
    has_tumor: bool
    tumor_area_px: int = 0
    tumor_pct: float = 0.0
    mask: Optional[np.ndarray] = None
    features: Dict[str, float] = field(default_factory=dict)
    clinical: Dict[str, str] = field(default_factory=dict)
    tumor_type_guess: str = "no_tumor"
    tumor_profile: Optional[dict] = None
    bbox: Optional[tuple] = None


_cache = {"model": None}


def load_unet():
    """Load the U-Net segmentation model lazily."""
    if _cache["model"] is None:
        if not config.UNET_BRAIN.exists():
            raise FileNotFoundError(
                f"Brain segmentation model not found at {config.UNET_BRAIN}. "
                "Please place segmentation_model.h5 inside the models/ folder."
            )
        # Suppress TF noise
        import os
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        from tensorflow.keras.models import load_model
        _cache["model"] = load_model(str(config.UNET_BRAIN), compile=False)
    return _cache["model"]


# ====================================================================
def segment_tumor(image_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (grayscale image, binary mask) at original resolution."""
    model = load_unet()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    resized = cv2.resize(gray, (256, 256)) / 255.0
    batch = np.expand_dims(np.expand_dims(resized, axis=-1), axis=0)
    pred = model.predict(batch, verbose=0)[0]
    binary = (pred > 0.5).astype(np.uint8) * 255
    binary = binary.squeeze()
    mask = cv2.resize(binary, (w, h), interpolation=cv2.INTER_NEAREST)
    return gray, mask


def tumor_bbox(mask: np.ndarray) -> Optional[tuple]:
    """Tight bounding box around the largest connected tumor component."""
    if np.sum(mask) == 0:
        return None
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return (x, y, x + w, y + h)


def radiomics_to_clinical_brain(feats: dict, area_pct: float) -> dict:
    """Translate radiomic numbers into clinical descriptors (English)."""
    mean = feats.get("original_firstorder_Mean", 0)
    entropy = feats.get("original_firstorder_Entropy", 0)
    contrast = feats.get("original_glcm_Contrast", 0)
    skewness = feats.get("original_firstorder_Skewness", 0)

    if mean < 80:
        intensity = "hypointense"
    elif mean < 150:
        intensity = "isointense"
    else:
        intensity = "hyperintense"

    homogeneity = "homogeneous" if contrast < 1 else "heterogeneous"
    complexity = "low" if entropy < 5 else "high"
    asymmetry = "symmetric" if abs(skewness) < 1 else "asymmetric"

    if area_pct < 1:
        size_desc = "small"
    elif area_pct < 5:
        size_desc = "moderate"
    else:
        size_desc = "large"

    return {
        "intensity": intensity,
        "homogeneity": homogeneity,
        "complexity": complexity,
        "asymmetry": asymmetry,
        "size_desc": size_desc,
        "area_pct": f"{area_pct:.2f}%",
    }


def guess_tumor_type(mask: np.ndarray, clinical: dict, image_shape: tuple) -> str:
    """
    Heuristic guess of tumor type based on location + radiomic signal.
    NOT a classifier - this is purely supportive context for the RAG prompt.
    The final classification comes from the radiologist (and the RAG report).
    """
    if np.sum(mask) == 0:
        return "no_tumor"

    bbox = tumor_bbox(mask)
    if bbox is None:
        return "no_tumor"
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    cx = (x1 + x2) / 2 / w
    cy = (y1 + y2) / 2 / h

    # Pituitary: small, central, near the bottom-center of the image
    if 0.35 < cx < 0.65 and 0.55 < cy < 0.85 and clinical["size_desc"] == "small":
        return "pituitary"
    # Meningioma: usually peripheral (cortex/dura), more homogeneous
    if (cx < 0.30 or cx > 0.70) and clinical["homogeneity"] == "homogeneous":
        return "meningioma"
    # Glioma: heterogeneous, often larger and not peripheral
    if clinical["homogeneity"] == "heterogeneous" or clinical["complexity"] == "high":
        return "glioma"
    return "glioma"


# ====================================================================
def run_pipeline(image_bgr: np.ndarray, progress_cb=None) -> BrainAnalysis:
    def p(s, v):
        if progress_cb: progress_cb(s, v)

    p("Loading U-Net model...", 0.10)
    p("Segmenting tumor region...", 0.30)
    gray, mask = segment_tumor(image_bgr)
    tumor_area = int(np.sum(mask > 0))
    total_area = mask.size
    pct = (tumor_area / total_area) * 100 if total_area else 0.0

    analysis = BrainAnalysis(
        has_tumor=tumor_area > 0,
        tumor_area_px=tumor_area,
        tumor_pct=pct,
        mask=mask,
    )

    if tumor_area == 0:
        analysis.tumor_type_guess = "no_tumor"
        analysis.tumor_profile = config.BRAIN_TUMOR_PROFILES["no_tumor"]
        analysis.clinical = {
            "intensity": "-", "homogeneity": "-", "complexity": "-",
            "asymmetry": "-", "size_desc": "none", "area_pct": "0.00%",
        }
        p("No tumor detected", 1.0)
        return analysis

    p("Extracting tumor radiomics...", 0.60)
    bbox = tumor_bbox(mask)
    analysis.bbox = bbox
    # ROI = grayscale within the bounding box, multiplied by mask
    x1, y1, x2, y2 = bbox
    roi_img = gray[y1:y2, x1:x2].copy()
    roi_mask = (mask[y1:y2, x1:x2] > 0).astype(np.uint8)
    roi_img = roi_img * roi_mask  # zero out non-tumor pixels
    try:
        feats = radiomics_lite.extract_all_features(roi_img)
    except Exception:
        feats = {}
    analysis.features = feats

    p("Building clinical descriptors...", 0.85)
    analysis.clinical = radiomics_to_clinical_brain(feats, pct)
    analysis.tumor_type_guess = guess_tumor_type(mask, analysis.clinical, image_bgr.shape)
    analysis.tumor_profile = config.BRAIN_TUMOR_PROFILES.get(
        analysis.tumor_type_guess, config.BRAIN_TUMOR_PROFILES["no_tumor"]
    )
    p("Done", 1.0)
    return analysis


def draw_overlay(image_bgr: np.ndarray, analysis: BrainAnalysis) -> np.ndarray:
    """Overlay the tumor mask on the original MRI."""
    out = image_bgr.copy()
    if analysis.mask is None or not analysis.has_tumor:
        return out
    color_mask = np.zeros_like(out)
    color_mask[analysis.mask > 0] = (0, 0, 255)  # red overlay
    out = cv2.addWeighted(out, 0.7, color_mask, 0.3, 0)
    # Draw bounding box
    if analysis.bbox:
        x1, y1, x2, y2 = analysis.bbox
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        label = f"Tumor: {analysis.tumor_pct:.2f}% area"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(out, (x1, max(0, y1 - th - 8)),
                      (x1 + tw + 6, y1), (0, 255, 255), -1)
        cv2.putText(out, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return out
