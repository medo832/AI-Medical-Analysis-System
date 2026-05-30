"""
Medical AI Diagnostic System - Configuration
=============================================
Premium hospital-grade configuration.
"""
from pathlib import Path

# ============== Paths ==============
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
CHROMA_DIR = BASE_DIR / "chroma_dbs"
CT_CHROMA_DIR = CHROMA_DIR / "ct"
MRI_CHROMA_DIR = CHROMA_DIR / "mri"
SAMPLE_DIR = BASE_DIR / "sample_images"
CT_SAMPLES = SAMPLE_DIR / "ct"
MRI_SAMPLES = SAMPLE_DIR / "mri"
PATIENTS_DIR = BASE_DIR / "patients_data"
PATIENTS_CSV = PATIENTS_DIR / "patients_database.csv"
ASSETS_DIR = BASE_DIR / "assets"

# ============== Models ==============
YOLO_KIDNEY = MODELS_DIR / "yolo_kidney.pt"
KMEANS_KIDNEY = MODELS_DIR / "kmeans_kidney.pkl"
SCALER_KIDNEY = MODELS_DIR / "scaler_kidney.pkl"
UNET_BRAIN = MODELS_DIR / "segmentation_model.h5"

# ============== Branding ==============
HOSPITAL_NAME = "Medical AI Diagnostic System"
HOSPITAL_SUBTITLE = "Advanced Imaging Analysis & Clinical Decision Support"
DEPARTMENT = "Radiology · AI-Assisted Diagnostics"

# ============== Premium Color Palette ==============
# Deep professional navy + refined emerald + premium gold accent
PRIMARY = "#1A4D8C"          # Deep professional navy
PRIMARY_DARK = "#143E73"     # Hover/gradient end
PRIMARY_LIGHT = "#2563EB"    # Active state
ACCENT = "#0CA678"           # Refined emerald (health/recovery)
ACCENT_DARK = "#087F5B"      # Emerald hover
GOLD = "#D4A574"             # Premium subtle gold

BG_LIGHT = "#FAFBFC"         # Clean off-white
BG_PANEL = "#FFFFFF"         # White panels
BG_SOFT = "#F1F5F9"          # Soft alt background

BORDER_LIGHT = "#E5E9F0"
BORDER = "#CBD5E1"

TEXT_DARK = "#0F172A"
TEXT_DEFAULT = "#334155"
TEXT_LIGHT = "#64748B"
TEXT_MUTED = "#94A3B8"

SUCCESS = "#10B981"
WARNING = "#F59E0B"
DANGER = "#EF4444"
INFO = "#3B82F6"

# ============== Detection ==============
KIDNEY_CONF = 0.5
KIDNEY_IOU = 0.45
DEFAULT_MM_PER_PIXEL = 0.2

# ============== RAG ==============
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVAL = 10
TOP_K_RERANK = 5

# ============== LLM ==============
GEMINI_MODEL = "gemini-flash-lite-latest"
DEFAULT_GEMINI_KEY = "AIzaSyAl0J8GbNqMr1_BP1b6XuE5gI4WCdxS214"

# ============== Cluster phenotypes (CT/Kidney) ==============
KIDNEY_CLUSTERS = {
    0: {
        "name_en": "Soft / Low-Density Stone",
        "name_ar": "حصوة لينة منخفضة الكثافة",
        "color_bgr": (0, 200, 0),
        "color_hex": "#10B981",
        "stone_type": "Uric acid / Struvite",
        "stone_type_ar": "حمض اليوريك / ستروفيت",
        "likely_cause_en": "Infection or metabolic imbalance",
        "likely_cause_ar": "عدوى أو خلل في الأيض",
        "patient_friendly_en": (
            "This appears to be a softer type of stone. The good news is that "
            "soft stones often respond well to medication and lifestyle changes."
        ),
        "patient_friendly_ar": (
            "يبدو أنها حصوة من النوع اللين. الخبر الجيد هو أن هذا النوع غالباً "
            "يستجيب جيداً للأدوية والتغييرات في نمط الحياة."
        ),
        "interpretation_en": (
            "Soft, less calcified stones. May respond to medical expulsive therapy (MET), "
            "urinary alkalinization, or antibiotics if infection-related."
        ),
        "interpretation_ar": (
            "حصوات لينة وأقل تكلساً. قد تستجيب للعلاج الطارد للحصوات أو لقلونة البول "
            "أو للمضادات الحيوية في حال العدوى."
        ),
    },
    1: {
        "name_en": "Dense / Calcified Stone",
        "name_ar": "حصوة كثيفة متكلسة",
        "color_bgr": (0, 0, 255),
        "color_hex": "#EF4444",
        "stone_type": "Calcium oxalate / Calcium phosphate",
        "stone_type_ar": "أوكسالات الكالسيوم / فوسفات الكالسيوم",
        "likely_cause_en": "Dehydration, hypercalciuria, metabolic disorders",
        "likely_cause_ar": "الجفاف، فرط كالسيوم البول، اضطرابات أيضية",
        "patient_friendly_en": (
            "This is a denser, harder type of stone. These often need active "
            "treatment, but modern procedures are minimally invasive and effective."
        ),
        "patient_friendly_ar": (
            "هذه حصوة من النوع الكثيف الصلب. عادةً تحتاج لعلاج نشط، لكن "
            "الإجراءات الحديثة بسيطة وفعالة جداً."
        ),
        "interpretation_en": (
            "Highly calcified hard stones. Often require active treatment such as shock-wave "
            "lithotripsy (SWL), ureteroscopy (URS/RIRS), or percutaneous nephrolithotomy (PCNL)."
        ),
        "interpretation_ar": (
            "حصوات متكلسة وصلبة. عادةً تحتاج علاجاً نشطاً مثل تفتيت الحصوات بالموجات "
            "التصادمية أو منظار الحالب أو استئصال الحصوة بالمنظار عبر الجلد."
        ),
    },
}

# ============== Brain Tumor Classes ==============
BRAIN_TUMOR_PROFILES = {
    "glioma": {
        "name_en": "Glioma",
        "name_ar": "ورم دبقي",
        "color_hex": "#EF4444",
        "patient_friendly_en": (
            "A glioma is a tumor that originates in the supportive cells of the brain. "
            "Many gliomas can be effectively treated, especially when caught early."
        ),
        "patient_friendly_ar": (
            "الورم الدبقي هو ورم ينشأ من الخلايا الداعمة في الدماغ. "
            "كثير من هذه الأورام يمكن علاجها بفعالية خاصة عند اكتشافها مبكراً."
        ),
        "description_en": (
            "Tumors arising from glial cells. Range from low-grade (slow-growing) to "
            "high-grade malignant gliomas including glioblastoma."
        ),
        "description_ar": (
            "أورام تنشأ من الخلايا الدبقية. تتراوح من منخفضة الدرجة (بطيئة النمو) إلى "
            "أورام دبقية عالية الدرجة خبيثة بما في ذلك الورم الأرومي الدبقي."
        ),
    },
    "meningioma": {
        "name_en": "Meningioma",
        "name_ar": "ورم سحائي",
        "color_hex": "#F59E0B",
        "patient_friendly_en": (
            "A meningioma is usually a slow-growing, benign tumor that forms on the "
            "membranes covering the brain. Most are highly treatable."
        ),
        "patient_friendly_ar": (
            "الورم السحائي عادةً ورم حميد بطيء النمو يتكون على الأغشية المغطية للدماغ. "
            "معظم هذه الأورام قابلة للعلاج بدرجة عالية."
        ),
        "description_en": (
            "Tumors arising from the meninges (membranes surrounding the brain). "
            "Most are benign and slow-growing."
        ),
        "description_ar": (
            "أورام تنشأ من السحايا (الأغشية المحيطة بالدماغ). معظمها حميد وبطيء النمو."
        ),
    },
    "pituitary": {
        "name_en": "Pituitary Tumor",
        "name_ar": "ورم الغدة النخامية",
        "color_hex": "#0CA678",
        "patient_friendly_en": (
            "A pituitary tumor is a growth on the pituitary gland. The vast majority "
            "are benign and can be managed effectively with medication or surgery."
        ),
        "patient_friendly_ar": (
            "ورم الغدة النخامية هو نمو في الغدة النخامية. الغالبية العظمى حميدة "
            "ويمكن التعامل معها بفعالية عبر الأدوية أو الجراحة."
        ),
        "description_en": (
            "Tumors of the pituitary gland. Usually benign adenomas; may cause hormonal "
            "imbalances or pressure effects."
        ),
        "description_ar": (
            "أورام الغدة النخامية. عادةً أورام غدية حميدة؛ قد تسبب خللاً هرمونياً أو "
            "ضغطاً على الأنسجة المجاورة."
        ),
    },
    "no_tumor": {
        "name_en": "No Tumor Detected",
        "name_ar": "لم يتم اكتشاف ورم",
        "color_hex": "#10B981",
        "patient_friendly_en": (
            "Good news! The AI did not detect any suspicious mass in this scan. "
            "Continue routine follow-up as advised by your doctor."
        ),
        "patient_friendly_ar": (
            "خبر جيد! الذكاء الاصطناعي لم يكتشف أي كتلة مشبوهة في هذه الصورة. "
            "استمر في المتابعة الدورية كما أوصى طبيبك."
        ),
        "description_en": "No suspicious mass detected by the segmentation model.",
        "description_ar": "لم يكتشف نموذج التجزئة أي كتلة مشبوهة.",
    },
}
