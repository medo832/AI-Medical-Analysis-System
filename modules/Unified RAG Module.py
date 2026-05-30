"""
Unified RAG Module (CT + MRI) — v3 with patient-friendly summaries
==================================================================
"""
from __future__ import annotations

import os
from typing import List, Optional

import config


_state = {"embedder": None, "reranker": None, "ct_db": None, "mri_db": None}


def _embedder():
    if _state["embedder"] is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _state["embedder"] = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return _state["embedder"]


def _reranker():
    if _state["reranker"] is None:
        from sentence_transformers import CrossEncoder
        _state["reranker"] = CrossEncoder(config.RERANKER_MODEL)
    return _state["reranker"]


def _db(modality: str):
    key = f"{modality}_db"
    if _state[key] is None:
        from langchain_community.vectorstores import Chroma
        path = config.CT_CHROMA_DIR if modality == "ct" else config.MRI_CHROMA_DIR
        _state[key] = Chroma(persist_directory=str(path),
                             embedding_function=_embedder())
    return _state[key]


def retrieve(query: str, modality: str, k: int = config.TOP_K_RETRIEVAL) -> List:
    return _db(modality).similarity_search(query, k=k)


def rerank(query: str, docs: List, top_k: int = config.TOP_K_RERANK) -> List:
    if not docs:
        return []
    rr = _reranker()
    pairs = [[query, d.page_content] for d in docs]
    scores = rr.predict(pairs)
    paired = sorted(zip(scores, docs), key=lambda x: float(x[0]), reverse=True)
    return [d for _, d in paired[:top_k]]


# ====================== Query builders =============================
def build_ct_query(clinical: dict, n_stones: int) -> str:
    return f"""EAU urolithiasis guidelines: treatment selection by stone characteristics.

Clinical profile:
- Number of stones detected: {n_stones}
- Index stone size: {clinical.get('size', 'unknown')}
- Radiologic density: {clinical.get('density', 'unknown')}
- Internal texture: {clinical.get('texture', 'unknown')}
- Structural complexity: {clinical.get('complexity', 'unknown')}

Treatment options to evaluate: MET, SWL, RIRS/URS, PCNL, observation.
Include indications, contraindications, success rates."""


def build_mri_query(analysis) -> str:
    if not analysis.has_tumor:
        return ("WHO CNS5 brain tumor classification: normal MRI findings, "
                "follow-up imaging recommendations.")
    c = analysis.clinical
    return f"""WHO CNS5 brain tumor classification and management.

Imaging profile:
- Suspected tumor type: {analysis.tumor_type_guess}
- Tumor area: {c.get('area_pct', '?')} of slice
- Size: {c.get('size_desc', '?')}
- Signal intensity: {c.get('intensity', '?')}
- Homogeneity: {c.get('homogeneity', '?')}
- Texture complexity: {c.get('complexity', '?')}

Include WHO CNS5 grading, molecular markers, treatment guidelines."""


# ====================== Patient-Friendly Summaries =================
def generate_ct_patient_summary(stones, idx_stone, lang: str = "en") -> dict:
    """Patient-friendly summary of CT findings. Returns dict with 'short' and 'detailed'."""
    if not stones or not idx_stone:
        if lang == "ar":
            return {
                "headline": "✅ نتائج جيدة",
                "short": "لم يتم اكتشاف أي حصوات في الأشعة المقطعية.",
                "detailed": (
                    "الذكاء الاصطناعي قام بفحص الصورة بدقة ولم يكتشف أي حصوات في الكلى. "
                    "يُنصح بمتابعة دورية مع طبيبك حسب توصياته."
                ),
                "next_steps": "اشرب الماء بكميات كافية واتبع تعليمات طبيبك.",
            }
        return {
            "headline": "✅ Good news",
            "short": "No kidney stones detected on this CT scan.",
            "detailed": (
                "The AI carefully analyzed your scan and did not detect any kidney "
                "stones. Continue routine follow-up as advised by your doctor."
            ),
            "next_steps": "Stay well-hydrated and follow your doctor's guidance.",
        }

    prof = idx_stone.cluster_profile or {}
    size_mm = idx_stone.diameter_mm
    n = len(stones)

    if lang == "ar":
        plural = f"{n} حصوات" if n > 1 else "حصوة واحدة"
        friendly = prof.get("patient_friendly_ar",
                            "حصوة كلوية تستدعي تقييم طبيبك.")
        headline = f"🔍 تم اكتشاف {plural}"
        short = (f"تم اكتشاف {plural} في الكلى. "
                 f"الحصوة الرئيسية حجمها {size_mm:.1f} ملم.")
        detailed = (
            f"{friendly}\n\n"
            f"• عدد الحصوات: {n}\n"
            f"• حجم الحصوة الرئيسية: {size_mm:.1f} ملم ({idx_stone.size_bucket})\n"
            f"• نوع الحصوة المحتمل: {prof.get('name_ar', '-')}\n"
            f"• الكثافة: {idx_stone.clinical.get('density', '-')}"
        )
        next_steps = ("ناقش هذه النتائج مع طبيبك. التطبيق يحتوي على توصيات "
                      "علاج مفصلة من إرشادات EAU.")
    else:
        plural = f"{n} stones" if n > 1 else "1 stone"
        friendly = prof.get("patient_friendly_en",
                            "A kidney stone that warrants medical assessment.")
        headline = f"🔍 {plural} detected"
        short = (f"{plural.capitalize()} detected in the kidney. "
                 f"The main stone measures {size_mm:.1f} mm.")
        detailed = (
            f"{friendly}\n\n"
            f"• Number of stones: {n}\n"
            f"• Main stone size: {size_mm:.1f} mm ({idx_stone.size_bucket})\n"
            f"• Likely stone type: {prof.get('name_en', '-')}\n"
            f"• Density: {idx_stone.clinical.get('density', '-')}"
        )
        next_steps = ("Discuss these findings with your doctor. The app includes "
                      "detailed EAU-based treatment recommendations.")

    return {"headline": headline, "short": short,
            "detailed": detailed, "next_steps": next_steps}


def generate_mri_patient_summary(analysis, lang: str = "en") -> dict:
    """Patient-friendly summary of MRI findings."""
    if not analysis.has_tumor:
        prof = config.BRAIN_TUMOR_PROFILES["no_tumor"]
        if lang == "ar":
            return {
                "headline": "✅ نتائج مطمئنة",
                "short": "لم يتم اكتشاف أي ورم في الرنين المغناطيسي.",
                "detailed": prof.get("patient_friendly_ar", ""),
                "next_steps": "استمر في المتابعة الدورية مع طبيبك.",
            }
        return {
            "headline": "✅ Reassuring findings",
            "short": "No tumor detected on this MRI scan.",
            "detailed": prof.get("patient_friendly_en", ""),
            "next_steps": "Continue routine follow-up with your doctor.",
        }

    prof = analysis.tumor_profile or {}
    pct = analysis.tumor_pct

    if lang == "ar":
        friendly = prof.get("patient_friendly_ar", "")
        headline = f"⚠️ تم اكتشاف منطقة مشبوهة"
        short = (f"الذكاء الاصطناعي اكتشف منطقة قد تحتاج لمراجعة طبية. "
                 f"المنطقة المكتشفة تشكل {pct:.2f}% من الصورة.")
        detailed = (
            f"{friendly}\n\n"
            f"• النوع المتوقع (مبدئي): {prof.get('name_ar', '-')}\n"
            f"• حجم المنطقة: {pct:.2f}% من الصورة\n"
            f"• الكثافة: {analysis.clinical.get('intensity', '-')}\n"
            f"• التجانس: {analysis.clinical.get('homogeneity', '-')}\n\n"
            f"⚠️ هذا التحليل مبدئي. التشخيص النهائي يحتاج لتقييم طبيب أعصاب "
            f"وفحوصات إضافية (رنين بالصبغة، خزعة عند الحاجة)."
        )
        next_steps = ("يجب مراجعة طبيب الأعصاب في أقرب وقت لمناقشة الفحوصات "
                      "الإضافية وخطة العلاج.")
    else:
        friendly = prof.get("patient_friendly_en", "")
        headline = "⚠️ Suspicious area detected"
        short = (f"The AI detected an area that may need medical review. "
                 f"It covers {pct:.2f}% of the scan slice.")
        detailed = (
            f"{friendly}\n\n"
            f"• Suspected type (preliminary): {prof.get('name_en', '-')}\n"
            f"• Area size: {pct:.2f}% of slice\n"
            f"• Intensity: {analysis.clinical.get('intensity', '-')}\n"
            f"• Homogeneity: {analysis.clinical.get('homogeneity', '-')}\n\n"
            f"⚠️ This is a preliminary analysis. Final diagnosis requires "
            f"neurologist review and additional imaging (contrast MRI, "
            f"biopsy if needed)."
        )
        next_steps = ("Please consult a neurologist promptly to discuss "
                      "follow-up imaging and treatment planning.")

    return {"headline": headline, "short": short,
            "detailed": detailed, "next_steps": next_steps}


# ====================== Gemini =====================================
PROMPT_TEMPLATE_CT = """You are a STRICT urology decision-support assistant.

Rules:
- Use ONLY the GUIDELINES section below.
- If something is not in the guidelines, write "Not specified in retrieved guidelines."
- Be concise and clinical.

PATIENT / IMAGING PROFILE
{query}

RETRIEVED EAU GUIDELINE EXCERPTS
{context}

OUTPUT (markdown)
## 1. Imaging Summary
(2-3 lines summarising the stone characteristics.)

## 2. Treatment Options
For each option (MET, SWL, RIRS/URS, PCNL, observation): indication match,
brief evidence-based rationale, notable contraindications.

## 3. Recommended Next Step
One sentence: the option with the strongest guideline support.

## 4. Caveats
What additional clinical data would refine the recommendation."""


PROMPT_TEMPLATE_MRI = """You are a STRICT neuro-radiology decision-support assistant.

Rules:
- Use ONLY the GUIDELINES section below.
- Follow WHO CNS5 (2021) terminology.
- Final diagnosis belongs to the neuro-radiologist and pathologist.

IMAGING PROFILE
{query}

RETRIEVED WHO CNS5 EXCERPTS
{context}

OUTPUT (markdown)
## 1. Integrated Imaging Findings
(2-3 lines describing the lesion.)

## 2. Differential Diagnosis
Top 2-3 most likely diagnoses with brief rationale.

## 3. Recommended Next Steps
- Additional imaging sequences
- Molecular markers (IDH, MGMT, 1p/19q, etc.)
- Clinical workup priorities

## 4. Caveats
What is missing to make a confident diagnosis."""


def _call_gemini(prompt: str, api_key: Optional[str] = None,
                 model_name: str = config.GEMINI_MODEL,
                 temperature: float = 0.1) -> str:
    import google.generativeai as genai
    key = api_key or config.DEFAULT_GEMINI_KEY
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(temperature=temperature),
    )
    return resp.text


def generate_report(modality, clinical_or_analysis, n_stones=None,
                    gemini_api_key=None) -> tuple[str, list, str]:
    if modality == "ct":
        query = build_ct_query(clinical_or_analysis, n_stones or 0)
        tmpl = PROMPT_TEMPLATE_CT
    else:
        query = build_mri_query(clinical_or_analysis)
        tmpl = PROMPT_TEMPLATE_MRI

    raw_docs = retrieve(query, modality)
    docs = rerank(query, raw_docs)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = tmpl.format(query=query, context=context)

    try:
        report = _call_gemini(prompt, gemini_api_key)
    except Exception as e:
        report = _fallback_report(query, docs, str(e))
    return report, docs, query


def _fallback_report(query, docs, error=""):
    lines = []
    if error:
        lines.append(f"_LLM unavailable ({error}). Retrieved evidence:_\n")
    for i, d in enumerate(docs, 1):
        meta = getattr(d, "metadata", {}) or {}
        header = " > ".join(v for v in meta.values() if v) or f"Passage {i}"
        lines.append(f"### {i}. {header}")
        lines.append(d.page_content.strip())
        lines.append("")
    return "\n".join(lines)


# ====================== Chatbot ====================================
CHAT_SYSTEM_PROMPT = """You are a warm, professional medical assistant chatbot
embedded in a {modality_full} diagnostic system.

You are speaking DIRECTLY with the PATIENT (or their family). Your job is to
help them UNDERSTAND their condition in plain, reassuring language, while
remaining medically accurate.

GUIDELINES FOR YOUR REPLIES:
- Use SIMPLE language — avoid jargon unless the patient asks for technical detail
- Be WARM and reassuring, but never dismiss legitimate concerns
- Always emphasize that the final decision belongs to their treating doctor
- If asked "is it serious?", give an HONEST but compassionate answer
- The conversation may switch between Arabic and English — ALWAYS reply in
  the SAME language as the latest user message
- Use the PATIENT CONTEXT below to personalize your answers to THIS patient
- Cite guideline information when relevant (but in plain language)

PATIENT CONTEXT (this patient's current scan results):
{patient_context}

CLINICAL EXCERPTS FROM GUIDELINES (use these to ground your answers):
{excerpts}

IMPORTANT: If the patient asks something not covered in the guidelines,
say so honestly and suggest they ask their doctor.
"""


def detect_language(text: str) -> str:
    for ch in text:
        if "\u0600" <= ch <= "\u06FF":
            return "ar"
    return "en"


def chat_answer(user_message, modality, history, patient_context="",
                gemini_api_key=None) -> tuple[str, list]:
    raw = retrieve(user_message, modality, k=config.TOP_K_RETRIEVAL)
    docs = rerank(user_message, raw, top_k=config.TOP_K_RERANK)
    excerpts_block = "\n\n---\n\n".join(d.page_content for d in docs)

    modality_full = ("CT kidney stone (urology, EAU guidelines)"
                     if modality == "ct"
                     else "MRI brain tumor (neuro-oncology, WHO CNS5)")

    system_prompt = CHAT_SYSTEM_PROMPT.format(
        modality_full=modality_full,
        patient_context=patient_context or "(no current scan analyzed yet)",
        excerpts=excerpts_block or "(no excerpts retrieved)",
    )

    lang = detect_language(user_message)
    lang_hint = ("Reply in clear, simple Arabic that a non-medical person "
                 "can understand."
                 if lang == "ar"
                 else "Reply in clear, simple English that a non-medical "
                 "person can understand.")

    convo = [system_prompt, f"\nLANGUAGE: {lang_hint}\n"]
    for turn in history[-6:]:
        role = "Patient" if turn["role"] == "user" else "Assistant"
        convo.append(f"{role}: {turn['content']}")
    convo.append(f"Patient: {user_message}")
    convo.append("Assistant:")
    prompt = "\n".join(convo)

    try:
        reply = _call_gemini(prompt, gemini_api_key, temperature=0.3)
    except Exception as e:
        if lang == "ar":
            reply = (f"⚠️ تعذر الاتصال بالنموذج. هذه أهم المعلومات من الإرشادات:\n\n"
                     + "\n\n".join(f"**{i+1}.** {d.page_content[:300]}..."
                                   for i, d in enumerate(docs)))
        else:
            reply = (f"⚠️ Could not reach the LLM. Top retrieved passages:\n\n"
                     + "\n\n".join(f"**{i+1}.** {d.page_content[:300]}..."
                                   for i, d in enumerate(docs)))
    return reply, docs
