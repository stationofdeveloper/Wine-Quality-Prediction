"""
🔍 Explainer — Wine Quality Prediction
========================================
Combines SHAP (local feature attribution) with an LLM
to generate human-readable explanations of each prediction.

LLM priority:
    1. Groq API (llama-3.1-70b — free, fastest)
    2. Google Gemini Flash (free fallback)
    3. Rule-based fallback (no API needed)

Dependencies:
    pip install shap groq google-generativeai
"""

import os
import re
import numpy as np
from typing import Optional
import joblib

# ─── SHAP Explanation ─────────────────────────────────────────────────────────

_shap_explainer_cache = {}


def get_shap_values(model, X_sample, feature_names: list) -> list[tuple]:
    """
    Compute SHAP values for a single prediction sample.

    Returns list of (feature_name, shap_value) sorted by |impact|.
    """
    try:
        import shap

        model_key = id(model)
        if model_key not in _shap_explainer_cache:
            # TreeExplainer works for RF, GB, XGBoost — all our models
            _shap_explainer_cache[model_key] = shap.TreeExplainer(model)

        explainer  = _shap_explainer_cache[model_key]
        shap_vals  = explainer.shap_values(X_sample)

        if shap_vals.ndim > 1:
            shap_vals = shap_vals[0]

        paired = list(zip(feature_names, shap_vals))
        paired.sort(key=lambda x: abs(x[1]), reverse=True)
        return paired

    except ImportError:
        print("Install shap:  pip install shap")
        return []
    except Exception as e:
        print(f"SHAP error: {e}")
        return []


def format_shap_summary(shap_pairs: list[tuple], top_n: int = 5) -> str:
    """Format top SHAP features into a readable summary for LLM prompt."""
    lines = []
    for feat, val in shap_pairs[:top_n]:
        direction = "↑ increased" if val > 0 else "↓ decreased"
        lines.append(f"  • {feat}: {direction} the score by {abs(val):.3f}")
    return "\n".join(lines)


# ─── LLM Clients ─────────────────────────────────────────────────────────────

def _call_groq(prompt: str, system: str) -> Optional[str]:
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return None
    try:
        from groq import Groq
        client  = Groq(api_key=api_key)
        model   = os.getenv("GROQ_CHAT_MODEL", "llama-3.1-70b-versatile")
        resp    = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=400,
            temperature=0.4,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def _call_gemini(prompt: str, system: str) -> Optional[str]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model  = genai.GenerativeModel("gemini-1.5-flash")
        full   = f"{system}\n\n{prompt}"
        resp   = model.generate_content(full)
        return resp.text.strip()
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def _rule_based_explanation(prediction: float,
                             shap_pairs: list[tuple],
                             feature_values: dict) -> str:
    """Fallback explanation when no LLM API is available."""
    score = round(prediction, 1)
    if score >= 7.5:
        quality = "excellent"
    elif score >= 6.5:
        quality = "good"
    elif score >= 5.5:
        quality = "average"
    else:
        quality = "below average"

    top = shap_pairs[:3] if shap_pairs else []
    positive = [f for f, v in top if v > 0]
    negative = [f for f, v in top if v < 0]

    parts = [f"This wine scores {score}/8 — {quality} quality."]

    if positive:
        parts.append(
            f"Key strengths: {', '.join(positive)} contributed positively.")
    if negative:
        parts.append(
            f"Areas to improve: {', '.join(negative)} lowered the score.")

    va = feature_values.get("volatile acidity", 0)
    if va > 0.7:
        parts.append(
            "⚠️  High volatile acidity is the main concern — "
            "it gives a vinegar-like taste. Aim for below 0.5.")

    alc = feature_values.get("alcohol", 0)
    if alc < 10:
        parts.append("Low alcohol reduces body — consider a richer style.")
    elif alc > 14:
        parts.append("Very high alcohol may taste hot; ensure it's balanced.")

    return " ".join(parts)


# ─── Main explanation function ────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert wine sommelier and data scientist.
Your role is to explain machine learning predictions about wine quality
in clear, friendly language that both wine enthusiasts and non-experts
can understand. Be concise (3-5 sentences), specific about the chemical
factors, and always include one actionable improvement tip.
Do not use markdown formatting in your response."""


def explain_prediction(prediction: float,
                       feature_values: dict,
                       shap_pairs: list[tuple],
                       rag_context: str = "") -> str:
    """
    Generate a natural language explanation for a wine quality prediction.

    Args:
        prediction:     predicted quality score
        feature_values: dict of feature name → value
        shap_pairs:     output of get_shap_values()
        rag_context:    retrieved RAG chunks (optional)

    Returns:
        Human-readable explanation string.
    """
    shap_summary = format_shap_summary(shap_pairs, top_n=5)

    # Build a few important feature highlights for the prompt
    highlights = []
    va = feature_values.get("volatile acidity", 0)
    alc = feature_values.get("alcohol", 0)
    ph = feature_values.get("pH", 0)
    sulph = feature_values.get("sulphates", 0)

    if va > 0.7:    highlights.append(f"volatile acidity is HIGH ({va:.2f}) — vinegar risk")
    if va < 0.3:    highlights.append(f"volatile acidity is LOW ({va:.2f}) — clean aroma")
    if alc > 13.5:  highlights.append(f"alcohol is HIGH ({alc:.1f}%) — full-bodied")
    if alc < 10:    highlights.append(f"alcohol is LOW ({alc:.1f}%) — light body")
    if sulph > 1.0: highlights.append(f"sulphates are HIGH ({sulph:.2f}) — enhanced preservation")
    if ph < 3.2:    highlights.append(f"pH is LOW ({ph:.2f}) — very acidic, crisp")
    if ph > 3.7:    highlights.append(f"pH is HIGH ({ph:.2f}) — low acidity, flat risk")

    feature_highlight_str = "; ".join(highlights) if highlights else "all features within typical range"

    prompt = f"""
A machine learning model predicted a red wine quality score of {prediction:.2f} out of 8.

Key chemical property notes:
{feature_highlight_str}

SHAP feature attribution (what drove this prediction):
{shap_summary if shap_summary else "SHAP analysis not available."}

{"Relevant wine chemistry background:" if rag_context else ""}
{rag_context[:800] if rag_context else ""}

Please explain this prediction in 3-5 friendly sentences. Tell the user:
1. What the score means for wine quality
2. Which 1-2 chemical factors most influenced this score and why
3. One specific, practical improvement tip to raise the score
"""

    # Try LLM APIs in order of preference
    result = _call_groq(prompt, SYSTEM_PROMPT)
    if not result:
        result = _call_gemini(prompt, SYSTEM_PROMPT)
    if not result:
        result = _rule_based_explanation(prediction, shap_pairs, feature_values)

    return result


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_features = {
        "fixed acidity": 8.5, "volatile acidity": 0.72, "citric acid": 0.3,
        "residual sugar": 2.0, "chlorides": 0.08, "free sulfur dioxide": 15,
        "total sulfur dioxide": 70, "density": 0.998, "pH": 3.4,
        "sulphates": 0.6, "alcohol": 11.0
    }
    explanation = explain_prediction(
        prediction=5.2,
        feature_values=sample_features,
        shap_pairs=[
            ("volatile acidity", -0.45),
            ("alcohol", +0.21),
            ("sulphates", +0.12),
        ],
    )
    print(explanation)