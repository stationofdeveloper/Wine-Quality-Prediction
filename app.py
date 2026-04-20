"""
🍷 Wine AI Sommelier — Unified App
====================================
Integrates:
  ✅  ML Quality Prediction  (Random Forest / GB / XGBoost)
  🌲  Agentic Vectorless RAG Chatbot  (PageIndex-style, 3 wine books)
  🧠  SHAP Explainability + LLM narration  (Groq LLaMA)
  🎤  Voice Input  (Groq Whisper)
  🔊  Voice Output  (gTTS)
  🍾  Similar Wine Recommender
  🚨  Anomaly Detection

Run:  streamlit run app.py
"""

import os, io, json, time, re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🍷 Wine AI Sommelier",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
  /* ── Global ── */
  body { font-family: 'Segoe UI', sans-serif; }
  .block-container { padding-top: 1rem; }

  /* ── Nav tabs ── */
  .nav-bar {
    display: flex; gap: 8px; margin-bottom: 20px;
    border-bottom: 2px solid #e8b4b8; padding-bottom: 8px;
  }
  .nav-btn {
    padding: 8px 22px; border-radius: 20px; border: none; cursor: pointer;
    font-weight: 600; font-size: 0.92rem; transition: all 0.2s;
  }
  .nav-btn.active { background: #7c0a02; color: white; }
  .nav-btn.inactive {
    background: #fdf0f0; color: #7c0a02;
    border: 1px solid #e8b4b8;
  }

  /* ── Cards ── */
  .metric-card {
    background: #fff8f8; border: 1.5px solid #e8b4b8;
    border-radius: 12px; padding: 14px 18px; text-align: center;
  }
  .metric-card h3 { margin: 0; color: #7c0a02; font-size: 1.35rem; }
  .metric-card p  { margin: 4px 0 0; color: #666; font-size: 0.82rem; }

  /* ── Prediction badge ── */
  .quality-badge { font-size: 3rem; font-weight: 800; text-align: center; margin: 8px 0; }

  /* ── Chat bubbles ── */
  .bubble-user {
    background: #7c0a02; color: white;
    border-radius: 18px 18px 4px 18px;
    padding: 12px 18px; margin: 8px 0 8px 80px;
    font-size: 0.95rem; line-height: 1.5;
  }
  .bubble-ai {
    background: white; border: 1.5px solid #e8b4b8;
    border-radius: 18px 18px 18px 4px;
    padding: 14px 18px; margin: 8px 80px 8px 0;
    font-size: 0.95rem; line-height: 1.6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .bubble-system {
    background: #f0fff4; border-left: 3px solid #38a169;
    border-radius: 6px; padding: 8px 12px;
    margin: 3px 0; font-size: 0.85rem; color: #2d7f4f;
  }
  .citation {
    background: #fff8e7; border: 1px solid #f0c040;
    border-radius: 8px; padding: 8px 14px; margin-top: 8px;
    font-size: 0.83rem; color: #5a3e00;
  }
  .tip-box {
    background: #f0fff4; border-left: 4px solid #38a169;
    border-radius: 6px; padding: 9px 13px; margin: 4px 0; font-size: 0.9rem;
  }
  .warn-box {
    background: #fffbf0; border-left: 4px solid #d69e2e;
    border-radius: 6px; padding: 9px 13px; margin: 4px 0; font-size: 0.9rem;
  }

  /* ── Source info cards (shown before LLM answer) ── */
  .source-card {
    background: linear-gradient(135deg, #fdf6f0 0%, #fef9f4 100%);
    border: 1.5px solid #e8c8a0;
    border-left: 4px solid #7c0a02;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    font-size: 0.88rem;
    line-height: 1.5;
    box-shadow: 0 2px 6px rgba(124,10,2,0.06);
  }
  .source-card .sc-header {
    font-weight: 700; color: #7c0a02;
    font-size: 0.92rem; margin-bottom: 6px;
  }
  .source-card .sc-meta {
    color: #555; font-size: 0.84rem; margin: 2px 0;
  }
  .source-card .sc-preview {
    color: #666; font-size: 0.82rem; font-style: italic;
    margin-top: 6px; padding-top: 6px;
    border-top: 1px dashed #ddd;
  }
  .source-cards-label {
    font-size: 0.82rem; font-weight: 600;
    color: #7c0a02; text-transform: uppercase;
    letter-spacing: 0.5px; margin-bottom: 4px;
  }

  .page-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #7c0a02, #c0392b);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    text-align: center; margin-bottom: 2px;
  }
  .page-sub { text-align: center; color: #888; font-size: 0.92rem; margin-bottom: 18px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers — FIXED Gemini + Groq (with environment variable model selection)
# ─────────────────────────────────────────────────────────────────────────────

def call_groq(messages: list[dict], model: str = None,
              max_tokens: int = 1200) -> str | None:
    """
    Call Groq Chat API. Uses environment variable GROQ_CHAT_MODEL if no model provided.
    Returns None on failure.
    """
    if not GROQ_API_KEY:
        return None
    try:
        from groq import Groq
        if model is None:
            # Read model from environment, with a sensible default
            model = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-3.3-70b-versatile")
        client = Groq(api_key=GROQ_API_KEY)
        resp = client.chat.completions.create(
            model=model, messages=messages,
            max_tokens=max_tokens, temperature=0.35,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq error: {e}")
        return None


def call_gemini(prompt: str, system: str = "", max_tokens: int = 1200) -> str | None:
    """
    Fixed Gemini call using the new google-genai SDK.
    Tries multiple model names for robustness.
    """
    if not GEMINI_API_KEY:
        return None
    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=GEMINI_API_KEY)
        full_prompt = f"{system}\n\n{prompt}" if system else prompt

        # Models to try in order of preference
        model_names = ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash"]

        for model_name in model_names:
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=full_prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=0.35,
                    )
                )
                if response and response.text:
                    return response.text.strip()
            except Exception:
                continue
        return None
    except ImportError:
        return None
    except Exception as e:
        print(f"Gemini error: {e}")
        return None


def llm_answer(prompt: str, system: str = "",
               messages: list[dict] | None = None,
               max_tokens: int = 1200) -> str:
    """Try Groq first, fallback to Gemini, fallback to rule-based."""
    if messages is None:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

    result = call_groq(messages, max_tokens=max_tokens)
    if result:
        return result

    result = call_gemini(prompt, system, max_tokens)
    if result:
        return result

    return "⚠️  Both LLM APIs unavailable. Check your API keys in the .env file."


# ─────────────────────────────────────────────────────────────────────────────
# Voice helpers
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes, filename: str = "audio.wav") -> str:
    if not GROQ_API_KEY:
        return ""
    try:
        import tempfile
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes); tmp_path = tmp.name
        with open(tmp_path, "rb") as f:
            # Use environment variable for whisper model
            whisper_model = os.getenv("GROQ_WHISPER_MODEL", "whisper-large-v3")
            t = client.audio.transcriptions.create(
                file=(filename, f.read()), model=whisper_model,
                response_format="text", language="en",
            )
        os.unlink(tmp_path)
        return str(t).strip()
    except Exception as e:
        return f"⚠️ {e}"


def tts(text: str) -> bytes | None:
    try:
        from gtts import gTTS
        spoken = text[:350].split("📚 Sources:")[0].strip()
        buf = io.BytesIO()
        gTTS(text=spoken, lang="en").write_to_fp(buf)
        buf.seek(0); return buf.read()
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Load ML artifacts
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_ml():
    files = ["best_wine_model.pkl", "wine_scaler.pkl",
             "feature_names.pkl", "model_meta.pkl"]
    if any(not Path(f).exists() for f in files):
        return None, None, None, None
    return (joblib.load("best_wine_model.pkl"), joblib.load("wine_scaler.pkl"),
            joblib.load("feature_names.pkl"),   joblib.load("model_meta.pkl"))


@st.cache_data
def load_df():
    if not Path("winequality-red.csv").exists():
        return None
    df = pd.read_csv("winequality-red.csv")
    df["acidity_ratio"]   = df["fixed acidity"]      / (df["volatile acidity"]     + 1e-9)
    df["so2_ratio"]       = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
    df["alcohol_density"] = df["alcohol"] / df["density"]
    return df


@st.cache_resource
def load_rag_agent():
    try:
        import wine_rag_agent as agent
        return agent
    except Exception:
        return None


@st.cache_resource
def load_rag_indexer():
    try:
        import wine_rag_indexer as idx
        return idx
    except Exception:
        return None


model, scaler, feature_names, meta = load_ml()
df = load_df()
agent_mod   = load_rag_agent()
indexer_mod = load_rag_indexer()

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────
if "page"           not in st.session_state: st.session_state.page           = "predictor"
if "chat_messages"  not in st.session_state: st.session_state.chat_messages  = []
if "agent_history"  not in st.session_state: st.session_state.agent_history  = []
if "step_log"       not in st.session_state: st.session_state.step_log       = []
if "tts_on"         not in st.session_state: st.session_state.tts_on         = True
if "last_audio"     not in st.session_state: st.session_state.last_audio     = None

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
# --- Enhanced Sidebar UI ---
with st.sidebar:
    # 1. Header with branding
    st.markdown("""
        <div style="display:flex; align-items: center; justify-content: center; text-align: center;">
            <h1 style='font-size: 2rem;'>🍷</h1>
            <h2 style=''>Sommelier AI</h2>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")

    # 2. Refined Navigation
    st.markdown("### 🧭 Exploration")
    
    # Using a cleaner dictionary for labels
    pages = {
        "predictor": "🤖 Quality Predictor",
        "chatbot":   "💬 AI Sommelier Chat",
        "eda":       "📊 EDA Dashboard",
        "models":    "📈 Model Performance",
    }

    # Custom CSS for buttons to make them feel more like a menu
    for pg, label in pages.items():
        is_active = st.session_state.page == pg
        if st.button(
            label, 
            key=f"nav_{pg}", 
            use_container_width=True,
            type="primary" if is_active else "secondary"
        ):
            st.session_state.page = pg
            st.rerun()

    st.markdown("---")

    # 3. New Aesthetic Feature: Session Info (Replacing API/Book info)
    # This keeps the sidebar from looking empty while adding "Sommelier" flair
    with st.container():
        st.markdown("### 🏷️ Tasting Notes")
        st.info("Current Session: **Active**")
        
        # Subtle stats or tips
        st.markdown("""
        <div style="background-color: rgba(100, 100, 100, 0.1); padding: 15px; border-radius: 10px; border-left: 5px solid #722f37;">
            <small><b>Pro Tip:</b><br>Higher fixed acidity and lower volatile acidity are strong indicators of premium wine quality in our current model.</small>
        </div>
        """, unsafe_allow_html=True)

    # 4. Bottom Footer
    # st.markdown("<br>" * 5, unsafe_allow_html=True)
    # st.markdown("---")
    # st.caption("✨ Powered by Advanced Enology Analytics")

# ─────────────────────────────────────────────────────────────────────────────
# RAW FEATURES
# ─────────────────────────────────────────────────────────────────────────────
RAW_FEATURES = [
    "fixed acidity","volatile acidity","citric acid","residual sugar",
    "chlorides","free sulfur dioxide","total sulfur dioxide",
    "density","pH","sulphates","alcohol"
]


def build_input(vals: dict) -> pd.DataFrame:
    row = dict(vals)
    row["acidity_ratio"]   = row["fixed acidity"]      / (row["volatile acidity"]     + 1e-9)
    row["so2_ratio"]       = row["free sulfur dioxide"] / (row["total sulfur dioxide"] + 1e-9)
    row["alcohol_density"] = row["alcohol"] / row["density"]
    return pd.DataFrame([row])[feature_names]


def quality_meta(score: float) -> tuple[str, str]:
    if   score >= 7.5: return "🏆 Excellent",   "#1a7a1a"
    elif score >= 6.5: return "😊 Good",         "#2171b5"
    elif score >= 5.5: return "🙂 Average",      "#d95f02"
    else:              return "😕 Below Average","#a50026"


# ─────────────────────────────────────────────────────────────────────────────
# ██  PAGE 1 — Quality Predictor
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "predictor":
    st.markdown('<p class="page-title">🍷 Wine Quality Predictor</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Adjust chemical properties → get instant AI-powered quality prediction</p>',
                unsafe_allow_html=True)

    if model is None or df is None:
        st.error("⚠️  Run `python model_training.py` first to generate .pkl files.")
        st.stop()

    # Model info bar
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip([c1,c2,c3,c4],
        ["Best Model","R² Score","MAE","RMSE"],
        [meta["label"], f"{meta['R2']:.3f}", f"{meta['MAE']:.3f}", f"{meta['RMSE']:.3f}"]):
        with col:
            st.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{lbl}</p></div>',
                        unsafe_allow_html=True)
    st.markdown("---")

    # ── Sliders ───────────────────────────────────────────────────────────────
    with st.expander("🎛️ Wine Chemical Properties", expanded=True):
        cols = st.columns(3)
        user_vals = {}
        for i, feat in enumerate(RAW_FEATURES):
            lo  = float(df[feat].min())
            hi  = float(df[feat].max())
            mid = float(df[feat].mean())
            user_vals[feat] = cols[i % 3].slider(
                feat.title(), lo, hi, mid,
                step=round((hi - lo) / 100, 4), key=f"pred_{feat}"
            )

    # ── Prediction ────────────────────────────────────────────────────────────
    inp_df = build_input(user_vals)
    inp_sc = scaler.transform(inp_df)
    pred   = round(float(model.predict(inp_sc)[0]), 2)
    qlbl, qclr = quality_meta(pred)

    left, right = st.columns([1, 1])

    with left:
        # Anomaly detection
        try:
            from recommender import detect_anomaly, improvement_tips
            is_anom, anom_msg = detect_anomaly(user_vals, df, RAW_FEATURES)
            # st.markdown(
            #     f'<div class="{"warn-box" if is_anom else "tip-box"}">{anom_msg}</div>',
            #     unsafe_allow_html=True
            # )
        except ImportError:
            pass

        # Input table
        st.subheader("Input Summary")
        disp = pd.DataFrame({"Feature": RAW_FEATURES,
                              "Value": [round(user_vals[f], 4) for f in RAW_FEATURES]})
        st.dataframe(disp.set_index("Feature"), width="stretch", height=360)  # ← fixed

    with right:
        # Score badge
        st.subheader("Predicted Quality")
        st.markdown(f"""
          <div style="border:2px solid {qclr};border-radius:14px;
                      padding:20px;text-align:center;background:#fafafa;">
            <div class="quality-badge" style="color:{qclr};">{pred}</div>
            <div style="font-size:1.3rem;color:{qclr};font-weight:600;">{qlbl}</div>
            <div style="color:#888;margin-top:6px;">Scale: 3 (poor) — 8 (excellent)</div>
          </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_g, ax_g = plt.subplots(figsize=(5, 1.0))
        for lo_b, hi_b, c in [(3,4.2,"#d73027"),(4.2,5.4,"#f46d43"),
                               (5.4,6.4,"#fee090"),(6.4,7.2,"#74add1"),(7.2,9,"#313695")]:
            ax_g.barh(0, hi_b-lo_b, left=lo_b, color=c, height=0.45, alpha=0.85)
        ax_g.axvline(pred, color="black", lw=3, linestyle="--")
        ax_g.text(pred, 0.3, f"{pred}", ha="center", fontsize=9, fontweight="bold")
        ax_g.set_xlim(3,9); ax_g.set_yticks([]); ax_g.set_xlabel("Quality Score")
        plt.tight_layout()
        st.pyplot(fig_g, width="stretch")  # ← fixed

        # ── SHAP + LLM Explanation ───────────────────────────────────────────
        st.subheader("🧠 AI Explanation")
        with st.spinner("Analysing …"):
            try:
                from explainer import get_shap_values
                shap_pairs = get_shap_values(model, inp_sc, feature_names)

                if shap_pairs:
                    top_f = [p[0] for p in shap_pairs[:6]]
                    top_v = [p[1] for p in shap_pairs[:6]]
                    clrs  = ["#d73027" if v < 0 else "#1a9850" for v in top_v]
                    fig_s, ax_s = plt.subplots(figsize=(5, 2.8))
                    ax_s.barh(top_f[::-1], top_v[::-1], color=clrs[::-1],
                              edgecolor="black", alpha=0.8)
                    ax_s.axvline(0, color="black", lw=0.8)
                    ax_s.set_xlabel("SHAP impact"); ax_s.set_title("Feature Contributions")
                    plt.tight_layout()
                    st.pyplot(fig_s, width="stretch")  # ← fixed
            except ImportError:
                shap_pairs = []

            # LLM explanation
            va    = user_vals.get("volatile acidity", 0)
            alc   = user_vals.get("alcohol", 0)
            sulph = user_vals.get("sulphates", 0)
            notes = []
            if va > 0.7:   notes.append(f"volatile acidity is HIGH ({va:.2f}) — vinegar taste risk")
            if va < 0.3:   notes.append(f"volatile acidity is LOW ({va:.2f}) — very clean aroma")
            if alc > 13.5: notes.append(f"alcohol is HIGH ({alc:.1f}%) — full-bodied")
            if alc < 10:   notes.append(f"alcohol is LOW ({alc:.1f}%) — light body")
            if sulph > 1:  notes.append(f"sulphates HIGH ({sulph:.2f}) — well preserved")
            notes_str = "; ".join(notes) if notes else "within typical ranges"

            shap_txt = "\n".join(
                f"  {f}: {'↑' if v>0 else '↓'} {abs(v):.3f}" for f, v in (shap_pairs[:4] if shap_pairs else [])
            ) or "  not available"

            explanation = llm_answer(
                prompt=(
                    f"Red wine predicted quality: {pred}/8.\n"
                    f"Chemical notes: {notes_str}\n"
                    f"SHAP impacts:\n{shap_txt}\n\n"
                    "In 3-4 clear sentences: what this score means, which 1-2 factors "
                    "drove it most, and one specific improvement tip. No markdown."
                ),
                system=(
                    "You are a wine sommelier and data scientist. "
                    "Give concise, friendly, precise explanations. No bullet points."
                ),
                max_tokens=300,
            )
            st.info(explanation)

            # Voice output
            if st.session_state.tts_on:
                audio = tts(explanation)
                if audio:
                    st.audio(audio, format="audio/mp3")
                    # No need to store in session state (avoids media file error)

    

    # ── Similar wines ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("🍾 Similar High-Quality Wines")
    try:
        from recommender import find_similar_wines
        similar = find_similar_wines(user_vals, df, feature_names, scaler,
                                     min_quality=7.0, top_n=5)
        if not similar.empty:
            st.dataframe(
                similar.style.format(precision=3)
                       .background_gradient(subset=["similarity %"], cmap="Greens"),
                width="stretch"  # ← fixed
            )
        else:
            st.info("No wines found at quality ≥ 7. Try different input values.")
    except ImportError:
        st.info("Install recommender.py for similar wine finder.")


    # ── Improvement tips ──────────────────────────────────────────────────────
    st.markdown("---")
    # st.subheader("💡 Data-Driven Improvement Tips")
    # try:
    #     from recommender import improvement_tips
    #     tips = improvement_tips(user_vals, df)
    #     tcols = st.columns(2)
    #     for i, tip in enumerate(tips):
    #         tcols[i % 2].markdown(f'<div class="tip-box text-primary" style="color:black;">{tip}</div>', unsafe_allow_html=True)
    # except ImportError:
    #     st.info("Install recommender.py for improvement tips.")
    st.markdown("### 💡 Data-Driven Improvement Tips")
    try:
        from recommender import improvement_tips
        tips = improvement_tips(user_vals, df)
        
        # Create columns for a side-by-side grid layout
        tcols = st.columns(2)
        
        for i, tip in enumerate(tips):
            # 1. Strip any markdown or literal bullet characters from the start of the string
            clean_tip = tip.lstrip('*-•·1234567890. ').strip()
            
            # 2. Enhanced UI: Sleek, wine-themed card design
            html_card = f"""
            <div style="
                background-color: #fcfcfc;
                border-left: 4px solid #722f37;
                padding: 15px;
                border-radius: 6px;
                margin-bottom: 15px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
                font-size: 0.95rem;
                color: #333;
                line-height: 1.5;
            ">
                <span style="color: #722f37; font-weight: bold; margin-right: 5px;">
                    ✨ Tip {i+1}:
                </span> 
                {clean_tip}
            </div>
            """
            
            tcols[i % 2].markdown(html_card, unsafe_allow_html=True)
            
    except ImportError:
        st.info("Install recommender.py for improvement tips.")

    # ── Ask chatbot button ─────────────────────────────────────────────────────
    st.markdown("---")
    if st.button("💬 Ask AI Sommelier about this wine →", width="stretch", type="primary"):  # ← fixed
        prefill = (
            f"My wine scored {pred}/8. Volatile acidity={va:.2f}, "
            f"alcohol={alc:.1f}%, sulphates={sulph:.2f}. "
            "What should I do to improve it? What food pairs well with it?"
        )
        st.session_state["voice_prefill"] = prefill
        st.session_state.page = "chatbot"
        st.rerun()

    


# ─────────────────────────────────────────────────────────────────────────────
# ██  PAGE 2 — AI Sommelier Chatbot
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "chatbot":
    # ── Header ─────────────────────────────────────────────────────────────────
    # st.markdown('<p class="page-title style="padding-top:20px">🍷 Wine AI Sommelier Chat</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-title style="margin:20px 0px 0px 0px">🍷 Wine AI Sommelier Chat</p>', unsafe_allow_html=True)
    # st.markdown(
    #     '<p class="page-sub">Agentic Vectorless RAG over 3 wine books · '
    #     'Groq LLaMA 3.1 · Voice in/out</p>',
    #     unsafe_allow_html=True
    # )

    st.markdown("#### 📚 Active Knowledge Base")
    st.info("""
        Our AI Sommelier is currently trained on these definitive texts:
        * 📖 Chemistry and Biochemistry of Winemaking, Wine Stabilization and Aging
        * 🌍 Perceptions of wine quality
        * 🍷 Wine Tasting: A Professional Handbook
    """)
    # Book status banner
    # if indexer_mod:
    #     status = indexer_mod.get_index_status()
    #     indexed = [info["emoji"] + " " + info["title"][:30]
    #                for info in status.values() if info["indexed"]]
    #     if indexed:
    #         st.success(f"📚 Knowledge base: {' · '.join(indexed)}")
    #     else:
    #         st.warning("⚠️  No books indexed yet. Use the sidebar to index your wine PDFs.")

    # ── Suggested questions ────────────────────────────────────────────────────
    # Hide if there are existing messages OR if a suggestion was just clicked
    if not st.session_state.chat_messages and not st.session_state.get("submit_prompt"):
        st.markdown("**✨ Try asking:**")
        suggestions = [
            "What is the role of volatile acidity in wine quality?",
            "How does alcohol content affect wine perception?",
            "Explain malolactic fermentation and why it matters.",
            "What food pairs best with a high-quality red wine?",
            "What chemical factors most predict high wine scores?",
            "How do professional tasters evaluate wine quality?",
            "What is the ideal pH range for red wine?",
            "How does sulphur dioxide preserve wine during aging?",
        ]
        sg_cols = st.columns(2)
        for i, s in enumerate(suggestions):
            if sg_cols[i % 2].button(f"💬 {s}", key=f"sg_{i}", use_container_width=True):
                st.session_state["submit_prompt"] = s
                st.rerun()

    # ── Helper: render source cards ─────────────────────────────────────────────
    def _render_source_cards(sources: list[dict]):
        """Render retrieved-source info cards above the LLM answer."""
        if not sources:
            return
        st.markdown('<p class="source-cards-label">📖 Retrieved from knowledge base</p>',
                    unsafe_allow_html=True)
        for src in sources:
            preview = src.get("preview", "")[:250]
            if len(src.get("preview", "")) > 250:
                preview += "…"
            card_html = (
                f'<div class="source-card">'
                f'<div class="sc-header">{src.get("book_emoji", "📖")} {src.get("book_title", "Unknown Book")}</div>'
                f'<div class="sc-meta">📑 <b>Section:</b> {src.get("section", "N/A")}</div>'
                f'<div class="sc-meta">📄 <b>Pages:</b> {src.get("pages", "?")}</div>'
                f'<div class="sc-preview">👁️ {preview}</div>'
                f'</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

    # ── Chat History Rendering (Gemini Style) ──────────────────────────────────
    for msg in st.session_state.chat_messages:
        avatar = "👤" if msg["role"] == "user" else "🍷"
        with st.chat_message(msg["role"], avatar=avatar):
            content = msg["content"]
            
            if msg["role"] == "assistant":
                # Show source cards above the answer (if any were retrieved)
                _render_source_cards(msg.get("sources", []))

                if "📚 Sources:" in content:
                    main, cite = content.split("📚 Sources:", 1)
                    st.markdown(main.strip())
                    st.caption(f"📚 Sources: {cite.strip()}")
                else:
                    st.markdown(content)
                
                if msg.get("audio"):
                    st.audio(msg["audio"], format="audio/mp3")
            else:
                st.markdown(content)

    # ── Input & Processing Logic ───────────────────────────────────────────────
    user_q = st.chat_input("Ask anything about wine... e.g. What role do tannins play?")
    
    # Check if a suggestion button was clicked
    if "submit_prompt" in st.session_state and st.session_state["submit_prompt"]:
        user_q = st.session_state.pop("submit_prompt")

    if user_q:
        # 1. Immediately show user input
        st.session_state.chat_messages.append({"role": "user", "content": user_q.strip()})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_q.strip())

        # 2. Show Assistant 'thinking' block
        st.session_state.step_log = []
        with st.chat_message("assistant", avatar="🍷"):
            prog = st.empty()
            retrieved_sources = []   # collect RETRIEVE:: events

            def cb(msg):
                # Detect structured RETRIEVE:: events from the RAG agent
                if msg.startswith("RETRIEVE::"):
                    try:
                        import json as _json
                        payload = _json.loads(msg[len("RETRIEVE::"):])
                        retrieved_sources.append(payload)
                        # Show a brief progress note for each retrieval
                        prog.markdown(
                            f"*📖 Found relevant content in "
                            f"**{payload.get('book_title','')}** "
                            f"(p. {payload.get('pages','')})…*"
                        )
                    except Exception:
                        pass
                    return
                st.session_state.step_log.append(msg)
                prog.markdown(f"*{msg}...*")

            audio_data = None
            with st.spinner("Consulting the cellar..."):
                if agent_mod:
                    try:
                        final, updated = agent_mod.answer(
                            user_q.strip(),
                            chat_history=st.session_state.agent_history[-6:],
                            stream_callback=cb,
                        )
                        st.session_state.agent_history = [
                            m for m in updated if m["role"] != "system"
                        ]
                    except Exception as e:
                        final = f"⚠️ Agent error: {e}"
                else:
                    final = llm_answer(
                        user_q.strip(),
                        system="You are an expert wine sommelier. Give helpful, accurate answers.",
                        max_tokens=600,
                    )
                    final += "\n\n📚 Sources: (Wine books not indexed — run setup_rag.py)"

                if st.session_state.get("tts_on", False):
                    audio_data = tts(final)

            prog.empty()

            # Show source cards BEFORE the answer
            _render_source_cards(retrieved_sources)

            # Write answer
            if "📚 Sources:" in final:
                main_txt, cite_txt = final.split("📚 Sources:", 1)
                st.markdown(main_txt.strip())
                st.caption(f"📚 Sources: {cite_txt.strip()}")
            else:
                st.markdown(final)

            if audio_data:
                st.audio(audio_data, format="audio/mp3")

        # 3. Save to state (include sources for history re-rendering)
        st.session_state.chat_messages.append({
            "role": "assistant", 
            "content": final, 
            "sources": retrieved_sources,
            "audio": audio_data
        })

# ─────────────────────────────────────────────────────────────────────────────
# ██  PAGE 3 — EDA Dashboard
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "eda":
    st.markdown('<p class="page-title">📊 EDA Dashboard</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Exploratory Data Analysis — Red Wine Quality Dataset</p>',
                unsafe_allow_html=True)

    if df is None:
        st.error("Dataset not found.")
        st.stop()

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    for col, lbl, val in zip([c1,c2,c3,c4],
        ["Total Samples","Features","Quality Range","Mean Quality"],
        [len(df), len(RAW_FEATURES), f"{df['quality'].min()}-{df['quality'].max()}",
         f"{df['quality'].mean():.2f}"]):
        with col:
            st.markdown(f'<div class="metric-card"><h3>{val}</h3><p>{lbl}</p></div>',
                        unsafe_allow_html=True)
    st.markdown("---")

    t1, t2, t3 = st.tabs(["🔥 Correlations","📦 Distributions","🔗 Feature vs Quality"])

    with t1:
        fig, ax = plt.subplots(figsize=(13, 8))
        mask = np.triu(np.ones_like(df.corr(numeric_only=True), dtype=bool))
        sns.heatmap(df.corr(numeric_only=True), mask=mask, annot=True, fmt=".2f",
                    cmap="coolwarm", linewidths=0.4, annot_kws={"size": 8}, ax=ax)
        ax.set_title("Correlation Matrix", fontsize=13)
        st.pyplot(fig, width="stretch")  # ← fixed

    with t2:
        col_a, col_b = st.columns(2)
        fig_q, ax_q = plt.subplots(figsize=(5, 3))
        df["quality"].value_counts().sort_index().plot(
            kind="bar", color="#7c0a02", edgecolor="black", alpha=0.8, ax=ax_q)
        ax_q.set_title("Quality Distribution"); ax_q.set_xlabel("Quality")
        plt.xticks(rotation=0)
        col_a.pyplot(fig_q, width="stretch")  # ← fixed

        feat_sel = col_b.selectbox("Feature histogram", RAW_FEATURES)
        fig_h, ax_h = plt.subplots(figsize=(5, 3))
        df[feat_sel].hist(bins=25, color="#2171b5", edgecolor="black", alpha=0.8, ax=ax_h)
        ax_h.set_title(f"{feat_sel} distribution")
        col_b.pyplot(fig_h, width="stretch")  # ← fixed

    with t3:
        feat_box = st.selectbox("Select feature", RAW_FEATURES, key="box_feat")
        fig_b, ax_b = plt.subplots(figsize=(10, 4))
        sns.boxplot(x="quality", y=feat_box, data=df, palette="viridis", ax=ax_b)
        ax_b.set_title(f"{feat_box} vs Wine Quality")
        st.pyplot(fig_b, width="stretch")  # ← fixed    

        corr_vals = df.corr(numeric_only=True)["quality"].drop("quality").abs().sort_values(ascending=False)
        fig_c, ax_c = plt.subplots(figsize=(10, 3))
        corr_vals.plot(kind="bar", color="#7c0a02", edgecolor="black", alpha=0.8, ax=ax_c)
        ax_c.set_title("Feature importance in Wine Quality")
        plt.xticks(rotation=45, ha="right")
        st.pyplot(fig_c, width="stretch")  # ← fixed


# ─────────────────────────────────────────────────────────────────────────────
# ██  PAGE 4 — Model Performance
# ─────────────────────────────────────────────────────────────────────────────
elif st.session_state.page == "models":
    st.markdown('<p class="page-title">📈 Model Performance</p>', unsafe_allow_html=True)
    st.markdown('<p class="page-sub">Before & After Fine-Tuning · All 3 Models</p>',
                unsafe_allow_html=True)

    if model is None:
        st.error("No model found. Run model_training.py first.")
        st.stop()

    # Comparison chart
    if Path("model_comparison.png").exists():
        st.subheader("All Models — Before & After Fine-Tuning")
        st.image("model_comparison.png", width="stretch")  # ← fixed (use_container_width replaced with width)
    else:
        st.info("Run `python model_training.py` to generate the model comparison chart.")

    # Dataset preview
    st.markdown("---")
    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), width="stretch")  # ← fixed
    st.download_button("⬇️ Download CSV", df.to_csv(index=False),
                       "winequality-red.csv", "text/csv")