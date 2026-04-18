"""
🎤 Voice Handler — Wine Quality Prediction
==========================================
Speech-to-Text  : Groq Whisper API (free tier)
Text-to-Speech  : gTTS — Google Text-to-Speech (free, no API key)

Dependencies:
    pip install groq gtts pydub
"""

import os
import io
import re
import tempfile
from pathlib import Path
from typing import Optional


# ─── Text-to-Speech ──────────────────────────────────────────────────────────

def text_to_speech(text: str) -> Optional[bytes]:
    """
    Convert text to MP3 audio bytes using gTTS (free, no API key).
    Returns raw MP3 bytes or None on failure.
    """
    try:
        from gtts import gTTS
        tts = gTTS(text=text, lang="en", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        return buf.read()
    except ImportError:
        print("Install gTTS:  pip install gTTS")
        return None
    except Exception as e:
        print(f"TTS error: {e}")
        return None


# ─── Speech-to-Text via Groq Whisper ─────────────────────────────────────────

def transcribe_audio(audio_bytes: bytes,
                     filename: str = "audio.wav") -> Optional[str]:
    """
    Transcribe audio bytes using Groq Whisper API (free tier).
    Requires GROQ_API_KEY in environment / .env file.
    """
    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        return "⚠️  GROQ_API_KEY not set. Add it to your .env file."

    try:
        from groq import Groq
        client = Groq(api_key=api_key)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            transcript = client.audio.transcriptions.create(
                file=(filename, f.read()),
                model="whisper-large-v3",
                response_format="text",
                language="en",
            )

        os.unlink(tmp_path)
        return str(transcript).strip()

    except ImportError:
        return "⚠️  Install groq:  pip install groq"
    except Exception as e:
        return f"⚠️  Transcription failed: {e}"


# ─── Parse spoken wine features ───────────────────────────────────────────────

FEATURE_ALIASES = {
    "fixed acidity":         ["fixed acidity", "fixed acid", "tartaric"],
    "volatile acidity":      ["volatile acidity", "volatile acid", "acetic"],
    "citric acid":           ["citric acid", "citric"],
    "residual sugar":        ["residual sugar", "sugar", "sweetness"],
    "chlorides":             ["chlorides", "chloride", "salt"],
    "free sulfur dioxide":   ["free sulfur", "free so2", "free sulfur dioxide"],
    "total sulfur dioxide":  ["total sulfur", "total so2", "total sulfur dioxide"],
    "density":               ["density"],
    "pH":                    ["ph", "acidity level"],
    "sulphates":             ["sulphates", "sulfates", "sulphate"],
    "alcohol":               ["alcohol", "abv", "alcohol content", "alcohol level"],
}


def parse_features_from_text(text: str) -> dict:
    """
    Parse wine feature values from a spoken/typed sentence.

    Example input:
      "alcohol 12.5 volatile acidity 0.4 sulphates 0.7 pH 3.4"

    Returns dict of {feature_name: float} for any features found.
    """
    text_lower = text.lower()
    found = {}

    # Build a pattern: look for feature alias followed by a number
    number_pattern = r"[-+]?\d*\.?\d+"

    for feature, aliases in FEATURE_ALIASES.items():
        for alias in aliases:
            # Escape alias for regex
            alias_escaped = re.escape(alias)
            pattern = rf"{alias_escaped}\s*[:\-=]?\s*({number_pattern})"
            match = re.search(pattern, text_lower)
            if match:
                try:
                    found[feature] = float(match.group(1))
                except ValueError:
                    pass
                break  # Stop checking aliases once found

    return found


def build_voice_prompt_for_wine(prediction: float,
                                quality_label: str,
                                top_features: list[tuple]) -> str:
    """
    Build a natural spoken response for wine prediction results.

    Args:
        prediction:    predicted quality score (float)
        quality_label: e.g. "Good", "Excellent", "Average"
        top_features:  list of (feature_name, shap_value) tuples
    """
    score_text = f"{prediction:.1f} out of 8"

    if top_features:
        top_name, top_val = top_features[0]
        direction = "positively" if top_val > 0 else "negatively"
        feature_sentence = (
            f"The most influential factor is {top_name}, "
            f"which affected the score {direction}."
        )
    else:
        feature_sentence = ""

    return (
        f"Based on the chemical properties you provided, "
        f"this wine is predicted to score {score_text}. "
        f"That puts it in the {quality_label} category. "
        f"{feature_sentence} "
        f"Would you like me to suggest a food pairing or explain how to improve this wine?"
    )


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test TTS
    audio = text_to_speech("This wine scores 7.2 — a very good quality red wine.")
    if audio:
        Path("test_tts.mp3").write_bytes(audio)
        print("✅  TTS test saved to test_tts.mp3")

    # Test feature parsing
    sample = "alcohol 12.5 volatile acidity 0.4 sulphates 0.65 pH 3.35"
    parsed = parse_features_from_text(sample)
    print("\nParsed features:", parsed)