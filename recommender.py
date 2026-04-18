"""
🍾 Wine Recommender & Anomaly Detector
=======================================
1. Similar wine finder — finds top-N wines from dataset that are
   chemically similar to the user's input AND scored 7+.
2. Anomaly detector — flags unrealistic input combinations using
   an Isolation Forest trained on the real wine dataset.

Both are fully local — no API keys required.
"""

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Optional


# ─── Anomaly Detection ────────────────────────────────────────────────────────

_anomaly_detector = None
_anomaly_scaler   = None


def _train_anomaly_detector(df: pd.DataFrame, feature_names: list):
    """Train an Isolation Forest on clean dataset features."""
    global _anomaly_detector, _anomaly_scaler
    from sklearn.ensemble  import IsolationForest
    from sklearn.preprocessing import StandardScaler

    X = df[feature_names].values
    _anomaly_scaler   = StandardScaler().fit(X)
    X_sc              = _anomaly_scaler.transform(X)
    _anomaly_detector = IsolationForest(
        n_estimators=200,
        contamination=0.05,   # expect ~5% outliers in wine data
        random_state=42,
    ).fit(X_sc)


def detect_anomaly(input_values: dict,
                   df: pd.DataFrame,
                   feature_names: list) -> tuple[bool, str]:
    """
    Check if the user's input is statistically unusual compared to real wine data.

    Returns:
        (is_anomaly: bool, message: str)
    """
    global _anomaly_detector, _anomaly_scaler

    if _anomaly_detector is None:
        _train_anomaly_detector(df, feature_names)

    try:
        # Build input vector in correct order
        row     = np.array([[input_values[f] for f in feature_names]])
        row_sc  = _anomaly_scaler.transform(row)
        pred    = _anomaly_detector.predict(row_sc)   # -1 = anomaly, 1 = normal
        score   = _anomaly_detector.score_samples(row_sc)[0]  # lower = more anomalous

        is_anomaly = (pred[0] == -1)

        # Find which feature(s) deviate most from the dataset median
        medians = df[feature_names].median()
        stds    = df[feature_names].std()
        deviations = {}
        for feat in feature_names:
            z = abs((input_values[feat] - medians[feat]) / (stds[feat] + 1e-9))
            if z > 3.5:
                deviations[feat] = round(z, 1)

        if is_anomaly:
            if deviations:
                dev_str = ", ".join(
                    f"{k} (z={v}σ)" for k, v in
                    sorted(deviations.items(), key=lambda x: -x[1])[:3]
                )
                msg = (f"⚠️ Unusual input detected. The following values are far outside "
                       f"typical wine ranges: {dev_str}. "
                       f"The prediction may not be reliable.")
            else:
                msg = ("⚠️ The combination of values is unusual. "
                       "The prediction may not be reliable.")
        else:
            msg = "✅ Input values look realistic for a red wine."

        return is_anomaly, msg

    except Exception as e:
        return False, f"Anomaly check skipped: {e}"


# ─── Similar Wine Recommender ─────────────────────────────────────────────────

def find_similar_wines(input_values: dict,
                       df: pd.DataFrame,
                       feature_names: list,
                       scaler,
                       min_quality: float = 6.5,
                       top_n: int = 3) -> pd.DataFrame:
    """
    Find the N most chemically similar wines from the dataset
    that scored >= min_quality.

    Uses cosine similarity in the scaled feature space.

    Returns a DataFrame with the top_n matches + their quality scores.
    """
    try:
        # Filter to quality wines only
        good_wines = df[df["quality"] >= min_quality].copy()
        if good_wines.empty:
            good_wines = df.nlargest(20, "quality").copy()

        raw_features = [f for f in feature_names
                        if f not in ("acidity_ratio", "so2_ratio", "alcohol_density")]

        # Scale both the input and the dataset
        input_row = np.array([[input_values.get(f, 0) for f in raw_features]])
        dataset_X = good_wines[raw_features].values

        # Use a simple StandardScaler fitted on the good_wines subset
        from sklearn.preprocessing import StandardScaler
        local_scaler = StandardScaler().fit(dataset_X)
        input_sc     = local_scaler.transform(input_row)
        dataset_sc   = local_scaler.transform(dataset_X)

        # Euclidean distance (lower = more similar)
        distances = np.linalg.norm(dataset_sc - input_sc, axis=1)
        good_wines = good_wines.copy()
        good_wines["_distance"] = distances
        good_wines["_similarity_%"] = (
            100 * (1 - distances / (distances.max() + 1e-9))
        ).round(1)

        top = good_wines.nsmallest(top_n, "_distance")[
            raw_features + ["quality", "_similarity_%"]
        ].reset_index(drop=True)

        top.rename(columns={"_similarity_%": "similarity %"}, inplace=True)
        return top

    except Exception as e:
        print(f"Recommender error: {e}")
        return pd.DataFrame()


def improvement_tips(input_values: dict, df: pd.DataFrame) -> list[str]:
    """
    Generate specific, data-driven tips to improve wine quality.
    Compares user's values to the median values of high-scoring wines (7+).
    """
    tips   = []
    ideal  = df[df["quality"] >= 7].median()
    raw_fs = [c for c in df.columns if c != "quality"]

    checks = {
        "volatile acidity": (
            0, 0.5,
            "volatile acidity is HIGH ({val:.2f}). "
            "Aim below 0.5 — high acetic acid causes vinegar taste."
        ),
        "alcohol": (
            11.5, 99,
            "alcohol is LOW ({val:.1f}%). "
            "Top wines average {ideal:.1f}%. Richer fermentation raises alcohol and body."
        ),
        "sulphates": (
            0.55, 1.5,
            "sulphates are LOW ({val:.2f}). "
            "Top wines average {ideal:.2f}. Moderate sulphates enhance fruit preservation."
        ),
        "citric acid": (
            0.25, 99,
            "citric acid is LOW ({val:.2f}). "
            "A touch more ({ideal:.2f}) adds freshness and balances other acids."
        ),
        "pH": (
            0, 3.7,
            "pH is HIGH ({val:.2f}) — low acidity. "
            "Top wines average pH {ideal:.2f}. Try adding tartaric acid to sharpen acidity."
        ),
        "fixed acidity": (
            7.0, 14,
            "fixed acidity ({val:.1f}) is outside the ideal range. "
            "Top wines average {ideal:.1f} g/L."
        ),
    }

    for feat, (lo, hi, template) in checks.items():
        val = input_values.get(feat)
        if val is None:
            continue
        if val < lo or val > hi:
            tip = template.format(val=val, ideal=ideal.get(feat, 0))
            tips.append(f"• {tip}")

    if not tips:
        tips.append("• 🌟 Your wine's chemical profile is already close to high-quality wines!")

    return tips[:4]   # Return at most 4 tips


# ─── Quick test ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = pd.read_csv("winequality-red.csv")
    feature_names = [c for c in df.columns if c != "quality"]

    sample = {f: float(df[f].mean()) for f in feature_names}
    sample["volatile acidity"] = 0.9   # deliberately high

    is_anom, msg = detect_anomaly(sample, df, feature_names)
    print("Anomaly:", is_anom, "|", msg)

    tips = improvement_tips(sample, df)
    print("\nImprovement tips:")
    for t in tips:
        print(t)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df[feature_names])
    similar = find_similar_wines(sample, df, feature_names, scaler, min_quality=7)
    print("\nSimilar high-quality wines:\n", similar)