"""
🍷 Wine Quality — Model Training Script
=======================================
Trains three models (RF, GB, XGBoost) before & after fine-tuning,
picks the best one, and saves:
    • best_wine_model.pkl
    • wine_scaler.pkl
    • feature_names.pkl
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings, joblib, sys, os

warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# ─────────────────────────────────────────────────────────────────────────────
# 1 ▸ Load Data
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "winequality-red.csv"
if not os.path.exists(CSV_PATH):
    sys.exit(f"❌  File not found: {CSV_PATH}\n"
             f"    Place winequality-red.csv in the same folder as this script.")

df = pd.read_csv(CSV_PATH)
print(f"✅  Loaded  →  {df.shape[0]} rows × {df.shape[1]} cols")

# ─────────────────────────────────────────────────────────────────────────────
# 2 ▸ Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
# 2.1 Drop duplicates
before = len(df)
df = df.drop_duplicates()
print(f"   Duplicates removed : {before - len(df)}")

# 2.2 Outlier removal (IQR ×3 — conservative to keep wine edge-cases)
features = [c for c in df.columns if c != "quality"]
for col in features:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df[col] >= Q1 - 3 * IQR) & (df[col] <= Q3 + 3 * IQR)]
print(f"   After outlier trim : {len(df)} rows remain")

# 2.3 Feature engineering
df["acidity_ratio"]   = df["fixed acidity"] / (df["volatile acidity"] + 1e-9)
df["so2_ratio"]       = df["free sulfur dioxide"] / (df["total sulfur dioxide"] + 1e-9)
df["alcohol_density"] = df["alcohol"] / df["density"]

X = df.drop("quality", axis=1)
y = df["quality"]

# 2.4 Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# 2.5 Scaling
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print(f"   Train: {X_train.shape}   Test: {X_test.shape}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 3 ▸ Helper
# ─────────────────────────────────────────────────────────────────────────────
results = []

def evaluate(name, estimator, X_tr, y_tr, X_te, y_te):
    """Fit, predict, cross-validate, and return metrics dict."""
    estimator.fit(X_tr, y_tr)
    pred    = estimator.predict(X_te)
    cv      = cross_val_score(estimator, X_tr, y_tr, cv=5, scoring="r2")
    mae     = mean_absolute_error(y_te, pred)
    rmse    = np.sqrt(mean_squared_error(y_te, pred))
    r2      = r2_score(y_te, pred)
    print(f"  {name}")
    print(f"    MAE={mae:.4f}  RMSE={rmse:.4f}  R²={r2:.4f}  CV-R²={cv.mean():.4f}")
    return {
        "name": name, "model": estimator, "pred": pred,
        "MAE": mae, "RMSE": rmse, "R2": r2, "CV_R2": cv.mean()
    }

# ─────────────────────────────────────────────────────────────────────────────
# 4 ▸ Random Forest
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  MODEL 1 — Random Forest")
print("=" * 55)

rf_base   = RandomForestRegressor(n_estimators=100, random_state=42)
r_rf_base = evaluate("Before tuning", rf_base,
                     X_train_sc, y_train, X_test_sc, y_test)

print("  → GridSearchCV …")
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=42),
    {"n_estimators": [100, 200, 300],
     "max_depth":    [None, 10, 20],
     "min_samples_split": [2, 5],
     "max_features": ["sqrt", "log2"]},
    cv=5, scoring="r2", n_jobs=-1, verbose=0)
rf_grid.fit(X_train_sc, y_train)
print(f"  Best RF params: {rf_grid.best_params_}")

r_rf_tuned = evaluate("After tuning", rf_grid.best_estimator_,
                      X_train_sc, y_train, X_test_sc, y_test)

results += [r_rf_base, r_rf_tuned]

# ─────────────────────────────────────────────────────────────────────────────
# 5 ▸ Gradient Boosting
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MODEL 2 — Gradient Boosting")
print("=" * 55)

gb_base   = GradientBoostingRegressor(n_estimators=100, random_state=42)
r_gb_base = evaluate("Before tuning", gb_base,
                     X_train_sc, y_train, X_test_sc, y_test)

print("  → GridSearchCV …")
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    {"n_estimators":  [100, 200, 300],
     "learning_rate": [0.05, 0.1, 0.2],
     "max_depth":     [3, 5, 7],
     "subsample":     [0.8, 1.0]},
    cv=5, scoring="r2", n_jobs=-1, verbose=0)
gb_grid.fit(X_train_sc, y_train)
print(f"  Best GB params: {gb_grid.best_params_}")

r_gb_tuned = evaluate("After tuning", gb_grid.best_estimator_,
                      X_train_sc, y_train, X_test_sc, y_test)

results += [r_gb_base, r_gb_tuned]

# ─────────────────────────────────────────────────────────────────────────────
# 6 ▸ XGBoost
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MODEL 3 — XGBoost")
print("=" * 55)

xgb_base   = xgb.XGBRegressor(n_estimators=100, random_state=42,
                               eval_metric="rmse", verbosity=0)
r_xgb_base = evaluate("Before tuning", xgb_base,
                       X_train_sc, y_train, X_test_sc, y_test)

print("  → GridSearchCV …")
xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=42, eval_metric="rmse", verbosity=0),
    {"n_estimators":     [100, 200, 300],
     "learning_rate":    [0.05, 0.1, 0.2],
     "max_depth":        [3, 5, 7],
     "subsample":        [0.8, 1.0],
     "colsample_bytree": [0.8, 1.0]},
    cv=5, scoring="r2", n_jobs=-1, verbose=0)
xgb_grid.fit(X_train_sc, y_train)
print(f"  Best XGB params: {xgb_grid.best_params_}")

r_xgb_tuned = evaluate("After tuning", xgb_grid.best_estimator_,
                        X_train_sc, y_train, X_test_sc, y_test)

results += [r_xgb_base, r_xgb_tuned]

# ─────────────────────────────────────────────────────────────────────────────
# 7 ▸ Comparison Plot
# ─────────────────────────────────────────────────────────────────────────────
print("\n📊  Generating comparison chart …")
labels = [
    "RF\nBefore", "RF\nAfter",
    "GB\nBefore", "GB\nAfter",
    "XGB\nBefore", "XGB\nAfter"
]
r2s   = [r["R2"]   for r in results]
maes  = [r["MAE"]  for r in results]
rmses = [r["RMSE"] for r in results]
clrs  = ["#6baed6", "#2171b5",
         "#74c476", "#238b45",
         "#fd8d3c", "#d94701"]

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for ax, vals, title, ylabel in zip(
        axes,
        [r2s, maes, rmses],
        ["R² Score (higher better)",
         "MAE (lower better)",
         "RMSE (lower better)"],
        ["R²", "MAE", "RMSE"]):
    bars = ax.bar(labels, vals, color=clrs, edgecolor="black", alpha=0.87)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{val:.3f}", ha="center", fontsize=9)

plt.suptitle("🍷 Wine Quality — All Models Before & After Tuning",
             fontsize=14)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150, bbox_inches="tight")
print("   Saved → model_comparison.png")
plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# 8 ▸ Select & Save Best Model
# ─────────────────────────────────────────────────────────────────────────────
tuned = [r for r in results if "After" in r["name"]]
best  = max(tuned, key=lambda r: r["R2"])

print("\n" + "=" * 55)
print(f"  🏆  BEST MODEL  →  R²={best['R2']:.4f}")
print(f"       {best['name']}")
print("=" * 55)

# Identify which model won (used in app.py label)
if   best["model"] is r_rf_tuned["model"]:  model_label = "Random Forest"
elif best["model"] is r_gb_tuned["model"]:  model_label = "Gradient Boosting"
else:                                         model_label = "XGBoost"

joblib.dump(best["model"],        "best_wine_model.pkl")
joblib.dump(scaler,               "wine_scaler.pkl")
joblib.dump(list(X.columns),      "feature_names.pkl")
joblib.dump({"label": model_label,
             "R2":    best["R2"],
             "MAE":   best["MAE"],
             "RMSE":  best["RMSE"]}, "model_meta.pkl")

print("\n✅  Saved:")
print("     best_wine_model.pkl  — best estimator")
print("     wine_scaler.pkl      — fitted StandardScaler")
print("     feature_names.pkl    — ordered column list")
print("     model_meta.pkl       — metrics for Streamlit UI")
print("\nRun the app:  streamlit run app.py")
