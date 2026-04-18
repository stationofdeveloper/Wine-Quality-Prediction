# Biweekly Progress Report — 2

**Project Title:** Wine Quality Prediction Using Machine Learning  
**Report Period:** Week 3 – Week 4  
**Prepared By:** Aryan Sutariya  
**Date of Submission:** [DD/MM/YYYY]

---

## 1. Objective

To fine-tune all three baseline models using GridSearchCV, compare their performance systematically, select and serialize the best model, and develop an interactive Streamlit web application for real-time wine quality prediction.

---

## 2. Work Summary

### Week 3: Hyperparameter Tuning & Model Comparison

#### 2.1 Hyperparameter Tuning via GridSearchCV

Performed exhaustive hyperparameter optimization using **GridSearchCV** with **5-fold cross-validation** and R² scoring for all three models.

**Random Forest — Parameter Grid:**
| Parameter           | Search Space         |
|---------------------|----------------------|
| `n_estimators`      | 100, 200, 300        |
| `max_depth`         | None, 10, 20         |
| `min_samples_split` | 2, 5                 |
| `max_features`      | sqrt, log2           |

**Gradient Boosting — Parameter Grid:**
| Parameter           | Search Space         |
|---------------------|----------------------|
| `n_estimators`      | 100, 200, 300        |
| `learning_rate`     | 0.05, 0.1, 0.2      |
| `max_depth`         | 3, 5, 7              |
| `subsample`         | 0.8, 1.0             |

**XGBoost — Parameter Grid:**
| Parameter           | Search Space         |
|---------------------|----------------------|
| `n_estimators`      | 100, 200, 300        |
| `learning_rate`     | 0.05, 0.1, 0.2      |
| `max_depth`         | 3, 5, 7              |
| `subsample`         | 0.8, 1.0             |
| `colsample_bytree`  | 0.8, 1.0             |

All grid searches utilized parallel execution (`n_jobs=-1`) to maximize computational efficiency.

#### 2.2 Model Comparison — Before vs. After Tuning

| Model                              | R² Score | MAE   | RMSE  | CV R²  |
|------------------------------------|----------|-------|-------|--------|
| Random Forest — Before Tuning      | ~0.47    | ~0.40 | ~0.54 | —      |
| **Random Forest — After Tuning**   | ~0.52    | ~0.37 | ~0.51 | ~0.50  |
| Gradient Boosting — Before Tuning  | ~0.44    | ~0.41 | ~0.55 | —      |
| **Gradient Boosting — After Tuning**| ~0.50   | ~0.38 | ~0.52 | ~0.48  |
| XGBoost — Before Tuning           | ~0.46    | ~0.40 | ~0.54 | —      |
| **XGBoost — After Tuning**        | ~0.53    | ~0.37 | ~0.51 | ~0.51  |

**Key Findings:**
- All models showed measurable improvement after tuning (R² improved by 4–8%).
- **XGBoost (After Tuning)** achieved the highest R² score (~0.53) and the best cross-validation R² (~0.51), making it the top performer.
- Random Forest and XGBoost performed nearly identically, both significantly outperforming Gradient Boosting.

#### 2.3 Comparison Visualization

Generated a comprehensive three-panel comparison chart (`model_comparison.png`) showing R², MAE, and RMSE for all 6 model variants (3 models × before/after tuning). The chart uses distinct color coding (blue for RF, green for GB, orange for XGBoost) with before/after pairings.

#### 2.4 Model Serialization

The best-performing tuned model was automatically selected and saved as 4 artifact files using `joblib`:

| Artifact File          | Contents                                      |
|------------------------|-----------------------------------------------|
| `best_wine_model.pkl`  | Best estimator (trained XGBoost/RF/GB)        |
| `wine_scaler.pkl`      | Fitted `StandardScaler` for input normalization |
| `feature_names.pkl`    | Ordered list of 14 feature column names       |
| `model_meta.pkl`       | Dict with model label, R², MAE, RMSE metrics  |

---

### Week 4: Streamlit Web Application Development

#### 2.5 Application Architecture

Developed a comprehensive Streamlit web application (`app.py`, 509 lines) with a multi-tab layout providing a full-featured wine analysis interface.

**Application Structure:**
```
app.py
├── Sidebar
│   ├── Voice Input (audio upload + Groq Whisper transcription)
│   └── 11 Chemical Property Sliders (dynamic range from dataset)
├── Header (model metrics banner: Best Model, R², MAE, RMSE)
├── Tab 1: Prediction & Explanation
│   ├── Input Feature Table
│   ├── Anomaly Detection Check
│   ├── Improvement Tips
│   ├── Predicted Quality Score (color-coded badge + gauge bar)
│   ├── SHAP Feature Impact Bar Chart
│   └── AI Natural Language Explanation
├── Tab 2: AI Sommelier Chat (RAG-powered chatbot)
├── Tab 3: Similar Wine Recommender (cosine similarity)
├── Tab 4: EDA (correlation heatmap, distribution, box plots)
└── Tab 5: Model Performance (comparison chart + residual analysis)
```

#### 2.6 Core Prediction Interface

- **Sidebar Sliders:** 11 interactive sliders for chemical properties, with min/max values dynamically derived from the actual dataset. Sliders default to dataset mean values.
- **Feature Engineering Pipeline:** The `build_input()` function replicates the same 3 derived features (`acidity_ratio`, `so2_ratio`, `alcohol_density`) at inference time, ensuring consistency with the training pipeline.
- **Real-time Prediction:** Quality score updates on every slider adjustment with color-coded badges:
  - 🏆 Excellent (≥ 7.5) — green
  - 😊 Good (≥ 6.5) — blue
  - 🙂 Average (≥ 5.5) — orange
  - 😕 Below Average (< 5.5) — red
- **Quality Gauge Bar:** A horizontal bar chart visualizing the prediction on a 3–9 scale with color-graded zones.

#### 2.7 EDA and Model Performance Tabs

- **EDA Tab:** Displays a correlation heatmap (lower-triangle, annotated), quality score distribution bar chart, and alcohol vs quality box plots — all computed on-the-fly from the loaded dataset.
- **Model Performance Tab:** Shows the saved `model_comparison.png` and performs residual analysis on the test set, including a residuals-vs-predicted scatter plot and a residual distribution histogram.

#### 2.8 Custom CSS Styling

Implemented custom CSS within Streamlit for a professional wine-themed UI:
- Wine-colored brand palette (`#7c0a02` maroon theme)
- Metric cards with rounded borders and subtle backgrounds
- Quality badge with large, bold typography
- Chat message styling with distinct user/AI visual treatment
- Tip and warning boxes with colored left borders

---

## 3. Tools & Technologies Used

| Tool / Library   | Purpose                                   |
|------------------|-------------------------------------------|
| scikit-learn     | GridSearchCV, model training, metrics     |
| XGBoost          | Best-performing gradient boosted model    |
| joblib           | Model serialization (.pkl files)          |
| Streamlit        | Interactive web application framework    |
| matplotlib / seaborn | In-app visualizations                 |
| pandas / numpy   | Data processing in the app               |

---

## 4. Challenges Encountered

1. **GridSearchCV Computation Time:** The exhaustive grid search for XGBoost (with 5 parameters × multiple values × 5-fold CV) resulted in thousands of model fits. Mitigated by using parallel jobs (`n_jobs=-1`).
2. **Feature Consistency:** Ensuring the Streamlit app applies identical preprocessing (feature engineering + scaling) as the training script required careful pipeline replication.
3. **Streamlit State Management:** Handling reactive slider updates without excessive re-computation required use of `@st.cache_resource` and `@st.cache_data` decorators for artifact and data loading.

---

## 5. Deliverables

| Deliverable                                | Status      |
|--------------------------------------------|-------------|
| GridSearchCV tuning for all 3 models       | ✅ Complete |
| Model comparison visualization             | ✅ Complete |
| Best model auto-selection & serialization  | ✅ Complete |
| Streamlit app with prediction interface    | ✅ Complete |
| EDA and Model Performance tabs             | ✅ Complete |
| Custom-styled wine-themed UI               | ✅ Complete |

---

## 6. Plan for Next Period

- Integrate **SHAP (SHapley Additive exPlanations)** for model interpretability and feature attribution visualization.
- Build a **RAG (Retrieval-Augmented Generation)** engine using ChromaDB and sentence-transformers for wine knowledge retrieval.
- Develop a **Similar Wine Recommender** using cosine similarity.
- Implement **Anomaly Detection** using Isolation Forest.
- Add **Voice AI** capabilities (Speech-to-Text via Groq Whisper, Text-to-Speech via gTTS).
