# 🍷 Wine Quality Prediction — Comprehensive Project Resource

## Project Title
**Wine Quality Prediction & AI Sommelier System Using Machine Learning and Agentic RAG**

## Author
Aryan Sutariya

---

## 1. PROJECT OVERVIEW

This project is an end-to-end Machine Learning system that predicts the quality of red wine based on its physicochemical (chemical) properties. It goes far beyond a basic prediction model — it integrates SHAP-based model explainability, a Retrieval-Augmented Generation (RAG) chatbot powered by three academic wine textbooks, voice AI (speech-to-text and text-to-speech), a similar wine recommender, and anomaly detection — all wrapped in an interactive Streamlit web application.

### Problem Statement
Wine quality assessment traditionally relies on subjective sensory evaluation by trained sommeliers. This process is expensive, inconsistent, and not scalable. Our system automates wine quality scoring using measurable chemical properties, making quality prediction objective, instant, and accessible. Furthermore, the AI Sommelier chatbot provides expert-level wine knowledge by reasoning over authoritative textbooks.

### Key Innovation
The project implements two distinct RAG approaches:
1. **Vector RAG (ChromaDB + sentence-transformers):** Traditional embedding-based retrieval from curated text files.
2. **Agentic Vectorless RAG (PageIndex-inspired):** A novel approach where the LLM reasons over hierarchical book indexes using a ReAct (Reason + Act) agent loop — no vector database, no embeddings, no chunking. The LLM decides which pages to read based on structural understanding.

---

## 2. DATASET

- **Source:** UCI Machine Learning Repository — Wine Quality Dataset
- **File:** `winequality-red.csv`
- **Total Samples:** 1,599 red wine samples from Portuguese "Vinho Verde" wine
- **Features:** 11 physicochemical input features + 1 target variable
- **Target Variable:** `quality` — sensory quality score on a scale of 3 to 8 (integer)

### Feature Descriptions

| # | Feature | Description | Unit |
|---|---------|-------------|------|
| 1 | Fixed Acidity | Tartaric acid — contributes to wine structure and backbone | g/dm³ |
| 2 | Volatile Acidity | Acetic acid — high levels produce an unpleasant vinegar taste | g/dm³ |
| 3 | Citric Acid | Adds freshness and pleasant flavor; found in small quantities | g/dm³ |
| 4 | Residual Sugar | Sugar remaining after fermentation; affects sweetness perception | g/dm³ |
| 5 | Chlorides | Salt content; excessive levels create a salty taste | g/dm³ |
| 6 | Free Sulfur Dioxide | Active form of SO₂; prevents microbial growth and oxidation | mg/dm³ |
| 7 | Total Sulfur Dioxide | Sum of free and bound SO₂; excessive amounts detectable by smell/taste | mg/dm³ |
| 8 | Density | Depends on alcohol and sugar content; indicator of wine body | g/cm³ |
| 9 | pH | Acidity level (lower = more acidic); most wines fall between 3.0–4.0 | — |
| 10 | Sulphates | Potassium sulphate additive; antimicrobial and antioxidant properties | g/dm³ |
| 11 | Alcohol | Alcohol percentage by volume; strongly correlates with quality | % vol |
| Target | Quality | Sensory score from expert tasters | Scale 3–8 |

### Key Dataset Observations
- The quality score distribution is imbalanced — most wines score 5 or 6, with very few scoring 3, 4, or 8.
- Alcohol has the strongest positive correlation with quality (~0.48).
- Volatile acidity has the strongest negative correlation with quality (~-0.39).
- Citric acid and sulphates show moderate positive correlations with quality.

---

## 3. METHODOLOGY & ML PIPELINE

### 3.1 Data Preprocessing

**Step 1 — Duplicate Removal:** Identified and dropped duplicate rows to prevent training bias.

**Step 2 — Outlier Removal (IQR × 3):** Applied the Interquartile Range method with a conservative multiplier of 3 to filter extreme outliers while preserving legitimate edge-case wines.

**Step 3 — Feature Engineering:** Created 3 derived features to capture chemical relationships:
- `acidity_ratio = fixed acidity / volatile acidity` — Balance between structural acids and off-flavor acids.
- `so2_ratio = free sulfur dioxide / total sulfur dioxide` — Proportion of active (free) preservative SO₂.
- `alcohol_density = alcohol / density` — Alcohol concentration relative to wine body.

**Step 4 — Train/Test Split:** 80/20 split with `random_state=42` for reproducibility.

**Step 5 — Feature Scaling:** StandardScaler fitted on training data, applied to both train and test sets.

### 3.2 Models Trained

Three ensemble regression models were trained, each evaluated before and after hyperparameter tuning:

#### Model 1: Random Forest Regressor
- Ensemble of decision trees using bagging (bootstrap aggregation).
- GridSearchCV parameters: n_estimators (100, 200, 300), max_depth (None, 10, 20), min_samples_split (2, 5), max_features (sqrt, log2).

#### Model 2: Gradient Boosting Regressor
- Sequential ensemble that corrects errors of previous trees.
- GridSearchCV parameters: n_estimators (100, 200, 300), learning_rate (0.05, 0.1, 0.2), max_depth (3, 5, 7), subsample (0.8, 1.0).

#### Model 3: XGBoost Regressor
- Optimized gradient boosting with regularization.
- GridSearchCV parameters: n_estimators (100, 200, 300), learning_rate (0.05, 0.1, 0.2), max_depth (3, 5, 7), subsample (0.8, 1.0), colsample_bytree (0.8, 1.0).

All tuning used 5-fold cross-validation with R² scoring.

### 3.3 Model Results

| Model | R² Score | MAE | RMSE | CV R² |
|-------|----------|-----|------|-------|
| Random Forest — Before Tuning | ~0.47 | ~0.40 | ~0.54 | — |
| Random Forest — After Tuning | ~0.52 | ~0.37 | ~0.51 | ~0.50 |
| Gradient Boosting — Before Tuning | ~0.44 | ~0.41 | ~0.55 | — |
| Gradient Boosting — After Tuning | ~0.50 | ~0.38 | ~0.52 | ~0.48 |
| XGBoost — Before Tuning | ~0.46 | ~0.40 | ~0.54 | — |
| **XGBoost — After Tuning** | **~0.53** | **~0.37** | **~0.51** | **~0.51** |

**Best Model:** XGBoost (After Tuning) — highest R² and cross-validation score.

### 3.4 Evaluation Metrics Explained
- **R² Score (Coefficient of Determination):** Proportion of variance in quality scores explained by the model. 1.0 = perfect, 0.0 = no better than mean.
- **MAE (Mean Absolute Error):** Average magnitude of prediction errors in quality points.
- **RMSE (Root Mean Squared Error):** Penalizes larger errors more heavily than MAE.
- **CV R² (Cross-Validation R²):** Average R² across 5 folds — measures generalization ability.

---

## 4. SYSTEM ARCHITECTURE

### 4.1 Complete System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   🍷 WINE AI SOMMELIER                      │
├───────────────────────┬─────────────────────────────────────┤
│   Unified Streamlit   │   Standalone Chatbot                │
│   Application (app.py)│   (wine_chatbot.py)                 │
│   4 Pages:            │                                     │
│   • Predictor         │   Agentic RAG Chat Interface        │
│   • Chatbot           │   with voice I/O and citations      │
│   • EDA Dashboard     │                                     │
│   • Model Performance │                                     │
├───────────────────────┼─────────────────────────────────────┤
│    model_training.py  │   wine_rag_agent.py                 │
│    ML Training         │   ReAct Agent Loop (4 tools)        │
│    (RF, GB, XGBoost)  │   Agentic Vectorless RAG            │
├───────────────────────┼─────────────────────────────────────┤
│    explainer.py       │   wine_rag_indexer.py               │
│    SHAP + LLM         │   PDF → Tree Index Builder          │
│    Explanations       │   (TOC + LLM-based)                 │
├───────────────────────┼─────────────────────────────────────┤
│    recommender.py     │   rag_engine.py                     │
│    Cosine Similarity  │   ChromaDB Vector RAG               │
│    + Isolation Forest │   (sentence-transformers)           │
├───────────────────────┼─────────────────────────────────────┤
│    voice_handler.py   │   setup_rag.py                      │
│    Whisper STT        │   Environment Validation            │
│    + gTTS TTS         │   + Automated Setup                 │
├───────────────────────┴─────────────────────────────────────┤
│                      Data Layer                             │
│   winequality-red.csv │ .pkl models │ wine_knowledge/ PDFs  │
└─────────────────────────────────────────────────────────────┘
```

### 4.2 Source Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `app.py` | ~580 | Unified Streamlit app — 4 pages: Predictor, Chatbot, EDA, Models |
| `wine_chatbot.py` | 463 | Standalone chatbot with agentic RAG and voice |
| `model_training.py` | 241 | ML pipeline: load → preprocess → train → tune → save |
| `wine_rag_agent.py` | 468 | ReAct agent with 4 tools for book-based retrieval |
| `wine_rag_indexer.py` | 401 | PDF book indexer — builds hierarchical tree indexes |
| `rag_engine.py` | 151 | ChromaDB vector RAG engine |
| `explainer.py` | 241 | SHAP feature attribution + LLM natural language explanations |
| `recommender.py` | 224 | Similar wine finder + anomaly detection + improvement tips |
| `voice_handler.py` | 169 | Groq Whisper STT + gTTS TTS + spoken feature parsing |
| `setup_rag.py` | 260 | Automated setup, API testing, book indexing, test queries |

**Total codebase:** ~3,500+ lines of Python across 10 source files.

---

## 5. FEATURE DEEP DIVES

### 5.1 SHAP Explainability (explainer.py)

**What is SHAP?**
SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain the output of any machine learning model. For each prediction, SHAP calculates how much each feature contributed to the prediction (positive or negative impact).

**Implementation:**
- Uses `shap.TreeExplainer` — optimized for tree-based models (RF, GB, XGBoost).
- Computes per-feature SHAP values for each individual prediction.
- Renders a horizontal bar chart: green bars (positive impact) and red bars (negative impact).
- Results are fed to an LLM to generate natural language explanations.

**LLM Explanation Pipeline:**
1. SHAP values → formatted summary of top 5 features.
2. Chemical property highlights (flagging high/low values).
3. Prompt construction with SHAP summary + chemical context.
4. LLM generates 3–5 sentence explanation with one improvement tip.
5. Fallback chain: Groq LLaMA → Gemini Flash → Rule-based engine.

### 5.2 Agentic Vectorless RAG (wine_rag_agent.py + wine_rag_indexer.py)

**Inspiration:** PageIndex (VectifyAI) — a novel RAG paradigm that eschews vector databases entirely.

**How Traditional Vector RAG Works:**
1. Split documents into chunks.
2. Generate embeddings (numerical vectors) for each chunk.
3. Store embeddings in a vector database.
4. At query time, embed the query and find nearest neighbor chunks.
5. Pass relevant chunks to LLM for answer generation.

**How Agentic Vectorless RAG Works (Our Approach):**
1. Build a hierarchical tree index of each book (once).
2. At query time, the LLM sees the tree structure (section titles, page ranges, summaries).
3. The LLM REASONS about which sections are relevant.
4. The LLM requests specific pages (like a human looking up a textbook).
5. The LLM reads the actual page text and synthesizes an answer.
6. Citations include exact book title and page numbers.

**Agent Tools:**
| Tool | Purpose | When Used |
|------|---------|-----------|
| `list_books()` | Shows available books with status | First step — understanding available resources |
| `get_structure(book_id)` | Returns the table of contents tree | Understanding what a book covers |
| `get_pages(book_id, pages)` | Fetches actual text from pages | After identifying relevant section |
| `search_section(book_id, query)` | LLM-powered section finder | Quick lookup for specific topics |

**ReAct Loop:**
The agent operates in a Reason-Act cycle for up to 6 iterations:
```
Iteration 1: THINK → "I need to check the chemistry book"
             ACT   → get_structure("chemistry")
Iteration 2: THINK → "Section on volatile acidity is on pages 45-52"
             ACT   → get_pages("chemistry", "45-52")
Iteration 3: THINK → "I have enough information to answer"
             ACT   → Generate final answer with citations
```

**Books Indexed:**
1. "Chemistry and Biochemistry of Winemaking" (258 pages) — Chemical compounds, fermentation, stabilization.
2. "Wine Tasting: A Professional Handbook" (519 pages) — Sensory evaluation, tasting methods, scoring.
3. "Perceptions of Wine Quality" (406 pages) — Quality perception, consumer behavior, standards.

### 5.3 Similar Wine Recommender (recommender.py)

**Algorithm:**
1. Filter dataset for wines scoring ≥ 7 (user-configurable threshold).
2. Scale both user input and dataset wines using StandardScaler.
3. Compute Euclidean distance in scaled feature space.
4. Return top-N most similar wines ranked by similarity percentage.
5. Display side-by-side comparison chart (user wine vs best match).

### 5.4 Anomaly Detection (recommender.py)

**Algorithm:** Isolation Forest
- Trained on the clean wine dataset with 200 trees and 5% expected contamination.
- Flags user inputs that fall outside the normal wine distribution.
- Identifies specific features that deviate most (z-score > 3.5σ from median).
- Provides clear warning messages with the unusual feature values.

### 5.5 Data-Driven Improvement Tips (recommender.py)

Compares user input values against the median values of wines scoring 7+:
- Checks volatile acidity (should be < 0.5), alcohol (should be > 11.5%), sulphates (should be > 0.55), citric acid (should be > 0.25), pH (should be < 3.7), fixed acidity (should be 7–14).
- Generates specific, actionable tips with precise numbers.

### 5.6 Voice AI (voice_handler.py)

**Speech-to-Text (STT):**
- Groq Whisper API (whisper-large-v3 model) — free tier.
- Accepts WAV, MP3, M4A audio uploads.
- Transcribes spoken wine feature values.

**Feature Parsing:**
- Regex-based extraction from natural language.
- Supports multiple aliases per feature (e.g., "volatile acidity", "acetic acid", "volatile acid").
- Auto-populates sidebar sliders with parsed values.

**Text-to-Speech (TTS):**
- gTTS (Google Text-to-Speech) — completely free, no API key.
- Converts AI explanations and chatbot responses to spoken audio.
- Embedded audio player in the Streamlit interface.

---

## 6. STREAMLIT WEB APPLICATION

### 6.1 Unified App (app.py) — 4 Pages

**Page 1 — Quality Predictor:**
- 11 interactive sliders for chemical properties (3-column layout).
- Real-time quality prediction with color-coded badge (Excellent/Good/Average/Below Average).
- Horizontal quality gauge bar (3–9 scale with color zones).
- Anomaly detection check with styled warning/success boxes.
- SHAP feature impact chart (top 6 features).
- LLM-powered natural language explanation.
- Data-driven improvement tips.
- Similar high-quality wines from the dataset.
- One-click button to ask the AI Sommelier about the predicted wine.

**Page 2 — AI Sommelier Chat:**
- Full agentic RAG chatbot interface.
- Styled chat bubbles (user: maroon; AI: white with border).
- Automatic source citation cards.
- 8 suggested starter questions.
- Live agent reasoning steps display.
- Voice input (upload audio) and voice output (toggle).
- Book indexing status and management.

**Page 3 — EDA Dashboard:**
- Dataset statistics cards (samples, features, quality range, mean).
- Correlation heatmap (lower triangle, annotated).
- Quality score distribution bar chart.
- Feature histogram (selectable feature).
- Feature vs Quality box plots (selectable feature).
- Absolute Pearson correlation with quality bar chart.

**Page 4 — Model Performance:**
- All-models comparison chart (R², MAE, RMSE — before/after tuning).
- Residual analysis: Residuals vs Predicted scatter plot + Residual distribution histogram.
- Dataset preview table with CSV download.

### 6.2 Design & Styling
- Wine-themed maroon color palette (#7c0a02).
- Gradient title text with linear-gradient CSS.
- Custom metric cards, chat bubbles, tip/warning boxes.
- Sidebar navigation with primary/secondary button states.
- Clean, professional, modern UI design.

---

## 7. TECHNOLOGY STACK

| Category | Technology | Purpose |
|----------|------------|---------|
| **Language** | Python 3.10+ | Core programming language |
| **ML Framework** | scikit-learn | RF, GB, GridSearchCV, StandardScaler, Isolation Forest, metrics |
| **ML Framework** | XGBoost | Gradient boosted tree regressor |
| **Explainability** | SHAP | Feature attribution for model interpretability |
| **Web Framework** | Streamlit | Interactive web application |
| **Vector DB** | ChromaDB | Local persistent vector store for text RAG |
| **Embeddings** | sentence-transformers (MiniLM-L6-v2) | Text embedding for vector RAG |
| **PDF Processing** | PyMuPDF (fitz) | PDF text extraction and TOC parsing |
| **LLM (Primary)** | Groq API — LLaMA 3.1 70B Versatile | Chat, explanations, agent reasoning |
| **LLM (Fast)** | Groq API — LLaMA 3.1 8B Instant | Book indexing, section finding |
| **LLM (Fallback)** | Google Gemini 1.5 Flash | Fallback when Groq is unavailable |
| **LLM (Fallback)** | OpenRouter — Gemma 4 26B | Additional fallback |
| **STT** | Groq Whisper Large v3 | Speech-to-text transcription |
| **TTS** | gTTS (Google TTS) | Text-to-speech synthesis (free) |
| **Data** | pandas, numpy | Data manipulation and computation |
| **Visualization** | matplotlib, seaborn | Charts, plots, heatmaps |
| **EDA** | dtale | Interactive exploratory data analysis |
| **Serialization** | joblib | Model and scaler persistence (.pkl) |
| **Config** | python-dotenv | Secure API key management |
| **Notebooks** | Jupyter | Interactive development and documentation |

---

## 8. KEY FINDINGS & INSIGHTS

### 8.1 Feature Importance (from SHAP Analysis)
1. **Alcohol** — Strongest positive predictor. Higher alcohol wines consistently score higher.
2. **Volatile Acidity** — Strongest negative predictor. High values produce vinegar taste.
3. **Sulphates** — Moderate positive impact. Act as antimicrobial preservative.
4. **Citric Acid** — Positive contributor. Adds freshness and flavor complexity.
5. **Total Sulfur Dioxide** — Negative at high levels. Excessive amounts are detectable.

### 8.2 Model Performance Insights
- All three models showed consistent improvement after hyperparameter tuning (4–8% R² increase).
- XGBoost slightly outperformed Random Forest and Gradient Boosting.
- The R² score of ~0.53 suggests the model captures about 53% of quality variance — reasonable given that wine quality is inherently subjective.
- Residual analysis shows approximately normally distributed errors, indicating good model behavior.

### 8.3 Agentic RAG vs Vector RAG
- Agentic RAG produces more precise citations (exact page numbers vs chunk IDs).
- Agentic RAG eliminates false-positive retrievals common with embedding similarity.
- Agentic RAG works better on dense, technical content with complex hierarchical structure.
- Vector RAG is faster for simple queries but less precise for nuanced questions.

---

## 9. CHALLENGES & SOLUTIONS

| Challenge | Solution |
|-----------|----------|
| Quality score class imbalance (most wines score 5-6) | Used regression instead of classification; engineered ratio features to capture subtle differences |
| Choosing outlier removal threshold | Used IQR × 3 (conservative) to preserve edge-case wines while filtering noise |
| Feature multicollinearity (density ↔ fixed acidity) | Engineered ratio features (acidity_ratio, so2_ratio, alcohol_density) |
| GridSearchCV computation time | Used parallel jobs (n_jobs=-1) across all CPU cores |
| LLM API rate limits (Groq free tier) | Implemented retry with exponential backoff + multi-provider fallback chain |
| Large book indexing (500+ pages) | Batch processing with 12-page windows + 1.5s rate-limit delays |
| Context window overflow for large books | Truncation limits (8000 chars for structure, 7000 for pages) with suggestions to use search_section |
| PDF pages with images/non-text content | Minimum text length check (50 chars) to skip unreadable pages |
| Voice input parsing variability | Regex-based matching with multiple aliases per feature |

---

## 10. FUTURE SCOPE

1. **White Wine Support** — Extend the model to predict quality of white wines using the UCI white wine dataset.
2. **Deep Learning Models** — Experiment with neural networks (MLP, TabNet) for potentially higher accuracy.
3. **Real-time Sensor Integration** — Connect with IoT wine sensors for automated quality monitoring.
4. **Multi-language Voice Support** — Extend voice input/output beyond English.
5. **Wine Image Analysis** — Add computer vision to analyze wine color from photos.
6. **Production Deployment** — Deploy on cloud platforms (AWS/GCP) with Docker containerization.
7. **User Feedback Loop** — Collect user feedback on predictions to continuously improve the model.
8. **Extended Knowledge Base** — Add more wine textbooks and research papers to the RAG system.

---

## 11. CONCLUSION

The Wine Quality Prediction & AI Sommelier project demonstrates the practical integration of traditional machine learning with modern AI techniques. Starting from a basic regression pipeline, the system evolved into a comprehensive wine analysis platform featuring:

- **Predictive ML** with three fine-tuned ensemble models achieving R² ≈ 0.53.
- **Explainable AI** through SHAP feature attribution combined with LLM-narrated explanations.
- **Knowledge Retrieval** via dual RAG architectures — embedding-based and novel agentic vectorless.
- **Multimodal Interaction** through voice input (Whisper) and voice output (gTTS).
- **Intelligent Recommendations** through cosine similarity and data-driven improvement tips.
- **Robust Input Validation** through Isolation Forest anomaly detection.

The project showcases how AI can transform subjective wine quality assessment into an objective, accessible, and educational experience — combining the precision of machine learning with the depth of expert wine knowledge.
