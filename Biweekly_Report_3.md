# Biweekly Progress Report — 3

**Project Title:** Wine Quality Prediction Using Machine Learning  
**Report Period:** Week 5 – Week 6  
**Prepared By:** Aryan Sutariya  
**Date of Submission:** [DD/MM/YYYY]

---

## 1. Objective

To enhance the Wine Quality Prediction system with advanced AI features: model interpretability via SHAP, a Retrieval-Augmented Generation (RAG) engine for domain-specific knowledge retrieval, a similar wine recommender, anomaly detection for input validation, and voice AI capabilities for multimodal interaction.

---

## 2. Work Summary

### Week 5: SHAP Explainer, RAG Engine & Wine Knowledge Base

#### 2.1 SHAP-Based Model Explainability (`explainer.py`)

Developed a comprehensive model explanation module (241 lines) that combines SHAP (SHapley Additive exPlanations) with LLM-powered natural language generation:

**SHAP Implementation:**
- Utilized `shap.TreeExplainer` for computing feature-level attributions — compatible with all three model types (Random Forest, Gradient Boosting, XGBoost).
- Implemented explainer caching (`_shap_explainer_cache`) to avoid redundant computation across multiple predictions in the same session.
- Outputs sorted `(feature_name, shap_value)` pairs ranked by absolute impact, enabling identification of the top contributing features for each individual prediction.
- Integrated SHAP bar charts in the Streamlit app showing the top 6 features with green (positive impact) and red (negative impact) color-coding.

**LLM-Powered Explanation Generation:**
- Built a multi-tier LLM fallback system for generating human-readable explanations:
  1. **Primary:** Groq API (LLaMA 3.1 70B Versatile) — free, fastest
  2. **Fallback:** Google Gemini Flash (Gemini 1.5 Flash) — free
  3. **Offline:** Rule-based explanation engine — no API required
- The `explain_prediction()` function constructs a detailed prompt incorporating:
  - Predicted quality score and its meaning
  - SHAP attribution summary (top 5 features)
  - Chemical property highlights (flagging high/low values with context)
  - Retrieved RAG context for domain-specific grounding
- System prompt instructs the LLM to act as a sommelier-data scientist hybrid, producing 3–5 sentence explanations with one actionable improvement tip.

#### 2.2 RAG Engine — ChromaDB Vector Store (`rag_engine.py`)

Built a local vector-based Retrieval-Augmented Generation engine (151 lines) for augmenting LLM responses with domain-specific wine knowledge:

**Architecture:**
- **Embedding Model:** `all-MiniLM-L6-v2` from `sentence-transformers` — a lightweight (22M params), fast, free model requiring no API key.
- **Vector Store:** ChromaDB (`PersistentClient`) stored locally in `chroma_db/` directory for persistence across sessions.
- **Knowledge Base:** Text files in `wine_knowledge/` directory:
  - `wine_chemistry.txt` — Chemical compounds and their effects on wine quality
  - `tasting_notes.txt` — Professional wine tasting terminology and methods
  - `food_pairings.txt` — Food and wine pairing guidelines

**Text Processing & Retrieval:**
- Implemented overlapping chunking (`_chunk_text`) with configurable `chunk_size=400` words and `overlap=60` words for better context preservation.
- Batch embedding and indexing with `batch_size=50` to handle large knowledge bases efficiently.
- `retrieve()` function encodes the query, performs similarity search, and returns the top-k (default 4) most relevant chunks concatenated for LLM context injection.
- Auto-builds index on first access if the collection doesn't exist; provides `rebuild_index()` for manual re-indexing after knowledge base updates.

#### 2.3 Wine Knowledge Base Curation

Assembled a domain-specific knowledge base in `wine_knowledge/`:

| File                 | Size    | Content                                              |
|----------------------|---------|------------------------------------------------------|
| `wine_chemistry.txt` | 4.7 KB  | Chemical compounds, their ranges, and effects on quality |
| `tasting_notes.txt`  | 4.6 KB  | Professional tasting methods, sensory evaluation criteria |
| `food_pairings.txt`  | 3.5 KB  | Food pairing recommendations by wine style           |

Additionally, sourced three academic PDF references for advanced RAG (used in Week 7-8):
- *Chemistry and Biochemistry of Winemaking* (14.6 MB, 258 pages)
- *Wine Tasting: A Professional Handbook* (7.1 MB, 519 pages)
- *Perceptions of Wine Quality* (1.5 MB, 406 pages)

---

### Week 6: Similar Wine Recommender, Anomaly Detection & Voice AI

#### 2.4 Similar Wine Recommender (`recommender.py`)

Developed a recommendation engine (224 lines) that finds chemically similar high-quality wines from the dataset:

**Similarity-Based Recommendation:**
- Filters the dataset for wines scoring above a user-defined quality threshold (default ≥ 7).
- Scales both user input and dataset using a locally-fitted `StandardScaler` on the filtered subset.
- Computes **Euclidean distance** in the scaled feature space between user input and all qualifying wines.
- Returns top-N most similar wines with a similarity percentage (100% = identical).
- Integrated in the Streamlit app with a side-by-side bar chart comparing user input vs. best match across key features (alcohol, volatile acidity, sulphates, citric acid, pH, fixed acidity).

**Data-Driven Improvement Tips:**
- `improvement_tips()` function compares user values to the median values of wines scoring 7+.
- Generates specific, actionable tips with concrete numbers (e.g., "volatile acidity is HIGH (0.72). Aim below 0.5 — high acetic acid causes vinegar taste.").
- Covers 6 key features: volatile acidity, alcohol, sulphates, citric acid, pH, and fixed acidity.

#### 2.5 Anomaly Detection (`recommender.py`)

Implemented input validation using an **Isolation Forest** model to detect unrealistic or unusual input combinations:

- Trained on the clean wine dataset features with `n_estimators=200` and `contamination=0.05` (expects ~5% natural outliers).
- Computes anomaly scores for user input and flags inputs that fall outside the training distribution.
- Identifies which specific features deviate most from the dataset median using z-score analysis (flags features with z > 3.5σ).
- Provides clear warning messages: either specifying the unusual features with their z-scores, or noting that the overall combination is unusual.
- Result displayed as a styled tip/warning box in the Streamlit prediction tab.

#### 2.6 Voice AI — Speech-to-Text & Text-to-Speech (`voice_handler.py`)

Built a bidirectional voice interface (169 lines) for multimodal interaction:

**Speech-to-Text (STT):**
- Integrated **Groq Whisper API** (`whisper-large-v3` model) for transcribing audio input — free tier, no cost.
- Accepts WAV, MP3, and M4A audio file uploads via the Streamlit sidebar.
- Processes audio through a temporary file pipeline for API compatibility.

**Spoken Feature Parsing:**
- `parse_features_from_text()` function extracts wine chemical property values from natural language input.
- Supports multiple aliases per feature (e.g., "volatile acidity", "volatile acid", "acetic" all map to `volatile acidity`).
- Uses regex pattern matching to handle formats like "alcohol 12.5" or "pH = 3.4".
- Parsed values automatically populate the corresponding sidebar sliders.

**Text-to-Speech (TTS):**
- Integrated **gTTS (Google Text-to-Speech)** — completely free, no API key required.
- `text_to_speech()` converts explanation text to MP3 audio bytes.
- `build_voice_prompt_for_wine()` constructs natural spoken responses summarizing prediction results and offering follow-up prompts.
- Audio playback embedded in the Streamlit app beneath predictions and chat responses.

---

## 3. Tools & Technologies Used

| Tool / Library          | Purpose                                        |
|-------------------------|-------------------------------------------------|
| SHAP                    | Model-agnostic feature attribution             |
| ChromaDB                | Local persistent vector store                  |
| sentence-transformers   | Text embedding model (MiniLM-L6-v2)            |
| Groq API                | LLM chat (LLaMA 3.1) + Whisper STT            |
| Google Gemini API       | LLM fallback (Gemini 1.5 Flash)               |
| gTTS                    | Free text-to-speech                            |
| scikit-learn            | Isolation Forest, StandardScaler               |
| python-dotenv           | Secure API key management via `.env` file      |

---

## 4. Challenges Encountered

1. **SHAP Compatibility:** Tree-based SHAP explainers handle multi-dimensional output differently across model types. Added dimension-checking logic (`shap_vals.ndim > 1`) to handle this gracefully.
2. **RAG Chunk Optimization:** Finding the right chunk size/overlap ratio to balance retrieval precision and context completeness. Settled on 400 words with 60-word overlap after experimentation.
3. **Voice Feature Parsing:** Natural language variations in how users express feature values (e.g., "12 and a half percent alcohol" vs. "alcohol 12.5") required a flexible regex-based approach with multiple aliases.
4. **LLM Rate Limiting:** Groq API free-tier rate limits necessitated retry logic with exponential backoff and a fallback chain to Gemini and then rule-based explanations.

---

## 5. Deliverables

| Deliverable                                    | Status      |
|------------------------------------------------|-------------|
| SHAP explainer module with LLM integration     | ✅ Complete |
| SHAP visualization in Streamlit app            | ✅ Complete |
| ChromaDB RAG engine with wine knowledge base   | ✅ Complete |
| Wine knowledge base (3 text files, 3 PDFs)     | ✅ Complete |
| Similar wine recommender with visualization    | ✅ Complete |
| Anomaly detection with Isolation Forest        | ✅ Complete |
| Data-driven improvement tips                   | ✅ Complete |
| Voice input (Groq Whisper STT)                 | ✅ Complete |
| Voice output (gTTS TTS)                        | ✅ Complete |
| Spoken feature parsing                         | ✅ Complete |
| Multi-tier LLM fallback system                 | ✅ Complete |

---

## 6. Plan for Next Period

- Develop an **Agentic Vectorless RAG** system inspired by PageIndex for reasoning-based retrieval from full PDF textbooks.
- Build a **PDF Book Indexer** that creates hierarchical tree indexes from academic wine textbooks.
- Create a dedicated **AI Sommelier Chatbot** with agentic (ReAct) reasoning, multi-turn conversation support, and source citations.
- Implement **setup, testing, and deployment** scripts for the complete system.
