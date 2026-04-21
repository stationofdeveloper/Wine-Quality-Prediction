# 🍷 Wine Quality Prediction & AI Sommelier

<div align="center">

![Header](./header.png)

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-f55036?style=for-the-badge&logo=groq&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-189C7D?style=for-the-badge&logo=xgboost&logoColor=white)

**A sophisticated Wine Quality ecosystem combining predictive Machine Learning with an Agentic AI Sommelier.**  
From physicochemical analysis to professional tasting advice—powered by Agentic Vectorless RAG.

[🚀 Quick Start](#-quick-start) · [🤖 AI Sommelier](#-agentic-ai-sommelier) · [📊 ML Predictor](#-machine-learning-predictor) · [🎤 Voice AI](#-voice-interaction)

</div>

---

## 🌟 Key Features

### 🤖 Agentic AI Sommelier (RAG)
An intelligent chatbot powered by **Agentic Vectorless Retrieval**. Unlike traditional RAG, this agent reasons over a document tree to find exact pages in authoritative wine textbooks.
- **Authoritative Knowledge**: Grounded in chemistry, tasting, and sensory perception textbooks.
- **Retrieval Transparency**: UI renders "source cards" showing the exact book, section, and page being cited.
- **Hallucination-Free**: Strictly limited to provided knowledge base.

### 📊 Professional Wine Quality Predictor
Interactive machine learning dashboard to predict wine quality based on 11 chemical properties.
- **Model Suite**: Random Forest, Gradient Boosting, and XGBoost with automated fine-tuning.
- **Live EDA**: Interactive distributions, correlation heatmaps, and feature importance analysis.
- **Prediction Insights**: Real-time quality scoring with visual gauge and descriptive badges.

### 🎤 Advanced Voice Interaction
Fully integrated voice-activated assistant for hands-free wine analysis.
- **STT**: High-accuracy transcription via **Groq Whisper Large v3**.
- **TTS**: Natural speech feedback using **gTTS**.
- **Voice Commands**: Ask the Sommelier questions or provide wine stats via voice.

---

## 📁 Project Structure

```
Wine Quality Prediction/
│
├── 🧠 AI SOMMELIER & RAG
│   ├── 🐍 wine_chatbot.py        ← Main Chatbot Web App (Streamlit)
│   ├── 🐍 wine_rag_agent.py      ← Agentic Reasoning Engine
│   ├── 🐍 wine_rag_indexer.py    ← Vectorless Tree Indexer
│   ├── 📁 wine_knowledge/        ← Authority Textbooks (PDFs)
│   └── 📁 chroma_db/             ← (Optional) Vector Storage
│
├── 📊 MACHINE LEARNING
│   ├── 📓 wine_quality_ml.ipynb  ← Full Research & EDA Notebook
│   ├── 🐍 model_training.py      ← Automated Training Pipeline
│   ├── 🐍 app.py                 ← Prediction Web App (Streamlit)
│   └── 🐍 recommender.py         ← Feature-based Recommendation
│
├── 📦 ARTIFACTS
│   ├── 📦 best_wine_model.pkl    ← Deployed Regressor
│   ├── 📦 wine_scaler.pkl        ← Preprocessing Scaler
│   └── 🖼️ model_comparison.png  ← Benchmark Visuals
│
└── 🎤 UTILS
    └── 🐍 voice_handler.py       ← Speech-to-Text & Text-to-Speech
```

---

## 🚀 Quick Start

### 1 — Environment Setup
```bash
# Clone the repository
git clone https://github.com/stationofdeveloper/Wine-Quality-Prediction.git
cd Wine-Quality-Prediction

# Create & activate virtual environment
python -m venv .venv
source .venv/Scripts/activate  # macOS/Linux: source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2 — Configure Secrets
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key  # Optional fallback
```

### 3 — Run the Ecosystem
**To launch the AI Sommelier Chatbot:**
```bash
streamlit run wine_chatbot.py
```

**To launch the ML Quality Predictor:**
```bash
streamlit run app.py
```

---

## 🤖 Deep Dive: Agentic Vectorless RAG

This project implements a next-generation retrieval strategy inspired by **PageIndex**. Instead of converting text into flat vector embeddings (which often loses hierarchical context), our agent:
1. **Indexes** PDFs into a hierarchical JSON tree of sections and sub-sections.
2. **Reasons** using an LLM (LLaMA 3.3) to navigate this hierarchy based on your query.
3. **Retrieves** exact page content for synthesis.
4. **Cites** precisely which page of which textbook provided the answer.

---

## 📈 Model Performance

| Model | R² Score | MAE | RMSE |
|:---|:---|:---|:---|
| **XGBoost (Tuned)** | **0.53** | 0.37 | 0.51 |
| Random Forest (Tuned) | 0.52 | 0.37 | 0.51 |
| Gradient Boosting (Tuned) | 0.50 | 0.38 | 0.52 |

*Best model is automatically selected and serialized for production use.*

---

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq LLaMA 3.1 & 3.3 (70B)
- **ML**: Scikit-Learn, XGBoost, Pandas, NumPy
- **RAG**: Custom Tree Indexing (Vectify PageIndex inspired)
- **Voice**: Groq Whisper API (STT), gTTS (TTS)
- **EDA**: D-Tale, Seaborn, Matplotlib

---

## 👥 Author

**Your Name**
- GitHub: [@stationofdeveloper](https://github.com/stationofdeveloper)
- LinkedIn: [Your Profile](https://linkedin.com/in/your-profile)

---

<div align="center">
Made with ❤️ and 🍷
</div>