# Biweekly Progress Report — 4

**Project Title:** Wine Quality Prediction Using Machine Learning  
**Report Period:** Week 7 – Week 8  
**Prepared By:** Aryan Sutariya  
**Date of Submission:** [DD/MM/YYYY]

---

## 1. Objective

To develop and integrate an advanced Agentic Vectorless RAG (Retrieval-Augmented Generation) system inspired by PageIndex, build a dedicated AI Sommelier Chatbot with multi-turn reasoning, create comprehensive setup and testing scripts, and finalize the project with complete documentation and deployment-ready packaging.

---

## 2. Work Summary

### Week 7: Agentic Vectorless RAG System & PDF Book Indexer

#### 2.1 Architectural Innovation — Agentic Vectorless RAG

Designed and implemented a novel retrieval approach inspired by [PageIndex](https://github.com/VectifyAI/PageIndex) that **eliminates the need for vector databases and embeddings entirely**. Instead of relying on semantic similarity scores (which can produce false positives), this system uses LLM reasoning to navigate structured document indexes.

**Key Advantages Over Traditional Vector RAG:**
| Aspect               | Vector RAG (ChromaDB)            | Agentic Vectorless RAG            |
|-----------------------|----------------------------------|-----------------------------------|
| Retrieval Method      | Embedding similarity search      | LLM reasoning over tree structure |
| Accuracy              | Approximate nearest neighbors    | Exact section + page identification |
| Citations             | Chunk-level (imprecise)          | Book + page number (precise)     |
| False Positives       | Common with semantic drift       | Eliminated through reasoning     |
| Dense Technical Books | Struggles with complex structure | Excels at hierarchical content   |

#### 2.2 PDF Book Indexer (`wine_rag_indexer.py`)

Built a comprehensive PDF book indexing system (401 lines) that creates hierarchical tree indexes from academic textbooks:

**Book Registry:**
Registered three authoritative wine reference books:
| Book ID       | Title                                                          | Pages |
|---------------|----------------------------------------------------------------|-------|
| `chemistry`   | Chemistry and Biochemistry of Winemaking, Wine Stabilization and Aging | 258   |
| `tasting`     | Wine Tasting: A Professional Handbook (2nd Edition)            | 519   |
| `perceptions` | Perceptions of Wine Quality                                    | 406   |

**Indexing Process (Dual-Mode):**

1. **Mode A — PDF TOC Extraction:**
   - Uses PyMuPDF (`fitz`) to extract the embedded Table of Contents from PDFs.
   - Converts TOC entries into a hierarchical tree with automatic `end_page` computation from sibling nodes.
   - Applies recursive nesting by TOC level to create chapter → section → subsection hierarchy.

2. **Mode B — LLM-Generated Index (Fallback):**
   - When no embedded TOC is available (or fewer than 5 entries), falls back to LLM-based indexing.
   - Processes pages in batches of 12 (`PAGES_PER_NODE`).
   - For each batch, extracts text with PyMuPDF and asks Groq LLaMA 3.1 8B (fast/free) to identify section titles and generate 2–3 sentence summaries.
   - Also generates a book-level overview from the first 30 pages.
   - Includes rate-limit handling with 1.5-second delays between batches.

**Resulting Tree Structure (JSON):**
```json
{
  "book_id": "chemistry",
  "title": "Chemistry and Biochemistry of Winemaking...",
  "total_pages": 258,
  "description": "This book covers...",
  "nodes": [
    {
      "title": "Introduction to Wine Chemistry",
      "node_id": "0000",
      "start_page": 1,
      "end_page": 24,
      "summary": "Covers the fundamental chemical compounds...",
      "nodes": [...]
    }
  ]
}
```

Indexes are cached as JSON files in `wine_knowledge/indexes/` and reused across sessions. Re-indexing is supported via CLI (`--force` flag) or the Streamlit UI.

#### 2.3 RAG Agent — ReAct Reasoning Loop (`wine_rag_agent.py`)

Developed a sophisticated agentic retrieval system (468 lines) implementing the **ReAct (Reason + Act)** loop pattern:

**4 Agent Tools:**

| Tool                | Purpose                                                | Input             |
|---------------------|--------------------------------------------------------|-------------------|
| `list_books()`      | Shows available books with descriptions and status     | None              |
| `get_structure(book_id)` | Returns hierarchical tree index (sections, page ranges, summaries) | book_id      |
| `get_pages(book_id, pages)` | Fetches actual text from specific pages (max 20 per call) | book_id, page range |
| `search_section(book_id, query)` | LLM-powered section finder — reasons over tree to find relevant pages | book_id, query |

**Agent Workflow:**
```
1. LLM receives question + system prompt with tool descriptions
2. LLM thinks: "This question is about wine chemistry..."
3. LLM calls tool: get_structure("chemistry")
   → Receives section titles, page ranges, and summaries
4. LLM thinks: "Chapter 3, pages 45-52 covers volatile acidity"
5. LLM calls tool: get_pages("chemistry", "45-52")
   → Receives actual page text
6. LLM synthesizes answer with citations: [Chemistry of Winemaking, p. 45-52]
```

**Implementation Details:**
- **System Prompt:** Comprehensive 60-line prompt defining the agent's personality (WineSommelier AI), available tools with exact calling syntax, and answering instructions.
- **Tool Call Parsing:** Regex-based extraction of `<tool_call>` blocks from LLM responses, supporting JSON-formatted tool invocations.
- **Iteration Control:** Maximum 6 ReAct iterations to prevent infinite loops; forces final answer after limit.
- **LLM Configuration:** Primary model: Groq LLaMA 3.1 70B Versatile for answering; Fast model: LLaMA 3.1 8B Instant for section finding.
- **Fallback Chain:** Groq → OpenRouter (Gemma 4 26B) → Error message.
- **Page Safety Cap:** Limits page retrieval to 20 pages per call to prevent context overflow.

---

### Week 8: Dedicated Chatbot UI, Setup Scripts & Project Finalization

#### 2.4 AI Sommelier Chatbot (`wine_chatbot.py`)

Built a standalone, dedicated chatbot interface (463 lines) as a separate Streamlit application focused entirely on the conversational RAG experience:

**UI Components:**
- **Chat Interface:** Styled chat bubbles with distinct visual treatment for user (maroon, right-aligned) and AI (white with border, left-aligned) messages.
- **Source Citations:** Automatically split from the main response and displayed in a styled citation card with amber background.
- **Suggested Questions:** 8 pre-configured domain-specific questions as clickable buttons for onboarding new users.
- **Live Agent Steps:** Real-time display of the agent's reasoning process (e.g., "🔧 Calling `get_structure(chemistry)`…") via a stream callback.

**Sidebar Features:**
- **📚 Wine Books Panel:** Displays all three books with their indexing status (✅ indexed, 📄 not indexed, ❌ PDF missing).
- **⚙️ Index Management:** Expandable panel for selecting and indexing/re-indexing individual books.
- **🎤 Voice Settings:** Toggle for voice responses, audio file upload with transcription.
- **📋 View Last Steps:** Button to inspect the agent's reasoning steps from the last query.
- **How It Works:** Architecture explanation panel describing the agentic vectorless RAG approach.

**Technical Integration:**
- Lazy-loaded modules (`wine_rag_agent`, `wine_rag_indexer`) via `@st.cache_resource` for fast app startup.
- Session state management for chat messages, agent history, step log, TTS state, and audio.
- Multi-turn conversation support by passing the last 3 turns (6 messages) as context to the agent.
- Integrated gTTS for voice responses with configurable toggle.

**Bottom Info Bar:** Displays the technology stack powering each component — LLM (Groq LLaMA 3.1 70B), RAG Type (Agentic Vectorless), STT (Groq Whisper), TTS (gTTS).

#### 2.5 Setup & Test Script (`setup_rag.py`)

Developed a comprehensive setup and validation script (260 lines) for automated environment verification and system testing:

**4-Step Validation Pipeline:**

| Step | Description                                            |
|------|--------------------------------------------------------|
| 1    | **Environment Check** — API keys, PDF files, Python packages |
| 2    | **API Tests** — Groq connectivity, Gemini fallback, gTTS |
| 3    | **Book Indexing** — Indexes all available PDF books    |
| 4    | **Test Query** — Runs a full agent query end-to-end    |

**CLI Options:**
- `python setup_rag.py` — Full setup (index all books + test)
- `python setup_rag.py chemistry` — Index only the chemistry book
- `python setup_rag.py --test-only` — Skip indexing, test APIs only
- `python setup_rag.py chemistry --force` — Force re-index of a specific book

#### 2.6 Environment Configuration (`.env`)

Configured secure API key management using python-dotenv:
- `GROQ_API_KEY` — Primary LLM and STT provider
- `GEMINI_API_KEY` — LLM fallback provider
- `OPENROUTER_API_KEY` — Additional LLM fallback via OpenRouter
- `OPENROUTER_CHAT_MODEL` — Configurable model selection (default: `google/gemma-4-26b-a4b-it:free`)

#### 2.7 Project Documentation

Finalized a comprehensive `README.md` (348 lines) covering:
- Project structure with file descriptions
- Dataset documentation with feature descriptions
- Quick start guide (6-step setup)
- Complete ML pipeline documentation
- Preprocessing steps and feature engineering details
- Model training configurations and hyperparameter grids
- Model comparison results table
- Streamlit app feature overview
- Artifact file descriptions with code examples
- Tech stack table with library purposes
- Requirements specification with version constraints

---

## 3. Final System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   WINE AI SOMMELIER                     │
├─────────────────────┬───────────────────────────────────┤
│   app.py            │   wine_chatbot.py                 │
│   (Prediction App)  │   (Dedicated Chatbot)             │
├─────────────────────┼───────────────────────────────────┤
│ model_training.py   │   wine_rag_agent.py               │
│ (ML Pipeline)       │   (ReAct Agent Loop)              │
├─────────────────────┼───────────────────────────────────┤
│ explainer.py        │   wine_rag_indexer.py             │
│ (SHAP + LLM)        │   (PDF Tree Indexer)              │
├─────────────────────┼───────────────────────────────────┤
│ recommender.py      │   rag_engine.py                   │
│ (Similarity + IF)   │   (ChromaDB Vector RAG)           │
├─────────────────────┼───────────────────────────────────┤
│ voice_handler.py    │   setup_rag.py                    │
│ (Whisper + gTTS)    │   (Setup & Validation)            │
├─────────────────────┴───────────────────────────────────┤
│                     Data Layer                          │
│  winequality-red.csv │ *.pkl models │ wine_knowledge/   │
└─────────────────────────────────────────────────────────┘
```

---

## 4. Tools & Technologies Used

| Tool / Library          | Purpose                                        |
|-------------------------|-------------------------------------------------|
| PyMuPDF (fitz)          | PDF text extraction and TOC parsing            |
| Groq API                | LLM (LLaMA 3.1), Whisper STT, indexing         |
| OpenRouter API          | LLM fallback (Gemma 4 26B)                     |
| Streamlit               | Chatbot UI and prediction app                  |
| gTTS                    | Free text-to-speech                            |
| python-dotenv           | Secure environment variable management         |
| JSON                    | Tree index serialization and caching           |

---

## 5. Challenges Encountered

1. **Agentic Loop Stability:** The ReAct loop can occasionally enter repetitive tool-calling cycles. Addressed by setting `MAX_ITERATIONS=6` and forcing a final answer synthesis after reaching the limit.
2. **PDF Text Quality:** Some PDF pages contain images, tables, or non-standard fonts that PyMuPDF extracts as garbled text. Added a minimum text length check (`< 50 chars`) to skip empty/unreadable pages during indexing.
3. **Context Window Management:** Large book structures (500+ pages) can exceed LLM context limits. Implemented truncation at 8,000 characters for structure responses and 7,000 characters for page retrieval, with a suggestion to use `search_section` for targeted lookup.
4. **Rate Limit Management:** Groq free-tier limits (30 requests/minute) required careful pacing during book indexing. Implemented 1.5-second delays between batches and exponential backoff on 429 errors.
5. **Multi-Model Coordination:** Coordinating 3 different LLM providers (Groq, Gemini, OpenRouter) with different API interfaces required a unified fallback chain with consistent error handling.

---

## 6. Deliverables

| Deliverable                                     | Status      |
|-------------------------------------------------|-------------|
| Agentic Vectorless RAG architecture             | ✅ Complete |
| PDF Book Indexer (dual-mode: TOC + LLM)         | ✅ Complete |
| ReAct agent with 4 tools                        | ✅ Complete |
| Dedicated AI Sommelier Chatbot                  | ✅ Complete |
| Multi-turn conversation support                 | ✅ Complete |
| Live agent reasoning display                    | ✅ Complete |
| Source citation system                          | ✅ Complete |
| Setup & validation script                       | ✅ Complete |
| Environment configuration & API key management  | ✅ Complete |
| Comprehensive README documentation              | ✅ Complete |
| Full system integration and testing             | ✅ Complete |

---

## 7. Project Summary

Over the 8-week development period, the Wine Quality Prediction project evolved from a basic ML regression pipeline into a comprehensive AI-powered wine analysis platform:

| Phase (Report) | Key Achievement                                         |
|-----------------|--------------------------------------------------------|
| Report 1        | Data exploration, EDA, and 3 baseline models           |
| Report 2        | Hyperparameter tuning and Streamlit prediction app     |
| Report 3        | SHAP explainability, vector RAG, recommender, voice AI |
| Report 4        | Agentic vectorless RAG, chatbot, and full integration  |

**Final System Capabilities:**
- ✅ ML quality prediction with 3 tuned models
- ✅ SHAP-powered model interpretability with LLM explanations
- ✅ Dual RAG systems (vector-based + agentic vectorless)
- ✅ AI Sommelier chatbot with ReAct reasoning
- ✅ Voice input (Whisper STT) and output (gTTS TTS)
- ✅ Similar wine recommender with cosine similarity
- ✅ Anomaly detection with Isolation Forest
- ✅ Data-driven improvement tips
- ✅ Two Streamlit applications (prediction + chatbot)
- ✅ Comprehensive documentation and setup automation

**Total Codebase:** ~3,500+ lines of Python across 10 source files.
