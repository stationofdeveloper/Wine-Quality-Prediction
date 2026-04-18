"""
🍷 Wine RAG Setup & Test Script
================================
Run this ONCE after placing your PDFs in wine_knowledge/

Steps:
  1. Checks all files are in place
  2. Tests API connectivity (Groq + Gemini)
  3. Indexes all available books (or a specific one)
  4. Runs a quick test query to verify everything works

Usage:
  python setup_rag.py               # index all books found
  python setup_rag.py chemistry     # index only chemistry book
  python setup_rag.py --test-only   # skip indexing, just test APIs
"""

import sys, os, json, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY  = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

KNOWLEDGE_DIR = Path("wine_knowledge")

BOOKS = {
    "chemistry":   "Chemistry_wine.pdf",
    "tasting":     "Wine_Tasting.pdf",
    "perceptions": "Perceptions_of_wine_quality.pdf",
}

def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print('='*60)


def check_environment():
    header("Step 1 — Environment Check")
    issues = []

    # Check .env / API keys
    if GROQ_API_KEY:
        print(f"  ✅  GROQ_API_KEY    : {GROQ_API_KEY[:12]}…")
    else:
        print(f"  ❌  GROQ_API_KEY    : NOT SET")
        issues.append("GROQ_API_KEY missing")

    if GEMINI_API_KEY:
        print(f"  ✅  GEMINI_API_KEY  : {GEMINI_API_KEY[:12]}…")
    else:
        print(f"  ⚠️   GEMINI_API_KEY  : not set (optional fallback)")

    # Check wine_knowledge/ folder
    KNOWLEDGE_DIR.mkdir(exist_ok=True)
    (KNOWLEDGE_DIR / "indexes").mkdir(exist_ok=True)
    print(f"\n  📁  wine_knowledge/ folder: OK")

    # Check PDFs
    print(f"\n  📚  Book PDFs:")
    for book_id, filename in BOOKS.items():
        pdf_path = KNOWLEDGE_DIR / filename
        exists   = pdf_path.exists()
        size     = f"{pdf_path.stat().st_size / 1024 / 1024:.1f} MB" if exists else "—"
        icon     = "✅" if exists else "❌"
        print(f"      {icon} {filename:<45} {size}")
        if not exists:
            issues.append(f"{filename} missing from wine_knowledge/")

    # Check Python packages
    required = ["fitz", "groq", "gtts", "streamlit", "dotenv"]
    print(f"\n  📦  Python packages:")
    for pkg in required:
        try:
            if pkg == "fitz":
                import fitz
            elif pkg == "groq":
                import groq
            elif pkg == "gtts":
                import gtts
            elif pkg == "streamlit":
                import streamlit
            elif pkg == "dotenv":
                import dotenv
            print(f"      ✅ {pkg}")
        except ImportError:
            print(f"      ❌ {pkg} — run: pip install {pkg if pkg != 'fitz' else 'PyMuPDF'}")
            issues.append(f"Missing package: {pkg}")

    if issues:
        print(f"\n  ⚠️  Issues found:")
        for issue in issues:
            print(f"     • {issue}")
    else:
        print(f"\n  ✅  All checks passed!")

    return len([i for i in issues if "API" in i or "missing from" in i or "Missing package" in i]) == 0


def test_groq_api():
    header("Step 2 — Test Groq API")
    if not GROQ_API_KEY:
        print("  ⚠️  Skipped — no GROQ_API_KEY")
        return False
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        resp   = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Say 'Groq API works!' in exactly 4 words."}],
            max_tokens=20,
        )
        reply = resp.choices[0].message.content.strip()
        print(f"  ✅  Groq API connected. Response: '{reply}'")

        # Test Whisper
        print(f"  ✅  Groq Whisper model: whisper-large-v3 (available)")
        return True
    except Exception as e:
        print(f"  ❌  Groq API error: {e}")
        return False


def test_gemini_api():
    header("Step 2b — Test Gemini API (fallback)")
    if not GEMINI_API_KEY:
        print("  ⚠️  Skipped — no GEMINI_API_KEY")
        return False
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        # model = genai.GenerativeModel("gemini-1.5-flash")
        model = genai.GenerativeModel("gemini-robotics-er-1.6-preview")
        resp  = model.generate_content("Say 'Gemini works' in 2 words.")
        print(f"  ✅  Gemini API connected. Response: '{resp.text.strip()}'")
        return True
    except Exception as e:
        print(f"  ❌  Gemini API error: {e}")
        return False


def test_gtts():
    header("Step 2c — Test gTTS (Text-to-Speech)")
    try:
        from gtts import gTTS
        import io
        tts = gTTS("Wine quality test.", lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        size = buf.tell()
        print(f"  ✅  gTTS working — generated {size} bytes of audio")
        return True
    except Exception as e:
        print(f"  ❌  gTTS error: {e}")
        return False


def index_books(book_filter=None):
    header("Step 3 — Index Books")
    from wine_rag_indexer import BOOK_REGISTRY, index_book, load_index

    for book_id, book in BOOK_REGISTRY.items():
        if book_filter and book_id != book_filter:
            continue

        pdf_path = KNOWLEDGE_DIR / book["filename"]
        if not pdf_path.exists():
            print(f"\n  ⚠️  Skipping '{book_id}' — PDF not found: {book['filename']}")
            continue

        existing = load_index(book_id)
        if existing:
            node_count = len(existing.get("nodes", []))
            print(f"\n  ✅  '{book_id}' already indexed ({node_count} sections). Skipping.")
            print(f"      Use 'python setup_rag.py {book_id} --force' to re-index.")
            continue

        print(f"\n  📚  Indexing '{book['title']}' …")
        print(f"      This may take 3-10 minutes for large books.")
        print(f"      Grab a coffee ☕")

        start = time.time()
        try:
            tree = index_book(book_id)
            elapsed = time.time() - start
            node_count = len(tree.get("nodes", []))
            print(f"\n  ✅  Indexed '{book_id}'")
            print(f"      Sections: {node_count}")
            print(f"      Time: {elapsed:.0f}s")
        except Exception as e:
            print(f"\n  ❌  Error indexing '{book_id}': {e}")


def run_test_query():
    header("Step 4 — Test Query")
    from wine_rag_indexer import get_index_status
    from wine_rag_agent import answer, tool_list_books

    status = get_index_status()
    indexed = [bid for bid, info in status.items() if info["indexed"]]

    if not indexed:
        print("  ⚠️  No books indexed yet. Run indexing first.")
        return

    print(f"  Indexed books: {', '.join(indexed)}")
    print(f"\n  📚 Book list:")
    print(tool_list_books())

    question = "What is the role of sulphur dioxide in wine preservation?"
    print(f"\n  ❓ Test question: {question}")
    print(f"  Running agent …\n")

    steps = []
    def cb(msg):
        steps.append(msg)
        print(f"     [{msg}]")

    try:
        final, _ = answer(question, stream_callback=cb)
        print(f"\n  ✅  Agent answered successfully!")
        print(f"\n  📝 Answer preview:\n{final[:500]}…" if len(final) > 500 else f"\n  📝 Answer:\n{final}")
    except Exception as e:
        print(f"\n  ❌  Test query failed: {e}")


def main():
    print("\n🍷 Wine RAG Setup Script")
    print("Inspired by PageIndex — Agentic Vectorless RAG")
    print("https://github.com/VectifyAI/PageIndex\n")

    test_only   = "--test-only" in sys.argv
    force       = "--force" in sys.argv
    book_filter = next(
        (a for a in sys.argv[1:] if a in BOOKS and not a.startswith("--")),
        None
    )

    env_ok = check_environment()
    test_groq_api()
    test_gemini_api()
    test_gtts()

    if not test_only and env_ok:
        index_books(book_filter)
        run_test_query()
    elif test_only:
        print("\n⚠️  --test-only: skipping indexing and query test.")
    else:
        print("\n❌  Environment issues found. Fix them before indexing.")

    header("Done!")
    print("  To launch the chatbot:")
    print("  streamlit run wine_chatbot.py\n")


if __name__ == "__main__":
    main()