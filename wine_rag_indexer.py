"""
🌲 Wine Book Indexer — Agentic Vectorless RAG (PageIndex-style)
================================================================
Inspired by: https://github.com/VectifyAI/PageIndex

How it works (NO vectors, NO chunking):
  1. Extract PDF text page-by-page with PyMuPDF
  2. Try to find the actual Table of Contents embedded in PDF
  3. If no TOC → use LLM (Groq llama-3.1-8b-instant) to generate
     section summaries for every N-page batch
  4. Build a hierarchical JSON tree:
       Book → Chapters → Sections → {pages, summary}
  5. Save tree as JSON (indexed ONCE, reused forever)

The tree is then used by wine_rag_agent.py for reasoning-based retrieval.

Install: pip install PyMuPDF groq python-dotenv
"""

import os, json, time, re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
INDEX_MODEL    = "llama-3.1-8b-instant"   # fast + free for indexing
PAGES_PER_NODE = 12                        # pages per leaf node in tree
KNOWLEDGE_DIR  = Path("wine_knowledge")
INDEX_DIR      = Path("wine_knowledge/indexes")

# ── Book registry ──────────────────────────────────────────────────────────────
BOOK_REGISTRY = {
    "chemistry": {
        "id":       "chemistry",
        "filename": "Chemistry_wine.pdf",
        "title":    "Chemistry and Biochemistry of Winemaking, Wine Stabilization and Aging",
        "pages":    258,
        "emoji":    "⚗️",
    },
    "tasting": {
        "id":       "tasting",
        "filename": "Wine_Tasting.pdf",
        "title":    "Wine Tasting: A Professional Handbook (2nd Edition)",
        "pages":    519,
        "emoji":    "🍷",
    },
    "perceptions": {
        "id":       "perceptions",
        "filename": "Perceptions_of_wine_quality.pdf",
        "title":    "Perceptions of Wine Quality",
        "pages":    406,
        "emoji":    "🎯",
    },
}


# ── PDF utilities ──────────────────────────────────────────────────────────────

def extract_page_text(pdf_path: Path, page_num: int) -> str:
    """Extract text from a single page (0-indexed internally, 1-indexed in output)."""
    try:
        import fitz
        doc  = fitz.open(str(pdf_path))
        page = doc[page_num]
        text = page.get_text("text")
        doc.close()
        return text.strip()
    except Exception as e:
        return f"[Error reading page {page_num+1}: {e}]"


def extract_pages_text(pdf_path: Path, start: int, end: int,
                       max_chars: int = 6000) -> str:
    """Extract text from a page range (1-indexed, inclusive). Truncates if needed."""
    try:
        import fitz
        doc    = fitz.open(str(pdf_path))
        parts  = []
        for i in range(start - 1, min(end, len(doc))):
            text = doc[i].get_text("text").strip()
            if text:
                parts.append(f"--- Page {i+1} ---\n{text}")
        doc.close()
        combined = "\n\n".join(parts)
        return combined[:max_chars]
    except Exception as e:
        return f"[Error: {e}]"


def extract_pdf_toc(pdf_path: Path) -> list[dict]:
    """
    Try to extract the built-in PDF Table of Contents.
    Returns list of {level, title, page} dicts, or [] if no TOC.
    """
    try:
        import fitz
        doc  = fitz.open(str(pdf_path))
        toc  = doc.get_toc()   # [[level, title, page], ...]
        doc.close()
        return [{"level": t[0], "title": t[1], "page": t[2]} for t in toc]
    except Exception:
        return []


def get_total_pages(pdf_path: Path) -> int:
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        n   = len(doc)
        doc.close()
        return n
    except Exception:
        return 0


# ── Groq helpers ───────────────────────────────────────────────────────────────

def _groq_call(prompt: str, system: str, model: str = INDEX_MODEL,
               max_tokens: int = 500, retries: int = 3) -> str:
    """Call Groq API with retry on rate-limit (429)."""
    if not GROQ_API_KEY:
        return "[No GROQ_API_KEY set]"
    from groq import Groq, RateLimitError
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system",  "content": system},
                    {"role": "user",    "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            wait = 20 * (attempt + 1)
            print(f"   Rate limit hit — waiting {wait}s …")
            time.sleep(wait)
        except Exception as e:
            if attempt == retries - 1:
                return f"[LLM error: {e}]"
            time.sleep(5)
    return "[Failed after retries]"


# ── Tree building ───────────────────────────────────────────────────────────────

def _toc_to_tree(toc: list[dict], total_pages: int,
                 book_id: str, book_title: str) -> dict:
    """
    Convert PDF TOC list into PageIndex-style hierarchical tree.
    Automatically computes end_page from the next sibling's start_page.
    """
    if not toc:
        return {}

    # Assign end_pages
    nodes_flat = []
    for i, entry in enumerate(toc):
        start = entry["page"]
        end   = toc[i+1]["page"] - 1 if i+1 < len(toc) else total_pages
        end   = max(start, end)
        nodes_flat.append({
            "title":      entry["title"],
            "node_id":    f"{i:04d}",
            "start_page": start,
            "end_page":   end,
            "level":      entry["level"],
            "summary":    "",
            "nodes":      [],
        })

    # Nest by level
    def nest(items, parent_level=0):
        result = []
        i = 0
        while i < len(items):
            item = items[i]
            if item["level"] == parent_level + 1:
                children_raw = []
                j = i + 1
                while j < len(items) and items[j]["level"] > parent_level + 1:
                    children_raw.append(items[j])
                    j += 1
                item["nodes"] = nest(children_raw, parent_level + 1)
                del item["level"]
                result.append(item)
                i = j
            else:
                i += 1
        return result

    return {
        "book_id":     book_id,
        "title":       book_title,
        "total_pages": total_pages,
        "description": "",
        "nodes":       nest(nodes_flat),
    }


def _llm_generate_tree(pdf_path: Path, book_id: str, book_title: str,
                       total_pages: int) -> dict:
    """
    LLM-based tree generation when no PDF TOC is available.
    Processes pages in batches, asks LLM to identify sections.
    """
    print(f"   No embedded TOC — using LLM to build tree index …")
    print(f"   Total pages: {total_pages}  |  Batch size: {PAGES_PER_NODE}")

    nodes      = []
    node_count = 0

    # First, extract first ~30 pages to get a book overview
    overview_text = extract_pages_text(pdf_path, 1, min(30, total_pages),
                                       max_chars=8000)
    overview = _groq_call(
        prompt=(
            f"Here are the first pages of '{book_title}':\n\n{overview_text}\n\n"
            "In 3 sentences, describe what this book covers and its key themes."
        ),
        system="You are an expert book analyst. Be concise.",
        max_tokens=200,
    )
    print(f"   Overview generated ✓")

    # Process in PAGES_PER_NODE-page batches
    batch_num = 0
    for start in range(1, total_pages + 1, PAGES_PER_NODE):
        end       = min(start + PAGES_PER_NODE - 1, total_pages)
        batch_num += 1

        print(f"   Indexing pages {start}–{end} (batch {batch_num}) …", end="", flush=True)
        page_text = extract_pages_text(pdf_path, start, end, max_chars=5000)

        if len(page_text.strip()) < 50:
            print(" (empty, skipping)")
            continue

        section_json = _groq_call(
            prompt=(
                f"Analyze pages {start}-{end} of '{book_title}'.\n\n"
                f"Text sample:\n{page_text[:3000]}\n\n"
                "Return ONLY valid JSON (no markdown) with this structure:\n"
                '{"title": "Section Title", "summary": "2-3 sentence summary"}'
            ),
            system=(
                "You are a document indexer. Extract section titles and summaries. "
                "Return ONLY raw JSON, no markdown backticks."
            ),
            max_tokens=200,
        )

        # Parse LLM JSON response
        try:
            # Strip any accidental markdown
            clean = re.sub(r"```(?:json)?|```", "", section_json).strip()
            parsed = json.loads(clean)
            title   = parsed.get("title",   f"Section {batch_num}")
            summary = parsed.get("summary", "")
        except Exception:
            title   = f"Pages {start}–{end}"
            summary = page_text[:200]

        nodes.append({
            "title":      title,
            "node_id":    f"{node_count:04d}",
            "start_page": start,
            "end_page":   end,
            "summary":    summary,
            "nodes":      [],
        })
        node_count += 1
        print(f" → '{title[:40]}' ✓")
        time.sleep(1.5)   # respect Groq rate limit

    return {
        "book_id":     book_id,
        "title":       book_title,
        "total_pages": total_pages,
        "description": overview,
        "nodes":       nodes,
    }


# ── Public API ─────────────────────────────────────────────────────────────────

def index_book(book_id: str, force: bool = False) -> dict:
    """
    Index a book and return its tree structure.

    If the index JSON already exists and force=False, loads from cache.
    Otherwise, builds the tree from the PDF (takes several minutes for large books).
    """
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    book     = BOOK_REGISTRY.get(book_id)
    if not book:
        raise ValueError(f"Unknown book_id: {book_id}")

    index_path = INDEX_DIR / f"{book_id}_index.json"
    pdf_path   = KNOWLEDGE_DIR / book["filename"]

    # Load cached index if available
    if index_path.exists() and not force:
        print(f"✅  Loaded cached index for '{book['title']}'")
        return json.loads(index_path.read_text(encoding="utf-8"))

    # Check PDF exists
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF not found: {pdf_path}\n"
            f"Place {book['filename']} inside wine_knowledge/ folder."
        )

    print(f"\n{'='*60}")
    print(f"📚 Indexing: {book['title']}")
    print(f"{'='*60}")
    total_pages = get_total_pages(pdf_path)
    print(f"   Pages: {total_pages}")

    # Try embedded TOC first
    toc = extract_pdf_toc(pdf_path)
    if len(toc) >= 5:
        print(f"   Found embedded TOC with {len(toc)} entries ✓")
        tree = _toc_to_tree(toc, total_pages, book_id, book["title"])

        # Generate description separately
        overview_text = extract_pages_text(pdf_path, 1, min(10, total_pages),
                                           max_chars=4000)
        tree["description"] = _groq_call(
            prompt=(
                f"First pages of '{book['title']}':\n\n{overview_text}\n\n"
                "In 2-3 sentences, describe what this book covers."
            ),
            system="You are a book analyst. Be concise.",
            max_tokens=150,
        )
    else:
        # LLM-based tree generation
        tree = _llm_generate_tree(pdf_path, book_id, book["title"], total_pages)

    # Save index
    index_path.write_text(json.dumps(tree, indent=2, ensure_ascii=False),
                          encoding="utf-8")
    print(f"\n✅  Index saved → {index_path}")
    return tree


def load_index(book_id: str) -> Optional[dict]:
    """Load a cached index. Returns None if not yet indexed."""
    index_path = INDEX_DIR / f"{book_id}_index.json"
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    return None


def get_index_status() -> dict:
    """Return status of all book indexes."""
    status = {}
    for book_id, book in BOOK_REGISTRY.items():
        pdf_path   = KNOWLEDGE_DIR / book["filename"]
        index_path = INDEX_DIR / f"{book_id}_index.json"
        status[book_id] = {
            "title":    book["title"],
            "emoji":    book["emoji"],
            "pdf_exists":   pdf_path.exists(),
            "indexed":      index_path.exists(),
            "index_path":   str(index_path) if index_path.exists() else None,
        }
    return status


def index_all_books(force: bool = False):
    """Index all books that have their PDFs available."""
    for book_id, book in BOOK_REGISTRY.items():
        pdf_path = KNOWLEDGE_DIR / book["filename"]
        if pdf_path.exists():
            try:
                index_book(book_id, force=force)
            except Exception as e:
                print(f"❌  Error indexing {book_id}: {e}")
        else:
            print(f"⚠️   PDF not found for {book_id}: {book['filename']}")


# ── CLI ─────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    force = "--force" in sys.argv
    book  = next((a for a in sys.argv[1:] if a in BOOK_REGISTRY), None)

    if book:
        index_book(book, force=force)
    else:
        print("Indexing all available books …")
        index_all_books(force=force)