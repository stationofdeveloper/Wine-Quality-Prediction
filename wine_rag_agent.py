"""
🤖 Wine RAG Agent — Agentic Vectorless Retrieval
=================================================
Inspired by PageIndex (https://github.com/VectifyAI/PageIndex)

RETRIEVAL TRANSPARENCY (new):
  Every time the agent fetches pages, it fires a structured
  RETRIEVE:: event via stream_callback so the UI renders a
  "source card" (book · section · pages · preview) BEFORE the answer.

  Event format:
    RETRIEVE::{"book_id":..., "book_title":..., "book_emoji":...,
               "pages":..., "section":..., "preview":...}
"""

import os, json, re, time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
ANSWER_MODEL   = os.getenv("GROQ_CHAT_MODEL", "meta-llama/llama-3.3-70b-versatile")
FAST_MODEL     = "llama-3.1-8b-instant"
MAX_ITERATIONS = 6

# Prefix constants used by the UI to distinguish event types
RETRIEVE_PREFIX = "RETRIEVE::"

from wine_rag_indexer import (
    load_index, extract_pages_text, BOOK_REGISTRY, KNOWLEDGE_DIR
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM helpers
# ─────────────────────────────────────────────────────────────────────────────

def _groq(messages: list[dict], model: str = ANSWER_MODEL,
          max_tokens: int = 1500) -> str:
    if not GROQ_API_KEY:
        return _gemini_fallback(messages[-1]["content"], max_tokens)
    from groq import Groq, RateLimitError
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, messages=messages,
                max_tokens=max_tokens, temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            time.sleep(15 * (attempt + 1))
        except Exception as e:
            if attempt == 2:
                return _gemini_fallback(messages[-1]["content"], max_tokens)
            time.sleep(3)
    return _gemini_fallback(messages[-1]["content"], max_tokens)


def _gemini_fallback(prompt: str, max_tokens: int = 1500) -> str:
    if not GEMINI_API_KEY:
        return "I need an API key to answer. Check your .env file."
    try:
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=GEMINI_API_KEY)
        for model_name in ["gemini-1.5-flash", "gemini-2.0-flash-exp", "gemini-2.0-flash"]:
            try:
                resp = client.models.generate_content(
                    model=model_name, contents=prompt,
                    config=types.GenerateContentConfig(
                        max_output_tokens=max_tokens, temperature=0.3
                    )
                )
                if resp and resp.text:
                    return resp.text.strip()
            except Exception:
                continue
        return "Gemini fallback failed."
    except Exception as e:
        return f"Error: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# Tree lookup: find which section covers a page range
# ─────────────────────────────────────────────────────────────────────────────

def _find_section_for_pages(book_id: str, start: int, end: int) -> str:
    """Return the section title whose page range best covers (start, end)."""
    tree = load_index(book_id)
    if not tree:
        return ""
    best_title, best_overlap = "", 0

    def walk(nodes):
        nonlocal best_title, best_overlap
        for node in nodes:
            ns = node.get("start_page", 0)
            ne = node.get("end_page",   0)
            overlap = max(0, min(end, ne) - max(start, ns))
            if overlap > best_overlap:
                best_overlap = overlap
                best_title   = node.get("title", "")
            walk(node.get("nodes", []))

    walk(tree.get("nodes", []))
    return best_title


# ─────────────────────────────────────────────────────────────────────────────
# Agent tools
# ─────────────────────────────────────────────────────────────────────────────

def tool_list_books() -> str:
    lines = ["Available wine books:\n"]
    for book_id, book in BOOK_REGISTRY.items():
        tree     = load_index(book_id)
        pdf_path = KNOWLEDGE_DIR / book["filename"]
        status   = ("✅ indexed"     if tree else
                    "📄 not indexed" if pdf_path.exists() else
                    "❌ PDF missing")
        desc = (tree.get("description", "")[:120] + "…") if tree else ""
        lines.append(
            f"• book_id: '{book_id}'\n"
            f"  Title: {book['title']}\n"
            f"  Pages: {book['pages']}\n"
            f"  Status: {status}\n"
            + (f"  About: {desc}\n" if desc else "")
        )
    return "\n".join(lines)


def tool_get_structure(book_id: str) -> str:
    tree = load_index(book_id)
    if not tree:
        book = BOOK_REGISTRY.get(book_id)
        if not book:
            return f"Unknown book_id '{book_id}'. Use list_books() to see valid IDs."
        return (
            f"Book '{book['title']}' is not yet indexed.\n"
            f"Run: python wine_rag_indexer.py {book_id}"
        )

    def format_node(node: dict, indent: int = 0) -> str:
        pad     = "  " * indent
        s_page  = node.get("start_page", "?")
        e_page  = node.get("end_page",   "?")
        title   = node.get("title", "Untitled")
        summary = node.get("summary", "")[:120]
        line    = f"{pad}[p.{s_page}-{e_page}] {title}"
        if summary:
            line += f"\n{pad}   ↳ {summary}"
        children = "\n".join(format_node(c, indent + 1)
                             for c in node.get("nodes", []))
        return line + ("\n" + children if children else "")

    book_info = (
        f"Book: {tree['title']}\n"
        f"Total pages: {tree['total_pages']}\n"
        f"Description: {tree.get('description', '')}\n\n"
        f"Structure:\n"
    )
    structure = "\n".join(format_node(n) for n in tree.get("nodes", []))
    result    = book_info + structure
    if len(result) > 8000:
        result = result[:8000] + \
            "\n\n[… truncated — use search_section for specific content]"
    return result


def tool_get_pages(book_id: str, pages: str, stream_callback=None) -> str:
    """
    Fetch page text and fire a RETRIEVE:: event for UI source card rendering.
    """
    book = BOOK_REGISTRY.get(book_id)
    if not book:
        return f"Unknown book_id '{book_id}'."

    pdf_path = KNOWLEDGE_DIR / book["filename"]
    if not pdf_path.exists():
        return f"PDF not found: {book['filename']}. Place it in wine_knowledge/."

    try:
        if "-" in pages and "," not in pages:
            parts = pages.split("-")
            start, end = int(parts[0].strip()), int(parts[1].strip())
        elif "," in pages:
            nums       = [int(x.strip()) for x in pages.split(",")]
            start, end = min(nums), max(nums)
        else:
            start = end = int(pages.strip())

        tree = load_index(book_id)
        if tree:
            end = min(end, tree.get("total_pages", book["pages"]))

        if end - start > 20:
            return (
                f"⚠️ Requesting {end-start+1} pages is too many (max 20). "
                "Please narrow the page range."
            )

        text = extract_pages_text(pdf_path, start, end, max_chars=7000)
        if not text.strip():
            text = "[No text found on these pages]"

        # Look up which section these pages belong to
        section = _find_section_for_pages(book_id, start, end)

        # ── Fire RETRIEVE:: event ─────────────────────────────────────────────
        if stream_callback:
            payload = json.dumps({
                "book_id":    book_id,
                "book_title": book["title"],
                "book_emoji": book.get("emoji", "📖"),
                "pages":      f"{start}–{end}",
                "section":    section or f"Pages {start}–{end}",
                "preview":    text[:350].replace("\n", " ").strip(),
            }, ensure_ascii=False)
            stream_callback(RETRIEVE_PREFIX + payload)

        return (
            f"=== {book['title']} | Pages {start}–{end} ===\n\n" + text
        )

    except ValueError:
        return f"Invalid page format '{pages}'. Use '5-10', '7', or '5,10,15'."
    except Exception as e:
        return f"Error reading pages: {e}"


def tool_search_section(book_id: str, query: str) -> str:
    structure = tool_get_structure(book_id)
    if "not yet indexed" in structure or "Unknown book_id" in structure:
        return structure

    resp = _groq(
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a document navigator. Given a book's structure and a query, "
                    "identify the MOST relevant section. "
                    'Return ONLY a JSON object: '
                    '{"section_title":"...","start_page":N,"end_page":N,"reason":"..."} '
                    "No markdown backticks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Book structure:\n{structure[:5000]}\n\n"
                    f"Query: {query}\n\n"
                    "Which section best answers this query? Return only JSON."
                ),
            },
        ],
        model=FAST_MODEL,
        max_tokens=200,
    )
    try:
        clean  = re.sub(r"```(?:json)?|```", "", resp).strip()
        parsed = json.loads(clean)
        title  = parsed.get("section_title", "Unknown")
        sp     = parsed.get("start_page", "?")
        ep     = parsed.get("end_page",   "?")
        reason = parsed.get("reason", "")
        return (
            f"Most relevant section: '{title}' (pages {sp}–{ep})\n"
            f"Reason: {reason}\n"
            f"→ Call get_pages('{book_id}', '{sp}-{ep}') to retrieve the content."
        )
    except Exception:
        return f"Section finder result:\n{resp}"


# ─────────────────────────────────────────────────────────────────────────────
# ReAct system prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are WineSommelier AI — an expert wine consultant powered by three authoritative wine textbooks:
1. "Chemistry and Biochemistry of Winemaking" (book_id: 'chemistry')
2. "Wine Tasting: A Professional Handbook" (book_id: 'tasting')
3. "Perceptions of Wine Quality" (book_id: 'perceptions')

═══════════════════ STRICT TOPIC BOUNDARIES ═══════════════════

You are STRICTLY LIMITED to answering questions about wine, winemaking, wine chemistry,
wine tasting, wine quality, wine perception, and closely related oenology topics that are
covered in your three reference books.

You MUST REFUSE to answer questions that are NOT related to wine or your knowledge base.
This includes but is not limited to:
- General knowledge, trivia, or factual questions unrelated to wine
- Programming, coding, math, science (unless directly about wine chemistry)
- Politics, history (unless directly about wine history), geography (unless about wine regions)
- Personal advice, recipes (unless wine pairing), health advice
- Any topic not covered by your three wine textbooks

When you receive an out-of-context question, respond EXACTLY like this (no tool calls):
"I'm sorry, but that question is outside my area of expertise. I am a Wine AI Sommelier
and I can only answer questions related to wine — including wine chemistry, winemaking,
tasting, quality assessment, and wine perception.

Here are some topics I can help you with:
• Wine chemistry (acidity, pH, sulphites, tannins, alcohol)
• Winemaking processes (fermentation, aging, stabilization)
• Professional wine tasting and sensory evaluation
• Wine quality factors and perception
• Food and wine pairing

Please ask me a wine-related question! 🍷"

═══════════════════ CORE RULES ═══════════════════

You MUST use tools to retrieve information from the books before answering.
Never make up information. Always base your answers on retrieved text.
Do NOT answer from general knowledge — ONLY use information retrieved from the books.

═══════════════════ TOOLS AVAILABLE ═══════════════════

Tool 1: list_books()
  → Lists all books with descriptions and status

Tool 2: get_structure(book_id)
  → Shows the hierarchical table of contents (section titles + page ranges + summaries)
  → Use this FIRST to understand what's in a book

Tool 3: get_pages(book_id, pages)
  → Retrieves actual text from specific pages
  → pages format: "45-52" or "10" or "10,15"
  → Max 20 pages per call
  → Use AFTER you've identified relevant sections from get_structure

Tool 4: search_section(book_id, query)
  → Quickly finds the most relevant section for a specific query
  → Returns the section title and page range
  → Use when you want to jump directly to relevant content

═══════════════════ HOW TO CALL TOOLS ═══════════════════

Output a tool call EXACTLY like this (pure JSON, no extra text):
<tool_call>
{"tool": "get_structure", "args": {"book_id": "chemistry"}}
</tool_call>

<tool_call>
{"tool": "get_pages", "args": {"book_id": "tasting", "pages": "45-52"}}
</tool_call>

<tool_call>
{"tool": "search_section", "args": {"book_id": "perceptions", "query": "sensory evaluation"}}
</tool_call>

<tool_call>
{"tool": "list_books", "args": {}}
</tool_call>

═══════════════════ WORKFLOW ═══════════════════

1. FIRST: Determine if the question is about wine. If NOT → refuse immediately (no tool calls).
2. Think: Which book(s) would cover this topic?
3. Call get_structure or search_section to find relevant sections
4. Call get_pages to read the relevant pages
5. Synthesize a comprehensive answer ONLY from retrieved text
6. ALWAYS cite sources: [Book Title, p. X-Y]

═══════════════════ ANSWERING ═══════════════════

When you have enough information to answer (no more tool calls needed):
- Write your answer in clear, engaging language
- Include specific details from the retrieved text
- End EVERY answer with citations in this format:
  📚 Sources: [Book Title, p. X-Y], [Book Title, p. X-Y]
- NEVER answer from general knowledge — only from book content
- If the retrieved pages do not contain relevant information, say so honestly
"""

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)


# ─────────────────────────────────────────────────────────────────────────────
# Tool dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, args: dict, stream_callback=None) -> str:
    if tool_name == "list_books":
        return tool_list_books()
    elif tool_name == "get_structure":
        return tool_get_structure(args.get("book_id", ""))
    elif tool_name == "get_pages":
        return tool_get_pages(
            args.get("book_id", ""),
            args.get("pages", "1-5"),
            stream_callback=stream_callback,
        )
    elif tool_name == "search_section":
        return tool_search_section(
            args.get("book_id", ""),
            args.get("query", ""),
        )
    else:
        return (
            f"Unknown tool: '{tool_name}'. "
            "Available: list_books, get_structure, get_pages, search_section"
        )


# ─────────────────────────────────────────────────────────────────────────────
# ReAct agent loop
# ─────────────────────────────────────────────────────────────────────────────

def answer(
    question: str,
    chat_history: list[dict] | None = None,
    stream_callback=None,
) -> tuple[str, list[dict]]:
    """
    Main entry point.  stream_callback receives:
      RETRIEVE::{json}  → render source card
      {other text}      → show as progress / thinking step
    """
    if chat_history is None:
        chat_history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += chat_history
    messages.append({"role": "user", "content": question})

    for iteration in range(MAX_ITERATIONS):
        if stream_callback:
            stream_callback(f"🔄 Thinking (step {iteration + 1})…")

        response = _groq(messages, model=ANSWER_MODEL, max_tokens=2000)
        tool_calls = TOOL_CALL_PATTERN.findall(response)

        if not tool_calls:
            messages.append({"role": "assistant", "content": response})
            return response, messages

        tool_results = []
        for tc_raw in tool_calls:
            try:
                tc     = json.loads(tc_raw.strip())
                t_name = tc.get("tool", "")
                t_args = tc.get("args", {})

                if stream_callback:
                    arg_str = ", ".join(str(v) for v in t_args.values())
                    stream_callback(f"🔧 Calling `{t_name}({arg_str})`…")

                result = _execute_tool(t_name, t_args,
                                       stream_callback=stream_callback)
                tool_results.append(
                    f"=== Tool: {t_name}({json.dumps(t_args)}) ===\n{result}\n"
                )
            except json.JSONDecodeError:
                tool_results.append(
                    f"[Invalid tool call JSON: {tc_raw[:100]}]"
                )
            except Exception as e:
                tool_results.append(f"[Tool error: {e}]")

        messages.append({"role": "assistant", "content": response})
        messages.append({
            "role":    "user",
            "content": (
                "TOOL RESULTS:\n\n"
                + "\n\n".join(tool_results)
                + "\n\nNow provide the final answer based on these results."
            ),
        })

    # Max iterations hit
    messages.append({
        "role":    "user",
        "content": "Please provide your final answer based on everything retrieved so far.",
    })
    final = _groq(messages, model=ANSWER_MODEL, max_tokens=1500)
    messages.append({"role": "assistant", "content": final})
    return final, messages


# ── CLI quick test ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🍷 Wine RAG Agent — CLI Test")
    print("=" * 50)
    print(tool_list_books())

    question = "What is the role of volatile acidity in wine quality?"
    print(f"\nQuestion: {question}\n")

    def cb(msg):
        if msg.startswith(RETRIEVE_PREFIX):
            data = json.loads(msg[len(RETRIEVE_PREFIX):])
            print(f"\n  📖 {data['book_emoji']}  {data['book_title']}")
            print(f"     Section : {data['section']}")
            print(f"     Pages   : p. {data['pages']}")
            print(f"     Preview : {data['preview'][:120]}…")
        else:
            print(f"  [{msg}]")

    final, _ = answer(question, stream_callback=cb)
    print(f"\n{'='*50}\nFinal Answer:\n{final}")