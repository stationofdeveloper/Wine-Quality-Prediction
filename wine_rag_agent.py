"""
🤖 Wine RAG Agent — Agentic Vectorless Retrieval
=================================================
Inspired by PageIndex (https://github.com/VectifyAI/PageIndex)

How it works (NO vector DB, NO semantic similarity):
  The agent is given 4 tools and a ReAct (Reason + Act) loop:

  Tool 1: list_books()
      → Shows available books with titles, page counts, descriptions

  Tool 2: get_structure(book_id)
      → Returns the hierarchical tree index of a book
        (section titles, page ranges, summaries) — no actual text yet

  Tool 3: get_pages(book_id, pages)
      → Fetches actual page text (e.g., "45-52" or "10,15,20")
        Only called AFTER the agent reasons about which pages to read

  Tool 4: search_section(book_id, query)
      → LLM-powered section finder: given a query, reasons over the
        tree to identify the most relevant section + page range

  The agent loop:
    1. LLM sees the question
    2. LLM thinks: "I need to check the structure first"
    3. LLM calls get_structure → sees chapters/sections
    4. LLM thinks: "Chapter 3, pages 45-52 looks relevant"
    5. LLM calls get_pages(45-52) → gets actual text
    6. LLM synthesizes a grounded answer with citations

This is superior to vector RAG because:
  - LLM REASONS about relevance instead of numeric similarity
  - Returns exact page citations
  - No false positives from embedding similarity
  - Works on dense technical books with complex structure
"""

import os, json, re, time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv


from openai import OpenAI
import os

# openrouter_client = OpenAI(
#     base_url="https://openrouter.ai/api/v1",
#     api_key=os.getenv("OPENROUTER_API_KEY"),
# )
# openrouter_model = os.getenv("OPENROUTER_CHAT_MODEL", "google/gemma-4-26b-a4b-it:free")


load_dotenv()

GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY", "")
ANSWER_MODEL    = "llama-3.1-70b-versatile"   # best free Groq model for answering
FAST_MODEL      = "llama-3.1-8b-instant"       # for section finding
MAX_ITERATIONS  = 6                            # max ReAct loop iterations


# ── Import from indexer ────────────────────────────────────────────────────────
from wine_rag_indexer import (
    load_index, extract_pages_text, BOOK_REGISTRY, KNOWLEDGE_DIR
)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _groq(messages: list[dict], model: str = ANSWER_MODEL,max_tokens: int = 1500) -> str:
    """Call Groq Chat API."""
    if not GROQ_API_KEY:
        return _gemini_fallback(messages[-1]["content"], max_tokens)
    from groq import Groq, RateLimitError
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except RateLimitError:
            wait = 15 * (attempt + 1)
            time.sleep(wait)
        except Exception as e:
            if attempt == 2:
                return _gemini_fallback(messages[-1]["content"], max_tokens)
            time.sleep(3)
    return _gemini_fallback(messages[-1]["content"], max_tokens)


def _gemini_fallback(prompt: str, max_tokens: int = 1500) -> str:
    """Gemini Flash as fallback when Groq fails."""
    if not GEMINI_API_KEY:
        return "I need an API key to answer questions. Please check your .env file."
    try:
        import google.generativeai as genai    
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel("gemini-robotics-er-1.6-preview")
        resp  = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens}
        )
        return resp.text.strip()        
        # try:
        #     response = openrouter_client.chat.completions.create(
        #         model=openrouter_model,
        #         messages=[
        #             {"role": "user", "content": prompt} # Make sure the variable 'prompt' matches what your function uses!
        #         ]
        #     )
        #     answer = response.choices[0].message.content
        #     return answer
        # except Exception as e:
        #     return f"OpenRouter API Error: {str(e)}"
    except Exception as e:
        return f"Error: {e}"


# ── Agent Tools ────────────────────────────────────────────────────────────────

def tool_list_books() -> str:
    """Tool: list all available books with status."""
    lines = ["Available wine books:\n"]
    for book_id, book in BOOK_REGISTRY.items():
        tree = load_index(book_id)
        pdf_path = KNOWLEDGE_DIR / book["filename"]
        status = "✅ indexed" if tree else ("📄 not indexed" if pdf_path.exists() else "❌ PDF missing")
        desc   = tree.get("description", "")[:120] + "…" if tree else ""
        lines.append(
            f"• book_id: '{book_id}'\n"
            f"  Title: {book['title']}\n"
            f"  Pages: {book['pages']}\n"
            f"  Status: {status}\n"
            + (f"  About: {desc}\n" if desc else "")
        )
    return "\n".join(lines)


def tool_get_structure(book_id: str) -> str:
    """Tool: return the tree index of a book (no page text, just structure)."""
    tree = load_index(book_id)
    if not tree:
        book = BOOK_REGISTRY.get(book_id)
        if not book:
            return f"Unknown book_id '{book_id}'. Use list_books() to see valid IDs."
        return (
            f"Book '{book['title']}' is not yet indexed.\n"
            f"Run: python wine_rag_indexer.py {book_id}\n"
            f"Or index from the Streamlit UI."
        )

    def format_node(node: dict, indent: int = 0) -> str:
        pad     = "  " * indent
        title   = node.get("title", "Untitled")
        s_page  = node.get("start_page", "?")
        e_page  = node.get("end_page",   "?")
        summary = node.get("summary", "")[:120]
        line    = f"{pad}[p.{s_page}-{e_page}] {title}"
        if summary:
            line += f"\n{pad}   ↳ {summary}"
        children = "\n".join(format_node(c, indent+1) for c in node.get("nodes", []))
        return line + ("\n" + children if children else "")

    book_info = (
        f"Book: {tree['title']}\n"
        f"Total pages: {tree['total_pages']}\n"
        f"Description: {tree.get('description', '')}\n\n"
        f"Structure:\n"
    )
    structure = "\n".join(format_node(n) for n in tree.get("nodes", []))
    result    = book_info + structure

    # Limit to avoid context overflow
    if len(result) > 8000:
        result = result[:8000] + "\n\n[… structure truncated. Use search_section to find specific sections.]"
    return result


def tool_get_pages(book_id: str, pages: str) -> str:
    """
    Tool: get actual text from specific pages.
    pages format: "5-10" (range) or "5,10,15" (individual) or "7" (single)
    """
    book = BOOK_REGISTRY.get(book_id)
    if not book:
        return f"Unknown book_id '{book_id}'."

    pdf_path = KNOWLEDGE_DIR / book["filename"]
    if not pdf_path.exists():
        return f"PDF file not found: {book['filename']}. Place it in wine_knowledge/ folder."

    # Parse page specification
    try:
        if "-" in pages and "," not in pages:
            parts = pages.split("-")
            start = int(parts[0].strip())
            end   = int(parts[1].strip())
        elif "," in pages:
            nums  = [int(x.strip()) for x in pages.split(",")]
            start = min(nums)
            end   = max(nums)
        else:
            start = int(pages.strip())
            end   = start

        # Safety cap
        tree = load_index(book_id)
        if tree:
            max_page = tree.get("total_pages", book["pages"])
            end      = min(end, max_page)

        if end - start > 20:
            return (
                f"⚠️ Requesting {end-start+1} pages is too many (max 20 at once). "
                f"Narrow your page range for better performance."
            )

        text = extract_pages_text(pdf_path, start, end, max_chars=7000)
        return (
            f"=== {book['title']} | Pages {start}–{end} ===\n\n"
            + (text if text.strip() else "[No text found on these pages]")
        )

    except ValueError:
        return f"Invalid page format '{pages}'. Use '5-10', '7', or '5,10,15'."
    except Exception as e:
        return f"Error reading pages: {e}"


def tool_search_section(book_id: str, query: str) -> str:
    """
    Tool: LLM-powered section finder.
    Reasons over the tree structure to find the most relevant section + page range.
    Returns page range recommendation (not the actual text — call get_pages after).
    """
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
                    "Return ONLY a JSON object: "
                    '{"section_title": "...", "start_page": N, "end_page": N, "reason": "..."}'
                    " No markdown backticks."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Book structure:\n{structure[:5000]}\n\n"
                    f"Query: {query}\n\n"
                    "Which section best answers this query? Return only JSON."
                )
            }
        ],
        model=FAST_MODEL,
        max_tokens=200,
    )

    try:
        clean  = re.sub(r"```(?:json)?|```", "", resp).strip()
        parsed = json.loads(clean)
        title  = parsed.get("section_title", "Unknown")
        sp     = parsed.get("start_page", "?")
        ep     = parsed.get("end_page", "?")
        reason = parsed.get("reason", "")
        return (
            f"Most relevant section: '{title}' (pages {sp}–{ep})\n"
            f"Reason: {reason}\n"
            f"→ Call get_pages('{book_id}', '{sp}-{ep}') to retrieve the content."
        )
    except Exception:
        return f"Section finder result:\n{resp}"


# ── ReAct Agent System Prompt ─────────────────────────────────────────────────

SYSTEM_PROMPT = """You are WineSommelier AI — an expert wine consultant powered by three authoritative wine textbooks:
1. "Chemistry and Biochemistry of Winemaking" (book_id: 'chemistry')
2. "Wine Tasting: A Professional Handbook" (book_id: 'tasting')
3. "Perceptions of Wine Quality" (book_id: 'perceptions')

You MUST use tools to retrieve information from the books before answering.
Never make up information. Always base your answers on retrieved text.

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

1. Think: Which book(s) would cover this topic?
2. Call get_structure or search_section to find relevant sections
3. Call get_pages to read the relevant pages
4. Synthesize a comprehensive answer
5. ALWAYS cite sources: [Book Title, p. X-Y]

═══════════════════ ANSWERING ═══════════════════

When you have enough information to answer (no more tool calls needed):
- Write your answer in clear, engaging language
- Include specific details from the retrieved text
- End EVERY answer with citations in this format:
  📚 Sources: [Book Title, p. X-Y], [Book Title, p. X-Y]
- If the user asks a simple/casual question not requiring book lookup, answer directly
"""

TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL
)


# ── ReAct Agent Loop ───────────────────────────────────────────────────────────

def _execute_tool(tool_name: str, args: dict) -> str:
    """Dispatch tool call to the appropriate function."""
    if tool_name == "list_books":
        return tool_list_books()
    elif tool_name == "get_structure":
        return tool_get_structure(args.get("book_id", ""))
    elif tool_name == "get_pages":
        return tool_get_pages(args.get("book_id", ""), args.get("pages", "1-5"))
    elif tool_name == "search_section":
        return tool_search_section(args.get("book_id", ""),
                                   args.get("query", ""))
    else:
        return f"Unknown tool: '{tool_name}'. Available: list_books, get_structure, get_pages, search_section"


def answer(question: str,
           chat_history: list[dict] | None = None,
           stream_callback=None) -> tuple[str, list[dict]]:
    """
    Main entry point for agentic retrieval.

    Args:
        question:        User's question
        chat_history:    Previous [{role, content}] messages for multi-turn
        stream_callback: Optional callback(text: str) for streaming updates

    Returns:
        (final_answer: str, updated_messages: list[dict])
    """
    if chat_history is None:
        chat_history = []

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += chat_history
    messages.append({"role": "user", "content": question})

    tool_log  = []   # track all tool calls for transparency

    for iteration in range(MAX_ITERATIONS):
        if stream_callback:
            stream_callback(f"🔄 Thinking (step {iteration+1})…")

        response = _groq(messages, model=ANSWER_MODEL, max_tokens=2000)

        # Check for tool calls
        tool_calls = TOOL_CALL_PATTERN.findall(response)

        if not tool_calls:
            # No tool calls → this is the final answer
            messages.append({"role": "assistant", "content": response})
            return response, messages

        # Execute tool calls
        tool_results = []
        for tc_raw in tool_calls:
            try:
                tc      = json.loads(tc_raw.strip())
                t_name  = tc.get("tool", "")
                t_args  = tc.get("args", {})

                if stream_callback:
                    stream_callback(f"🔧 Calling `{t_name}({', '.join(str(v) for v in t_args.values())})`…")

                result  = _execute_tool(t_name, t_args)
                tool_log.append({"tool": t_name, "args": t_args, "result_len": len(result)})

                tool_results.append(
                    f"=== Tool: {t_name}({json.dumps(t_args)}) ===\n{result}\n"
                )
            except json.JSONDecodeError:
                tool_results.append(f"[Invalid tool call JSON: {tc_raw[:100]}]")
            except Exception as e:
                tool_results.append(f"[Tool error: {e}]")

        # Append assistant's thinking + tool results to messages
        messages.append({"role": "assistant", "content": response})
        tool_result_text = "\n\n".join(tool_results)
        messages.append({
            "role":    "user",
            "content": f"TOOL RESULTS:\n\n{tool_result_text}\n\nNow provide the final answer based on these results."
        })

    # Max iterations reached — ask for final answer
    messages.append({
        "role":    "user",
        "content": "Please provide your final answer based on everything retrieved so far."
    })
    final = _groq(messages, model=ANSWER_MODEL, max_tokens=1500)
    messages.append({"role": "assistant", "content": final})
    return final, messages


# ── Quick CLI test ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("🍷 Wine RAG Agent — CLI Test")
    print("=" * 50)
    print(tool_list_books())
    print("\n" + "=" * 50)

    question = "What is the role of volatile acidity in wine quality?"
    print(f"Question: {question}\n")

    def cb(msg): print(f"  [{msg}]")

    final, _ = answer(question, stream_callback=cb)
    print(f"\n{'='*50}\nFinal Answer:\n{final}")