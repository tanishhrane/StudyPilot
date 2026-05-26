# ============================================================
# rag_summarizer.py — RAG-Powered Smart Summariser
# Replaces/enhances your existing summarizer.py
# ============================================================

from rag_engine import store_document, retrieve_relevant_chunks, call_llama_with_context
import re
from typing import List, Dict, Optional


# ============================================================
# SMART SUMMARY — Full Document Summary using RAG
# ============================================================

def summarize_document(
    user_id: str,
    doc_name: str,
    summary_type: str = "comprehensive",
    focus_topic: Optional[str] = None
) -> Dict:
    """
    Generates an intelligent summary of the uploaded document.

    WHY RAG FOR SUMMARISATION?
    Without RAG: You can only summarise text the user pastes (limited by context window)
    With RAG:    You can summarise a 200-page book by retrieving the key sections

    Two modes:
    1. Comprehensive: Retrieves broad chunks covering the whole document
    2. Topic-focused: Retrieves only chunks related to a specific topic
       e.g., "Summarise only the parts about Photosynthesis"

    Args:
        user_id:      Firebase user ID
        doc_name:     Which uploaded document to summarise
        summary_type: "comprehensive", "bullet_points", or "exam_focused"
        focus_topic:  Optional — if given, summarises only that topic from the doc

    Returns:
        Dict with: summary, key_points, source_document, word_count
    """

    # ── STEP 1: DECIDE WHAT TO RETRIEVE ──────────────────────────────────
    if focus_topic:
        # If user wants a topic-specific summary, search for that topic
        query = focus_topic
        print(f"🔍 Retrieving sections about '{focus_topic}' from '{doc_name}'...")
    else:
        # For full document summary, use a broad query to get varied chunks
        # We search for multiple general terms to get diverse coverage
        query = f"main concepts key ideas important topics {doc_name}"
        print(f"🔍 Retrieving key sections from '{doc_name}'...")

    # Retrieve top 8 chunks — more chunks = more complete summary
    chunks = retrieve_relevant_chunks(query=query, user_id=user_id, top_k=8)

    if not chunks:
        raise ValueError("No documents found. Please upload a PDF first.")

    # ── STEP 2: FILTER CHUNKS BY DOCUMENT ────────────────────────────────
    # If user uploaded multiple documents, only summarise the requested one
    # List comprehension: [item for item in list if condition]
    if doc_name != "all":
        # Filter to only keep chunks from the requested document
        chunks = [c for c in chunks if c["source"] == doc_name]

    if not chunks:
        raise ValueError(f"No content found for document '{doc_name}'")

    # ── STEP 3: BUILD CONTEXT ─────────────────────────────────────────────
    context_parts = [f"[Section {i+1}]\n{chunk['text']}" for i, chunk in enumerate(chunks)]
    context_text = "\n\n".join(context_parts)

    # ── STEP 4: CHOOSE PROMPT BASED ON SUMMARY TYPE ──────────────────────
    # Different prompts give very different outputs — this is "prompt engineering"

    system_prompt = """You are an expert academic summariser for StudyPilot, a study assistant app.
Your summaries help students understand and revise content efficiently.
Always be clear, concise, and use language a student would understand."""

    # Dictionary mapping summary_type → instructions
    # This is cleaner than multiple if/elif blocks
    summary_instructions = {
        "comprehensive": """
Write a comprehensive summary with:
1. OVERVIEW (2-3 sentences explaining the big picture)
2. KEY CONCEPTS (explain each major concept clearly)  
3. IMPORTANT DETAILS (facts, definitions, formulas mentioned)
4. CONNECTIONS (how the concepts relate to each other)
""",
        "bullet_points": """
Write a concise bullet-point summary with:
- Main topic in one sentence
- 8-10 bullet points covering the key takeaways
- Each bullet point should be one clear, memorable fact or concept
""",
        "exam_focused": """
Write an exam-preparation summary with:
1. LIKELY EXAM TOPICS (what topics from this material are commonly tested)
2. KEY DEFINITIONS (define all important terms)
3. FORMULAS/RULES (list any formulas or rules mentioned)
4. COMMON MISTAKES (what students typically get wrong about this topic)
5. MEMORY TIPS (mnemonics or tricks to remember key points)
"""
    }

    # .get() with a default — if summary_type isn't in the dict, use "comprehensive"
    instructions = summary_instructions.get(summary_type, summary_instructions["comprehensive"])

    focus_note = f"Focus specifically on: {focus_topic}" if focus_topic else ""

    user_prompt = f"""
STUDY MATERIAL TO SUMMARISE:
{context_text}

TASK: {focus_note}
{instructions}

Important: Only summarise what is in the provided material. Do not add outside knowledge.
"""

    # ── STEP 5: CALL LLAMA ────────────────────────────────────────────────
    print(f"🤖 Generating {summary_type} summary...")
    summary_text = call_llama_with_context(system_prompt, user_prompt)

    # ── STEP 6: EXTRACT KEY POINTS SEPARATELY ────────────────────────────
    # We make a second, smaller call to extract just the bullet-point key takeaways
    # This gives Streamlit something to display prominently as "Quick Takeaways"
    key_points = extract_key_points(context_text)

    # Count words in the original content vs summary — shows how much was compressed
    # .split() splits a string by whitespace into a list of words
    original_word_count = len(context_text.split())
    summary_word_count = len(summary_text.split())

    return {
        "summary": summary_text,
        "key_points": key_points,
        "source_document": doc_name,
        "focus_topic": focus_topic,
        "summary_type": summary_type,
        "original_word_count": original_word_count,
        "summary_word_count": summary_word_count,
        "compression_ratio": round(original_word_count / max(summary_word_count, 1), 1)
        # compression_ratio: if original=1000 words and summary=200 words → ratio=5x compression
    }


# ============================================================
# EXTRACT KEY POINTS (Quick Takeaways)
# ============================================================

def extract_key_points(context_text: str) -> List[str]:
    """
    Extracts 5 key bullet-point takeaways from the context.
    These are shown prominently in Streamlit as "Quick Takeaways".

    Args:
        context_text: The combined retrieved chunks

    Returns:
        List of 5 key point strings
    """
    system_prompt = "You extract key learning points from study material. Be concise and clear."

    user_prompt = f"""
From this study material, extract exactly 5 key points a student must remember.
Return ONLY a JSON array of 5 strings. No extra text. Example:
["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"]

Study material:
{context_text[:2000]}
"""
    # We limit to first 2000 chars for this sub-call to keep it fast
    # context_text[:2000] — Python slicing, takes first 2000 characters

    try:
        raw = call_llama_with_context(system_prompt, user_prompt)
        # Strip markdown code fences if present
        cleaned = re.sub(r'```json?\s*|\s*```', '', raw).strip()
        import json
        points = json.loads(cleaned)
        # isinstance() checks if points is actually a list (type safety)
        if isinstance(points, list):
            return points[:5]  # Return at most 5
    except Exception:
        pass  # If parsing fails, return a fallback message

    return ["Could not extract key points automatically. Please read the summary above."]


# ============================================================
# TOPIC-WISE SUMMARY (Advanced Feature)
# ============================================================

def summarize_by_topics(user_id: str, topics: List[str]) -> Dict[str, str]:
    """
    Generates individual summaries for a list of topics.
    Great for revision: user says "summarise: Photosynthesis, Cell Division, DNA"
    and gets a focused summary for each.

    Args:
        user_id: Firebase user ID
        topics:  List of topic strings

    Returns:
        Dict mapping topic name → its summary string
    """
    topic_summaries = {}

    for topic in topics:
        print(f"\n📖 Summarising: {topic}")
        try:
            result = summarize_document(
                user_id=user_id,
                doc_name="all",       # Search across all user's documents
                summary_type="bullet_points",
                focus_topic=topic
            )
            topic_summaries[topic] = result["summary"]
        except Exception as e:
            topic_summaries[topic] = f"Could not summarise this topic: {str(e)}"

    return topic_summaries
