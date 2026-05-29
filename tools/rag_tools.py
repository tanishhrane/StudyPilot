# ============================================================
# tools/rag_tools.py — RAG-powered versions of existing tools
#
# This file adds RAG capability to your 3 existing tools.
# Your original tools/quiz.py, tools/summarizer.py, tools/study_plan.py
# are NOT modified at all. They continue to work exactly as before.
#
# This file is only imported when the user is in RAG mode
# (i.e., they have uploaded documents AND want to use them).
# ============================================================

import json
import re
from typing import List, Dict, Optional

from llm import call_llm
from rag_engine import retrieve_relevant_chunks


# ============================================================
# HELPER: BUILD CONTEXT STRING
# ============================================================

def _build_context(chunks: List[Dict]) -> str:
    """
    Takes a list of retrieved chunk dicts and formats them into
    a readable context block to inject into LLM prompts.

    Each chunk looks like: {"text": "...", "source": "Physics Notes", "score": 0.87}

    We label each excerpt so the LLM knows where it came from,
    which also enables it to cite sources in its response.
    """
    if not chunks:
        return "No relevant content found in uploaded documents."

    parts = []
    for i, chunk in enumerate(chunks):
        parts.append(
            f"[Excerpt {i + 1} — from '{chunk['source']}']\n{chunk['text']}"
        )
    return "\n\n".join(parts)


# ============================================================
# RAG QUIZ GENERATOR
# ============================================================

def generate_quiz_from_docs(
    topic: str,
    user_id: str,
    num_questions: int = 5,
    difficulty: str = "medium",
    weak_topics: Optional[str] = None
) -> Dict:
    """
    Generates a quiz grounded in the user's own uploaded notes.

    Flow:
    1. Retrieve the top-5 most relevant chunks for the topic
    2. Inject chunks as context into the quiz generation prompt
    3. Llama generates questions ONLY from that context
    4. Parse and return the JSON quiz

    Args:
        topic:         Topic to quiz on (e.g. "photosynthesis")
        user_id:       Firebase UID to fetch correct user's docs
        num_questions: How many questions
        difficulty:    "easy", "medium", or "hard"
        weak_topics:   Comma-separated weak areas from memory (from agent.py)

    Returns:
        Quiz dict matching your existing format:
        {"topic": "", "questions": [{id, question, options, correct_answer, subtopic}]}
    """
    chunks = retrieve_relevant_chunks(query=topic, user_id=user_id, top_k=5)
    context = _build_context(chunks)

    weak_instruction = ""
    if weak_topics:
        weak_instruction = f"""
IMPORTANT: The student has previously struggled with: {weak_topics}.
Prioritize questions covering these weak areas and make them slightly harder.
"""

    system_prompt = """
You are an expert academic quiz generator for StudyPilot.

You MUST return ONLY valid JSON. No markdown. No explanation. No extra text.

STRICT RULES:
- Generate questions ONLY from the provided study material excerpts.
- Do NOT use any knowledge outside of what is given below.
- If the topic is not covered in the excerpts, still generate the best questions you can from what IS there.
- Every question must have exactly 4 options: A, B, C, D.
- Only one option is correct.
- The "subtopic" field must name the specific concept the question tests.
- Avoid vague, repetitive, or trivially easy questions.
- Prefer conceptual understanding over pure memorization.

Return format (EXACTLY this structure):
{
    "topic": "",
    "questions": [
        {
            "id": 1,
            "question": "",
            "options": {
                "A": "",
                "B": "",
                "C": "",
                "D": ""
            },
            "correct_answer": "",
            "subtopic": ""
        }
    ]
}
"""

    user_prompt = f"""
STUDY MATERIAL (generate questions ONLY from this):
{context}

TASK:
Generate {num_questions} {difficulty}-difficulty multiple choice questions about: "{topic}"

{weak_instruction}

Return ONLY valid JSON matching the specified format.
"""

    response = call_llm(system_prompt, user_prompt)

    # Strip markdown fences if model adds them (e.g. ```json ... ```)
    cleaned = re.sub(r'```json?\s*|\s*```', '', response).strip()

    try:
        quiz_data = json.loads(cleaned)
        # Attach source info to each question for the evaluator to use later
        for i, q in enumerate(quiz_data.get("questions", [])):
            q["source"] = chunks[i % len(chunks)]["source"] if chunks else "your notes"
        return quiz_data
    except json.JSONDecodeError:
        return {"error": "Failed to generate valid quiz from your documents. Please try again."}


# ============================================================
# RAG EVALUATOR
# ============================================================

def evaluate_quiz_with_rag(
    quiz_data: Dict,
    user_answers: Dict,
    user_id: str
) -> Dict:
    """
    Evaluates quiz answers and — for wrong answers — retrieves the exact
    passage from the student's notes that explains the correct concept.

    This is the feature that makes your evaluator stand out:
    Instead of "Wrong. Correct answer: B", it says:
    "Wrong. Here's the section from YOUR notes that covers this..."

    Args:
        quiz_data:    The quiz dict from generate_quiz_from_docs()
        user_answers: {question_id (int): chosen_letter (str)}
        user_id:      Firebase UID

    Returns:
        {score, total, percentage, weak_topics, results, performance_label}
        where each result item optionally has a "revision_passage" key
    """
    score       = 0
    total       = len(quiz_data["questions"])
    weak_topics = []
    results     = []

    for question in quiz_data["questions"]:
        qid            = question["id"]
        correct_answer = question["correct_answer"]
        user_answer    = user_answers.get(qid)
        is_correct     = user_answer == correct_answer

        if is_correct:
            score += 1
            revision_passage = None
        else:
            # ── RAG RETRIEVAL FOR REVISION HINT ────────────────
            # For each wrong answer, search the user's notes for
            # the specific concept this question tested.
            subtopic = question.get("subtopic", question["question"])
            chunks   = retrieve_relevant_chunks(
                query=subtopic, user_id=user_id, top_k=1
            )
            if chunks:
                best = chunks[0]
                revision_passage = {
                    "text":   best["text"][:400],
                    "source": best["source"],
                    "score":  round(best["score"], 2)
                }
            else:
                revision_passage = None

            weak_topics.append(subtopic)

        results.append({
            "question_id":      qid,
            "question":         question["question"],
            "user_answer":      user_answer,
            "correct_answer":   correct_answer,
            "is_correct":       is_correct,
            "revision_passage": revision_passage  # None for correct answers
        })

    percentage = round((score / total) * 100, 1) if total > 0 else 0

    return {
        "score":             score,
        "total":             total,
        "percentage":        percentage,
        "weak_topics":       list(set(weak_topics)),
        "results":           results,
        "performance_label": _performance_label(percentage)
    }


def _performance_label(pct: float) -> str:
    if pct >= 90: return "Excellent! 🌟"
    if pct >= 70: return "Good Job! 👍"
    if pct >= 50: return "Needs Improvement 📚"
    return "Revise Thoroughly 🔄"


# ============================================================
# RAG SUMMARIZER
# ============================================================

def summarize_document_rag(
    user_id: str,
    doc_name: str,
    summary_type: str = "comprehensive",
    focus_topic: Optional[str] = None
) -> Dict:
    """
    Summarises an uploaded document using RAG.

    Why this beats your existing summarizer:
    - Existing summarizer.py takes text the user pastes (limited by what fits in a prompt)
    - This version handles any size document by retrieving the most relevant sections
    - A 200-page PDF gets compressed to a clean summary automatically

    Args:
        user_id:      Firebase UID
        doc_name:     "all" or a specific document name
        summary_type: "comprehensive", "bullet_points", or "exam_focused"
        focus_topic:  Optional specific topic within the doc to focus on

    Returns:
        {summary, key_points, original_word_count, summary_word_count, compression_ratio}
    """
    query = focus_topic if focus_topic else f"main concepts key ideas {doc_name}"
    chunks = retrieve_relevant_chunks(query=query, user_id=user_id, top_k=8)

    if not chunks:
        raise ValueError("No documents found. Please upload a study material file first.")

    # If a specific doc was requested, filter to only that doc
    if doc_name != "all":
        chunks = [c for c in chunks if c["source"] == doc_name]
        if not chunks:
            raise ValueError(f"No content found for '{doc_name}'. Try selecting 'All documents'.")

    context = _build_context(chunks)

    instructions = {
        "comprehensive": """
Write a comprehensive summary with:
1. OVERVIEW (2-3 sentences: what is this material about?)
2. KEY CONCEPTS (explain each major concept clearly in plain language)
3. IMPORTANT DETAILS (definitions, facts, formulas found in the material)
4. CONNECTIONS (how do the concepts relate to each other?)
""",
        "bullet_points": """
Write a concise bullet-point summary:
- One sentence overview
- 8-10 bullet points: each one a clear, memorable key fact or concept
- Keep each bullet under 20 words
""",
        "exam_focused": """
Write an exam preparation summary:
1. LIKELY EXAM TOPICS (what from this material is commonly tested?)
2. KEY DEFINITIONS (define every important term found in the material)
3. FORMULAS & RULES (list any formulas, laws, or rules mentioned)
4. COMMON MISTAKES (what do students typically get wrong here?)
5. QUICK MEMORY AIDS (mnemonics or tricks for key points)
"""
    }.get(summary_type, "Write a clear, structured summary.")

    focus_note = f"Focus specifically on: {focus_topic}\n" if focus_topic else ""

    system_prompt = """
You are an expert academic summariser for StudyPilot.

STRICT RULES:
- Summarise ONLY the content provided in the study material excerpts below.
- Do NOT add information from outside the provided excerpts.
- If something is not in the material, do not include it.
- Write in clear, student-friendly language.
- Be structured and use proper headings/bullets as instructed.
"""

    user_prompt = f"""
STUDY MATERIAL:
{context}

{focus_note}TASK:
{instructions}
"""

    summary = call_llm(system_prompt, user_prompt)

    # Extract 5 key takeaways in a separate smaller call
    key_points = _extract_key_points(context)

    original_words = len(context.split())
    summary_words  = len(summary.split())

    return {
        "summary":            summary,
        "key_points":         key_points,
        "source":             doc_name,
        "focus_topic":        focus_topic,
        "summary_type":       summary_type,
        "original_word_count": original_words,
        "summary_word_count":  summary_words,
        "compression_ratio":   round(original_words / max(summary_words, 1), 1)
    }


def _extract_key_points(context: str) -> List[str]:
    """
    Extracts 5 bullet-point takeaways using a small focused LLM call.
    Returns a Python list of strings.
    """
    system_prompt = "Extract key learning points. Return ONLY a JSON array of 5 strings. No other text."
    user_prompt   = f"""
From this study material, extract exactly 5 key points a student must remember.
Return ONLY a JSON array: ["Point 1", "Point 2", "Point 3", "Point 4", "Point 5"]

Material:
{context[:1500]}
"""
    try:
        raw     = call_llm(system_prompt, user_prompt)
        cleaned = re.sub(r'```json?\s*|\s*```', '', raw).strip()
        points  = json.loads(cleaned)
        if isinstance(points, list):
            return points[:5]
    except Exception:
        pass
    return ["See the full summary above for key points."]


# ============================================================
# RAG STUDY PLAN GENERATOR
# ============================================================

def create_study_plan_from_docs(
    topic: str,
    user_id: str,
    days: int,
    start_date: str,
    time_slot: str
) -> Dict:
    """
    Generates a study plan tailored to the user's actual uploaded material.

    Your existing create_study_plan() generates a generic plan.
    This version retrieves what topics are actually IN the user's notes
    and builds the plan around that content specifically.

    Returns the same dict structure as create_study_plan() so the rest
    of agent.py and app.py works without any changes.
    """
    # Retrieve what content exists in the uploaded docs for this topic
    chunks  = retrieve_relevant_chunks(query=topic, user_id=user_id, top_k=6)
    context = _build_context(chunks)

    system_prompt = """
You are an expert academic planning assistant for StudyPilot.

You MUST return ONLY valid JSON.
Do NOT include explanations, markdown, or extra text.

STRICT RULES:
- Build the study plan around the specific content found in the uploaded study material.
- Reference actual concepts and topics from the material in each day's tasks.
- Do NOT invent topics not covered in the material.
- Make tasks specific, actionable, and practical.
- Avoid vague tasks like "Review concepts" or "Study the topic".
- Each day should cover distinct sub-topics.
- Progress from foundational → intermediate → advanced over the days.

Return format:
{
  "plan_title": "",
  "days": [
    {
      "day_number": 1,
      "start_time": "09:00 AM",
      "end_time": "12:00 PM",
      "topic": "",
      "tasks": [
        {
          "task": "",
          "description": ""
        }
      ]
    }
  ]
}
"""

    user_prompt = f"""
UPLOADED STUDY MATERIAL (base the plan on THIS content):
{context}

TASK:
Create a {days}-day study plan for: "{topic}"
Start date: {start_date}
Daily study time: {time_slot}

Requirements:
- Each day should cover specific concepts found in the material above
- Include practical exercises based on the material
- Progress logically from basics to advanced
- Include revision days for complex topics
- Return ONLY valid JSON
"""

    response = call_llm(system_prompt, user_prompt)

    # Now use the same formatting logic as your original create_study_plan()
    # by importing and delegating — so formatting is always consistent
    from tools.study_plan import create_study_plan as _original_plan

    # We call the LLM ourselves above, but to reuse the date-handling and
    # formatting logic in study_plan.py, we just pass it the same args.
    # The difference is we've already retrieved context; the original function
    # will call the LLM again but this time with topic enriched by our context.
    # To avoid a double LLM call, we parse and format the response ourselves:
    import json
    from datetime import datetime, timedelta

    cleaned = re.sub(r'```json?\s*|\s*```', '', response).strip()

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback to original function if our RAG call returns bad JSON
        return _original_plan(topic, days, start_date, time_slot)

    # ── DATE HANDLING (same logic as study_plan.py) ────────────
    today = datetime.today()
    sd    = start_date.lower()

    if sd == "today":
        base_date = today
    elif sd == "tomorrow":
        base_date = today + timedelta(days=1)
    elif sd in ["monday","tuesday","wednesday","thursday","friday","saturday","sunday"]:
        weekdays  = {"monday":0,"tuesday":1,"wednesday":2,"thursday":3,
                     "friday":4,"saturday":5,"sunday":6}
        days_ahead = (weekdays[sd] - today.weekday()) % 7 or 7
        base_date  = today + timedelta(days=days_ahead)
    else:
        try:
            base_date = datetime.strptime(start_date, "%Y-%m-%d")
        except ValueError:
            base_date = today

    # ── FORMAT OUTPUT (same as study_plan.py) ─────────────────
    formatted = f"📘 {parsed.get('plan_title', topic + ' Study Plan')}\n\n"
    formatted += f"📌 *Based on your uploaded study material*\n\n"

    for i, day in enumerate(parsed.get("days", [])):
        current_date = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        day["date"]  = current_date

        formatted += f"📅 Day {day.get('day_number', i+1)} ({current_date})\n"
        formatted += f"⏰ {day.get('start_time','09:00 AM')} - {day.get('end_time','12:00 PM')}\n"
        formatted += f"📖 Topic: {day.get('topic','Study Session')}\n"
        formatted += "Tasks:\n"

        for task in day.get("tasks", []):
            if isinstance(task, dict):
                formatted += f"- {task.get('task','')}\n"
                if task.get("description"):
                    formatted += f"   • {task['description']}\n"
            else:
                formatted += f"- {task}\n"
        formatted += "\n"

    return {
        "formatted_output": formatted,
        "plan_json":        parsed
    }
