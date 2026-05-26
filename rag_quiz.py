# ============================================================
# rag_quiz.py — RAG-Powered Quiz Generator
# Plug this into your existing quiz.py logic
# ============================================================

# Import the RAG engine functions we wrote in rag_engine.py
# These handle all the heavy lifting: PDF → chunks → embeddings → retrieval
from rag_engine import store_document, retrieve_relevant_chunks, call_llama_with_context, list_user_documents, delete_user_document

# json: built-in Python module to parse JSON strings into Python dicts/lists
# Why: Llama returns quiz questions as a JSON string, we need to parse it
import json

# re: built-in regex module
# Why: Sometimes Llama wraps JSON in ```json ... ``` markdown, we strip that
import re

from typing import List, Dict


# ============================================================
# QUIZ GENERATION WITH RAG
# ============================================================

def generate_quiz_from_document(
    topic: str,
    user_id: str,
    num_questions: int = 5,
    difficulty: str = "medium"
) -> List[Dict]:
    """
    Generates quiz questions sourced directly from the user's uploaded PDF.

    FLOW:
    1. retrieve_relevant_chunks() — finds PDF sections related to the topic
    2. Build a prompt that includes those chunks as context
    3. call_llama_with_context() — Llama reads the context and writes questions
    4. Parse the JSON response into a Python list of question dicts

    Args:
        topic:         What topic to quiz on (e.g., "photosynthesis", "Newton's laws")
        user_id:       Firebase user ID (to fetch the right user's documents)
        num_questions: How many questions to generate
        difficulty:    "easy", "medium", or "hard"

    Returns:
        List of dicts, each with keys: question, options (A/B/C/D), answer, explanation, source
    """

    # ── STEP 1: RETRIEVE relevant chunks from the user's uploaded notes ──
    print(f"🔍 Searching your notes for: '{topic}'")
    chunks = retrieve_relevant_chunks(query=topic, user_id=user_id, top_k=5)

    if not chunks:
        raise ValueError(
            "No documents found! Please upload your study material PDF first."
        )

    # ── STEP 2: BUILD CONTEXT STRING ────────────────────────────────────
    # Join the retrieved chunks into one readable block
    # We also track which sources (documents) were used
    context_parts = []
    sources_used = set()

    for i, chunk in enumerate(chunks):
        context_parts.append(f"[Excerpt {i+1} from '{chunk['source']}']\n{chunk['text']}")
        sources_used.add(chunk['source'])

    # "\n\n".join() — joins list items with double newline between them
    context_text = "\n\n".join(context_parts)

    # ── STEP 3: BUILD THE PROMPT ─────────────────────────────────────────
    # This is the most important part of RAG — how you structure the prompt
    # We explicitly tell Llama: use ONLY the provided context, not your training data

    system_prompt = """You are an expert quiz generator for a student study assistant called StudyPilot.
Your job is to create multiple choice questions STRICTLY based on the provided study material.
Do NOT use any knowledge outside of what is given in the context.
Always respond with valid JSON only — no extra text, no markdown formatting."""

    user_prompt = f"""
STUDY MATERIAL (use ONLY this to create questions):
{context_text}

TASK:
Generate {num_questions} {difficulty}-difficulty multiple choice questions about: "{topic}"

Rules:
- Every question MUST be answerable from the study material above
- Each question has exactly 4 options: A, B, C, D
- Only one option is correct
- Include a brief explanation citing which excerpt supports the answer
- Do not repeat similar questions

Respond in this EXACT JSON format (no extra text):
[
  {{
    "question": "What is ...?",
    "options": {{
      "A": "First option",
      "B": "Second option", 
      "C": "Third option",
      "D": "Fourth option"
    }},
    "answer": "A",
    "explanation": "According to the study material, ...",
    "topic": "{topic}",
    "source": "name of the document this came from"
  }}
]
"""

    # ── STEP 4: CALL LLAMA ────────────────────────────────────────────────
    print(f"🤖 Generating {num_questions} questions using your notes...")
    raw_response = call_llama_with_context(system_prompt, user_prompt)

    # ── STEP 5: PARSE THE JSON RESPONSE ──────────────────────────────────
    # re.sub() — removes ```json and ``` markdown code fences if Llama adds them
    # The pattern r'```json?\s*|\s*```' matches both ```json and ``` with any spaces
    cleaned = re.sub(r'```json?\s*|\s*```', '', raw_response).strip()

    try:
        questions = json.loads(cleaned)  # json.loads() converts JSON string → Python list
        print(f"✅ Generated {len(questions)} questions from your notes!")
        return questions
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Raw response was: {raw_response}")
        raise ValueError("Llama returned invalid JSON. Try again or rephrase the topic.")


# ============================================================
# EVALUATOR WITH RAG (Your existing evaluator, but smarter)
# ============================================================

def evaluate_quiz_with_rag(
    questions: List[Dict],
    user_answers: Dict[str, str],
    user_id: str
) -> Dict:
    """
    Evaluates the user's answers AND retrieves the exact passage from their
    notes to explain WHY they got something wrong.

    This is what makes your evaluator impressive:
    Instead of a generic "Wrong! The answer is B", it says:
    "Wrong. Here's the exact section from your notes that explains this: ..."

    Args:
        questions:    The list of question dicts from generate_quiz_from_document()
        user_answers: Dict mapping question index → user's chosen option
                      e.g., {"0": "A", "1": "C", "2": "B"}
        user_id:      Firebase user ID

    Returns:
        Dict with: score, total, percentage, weak_topics, detailed_feedback
    """

    score = 0
    weak_topics = []
    detailed_feedback = []

    for i, question in enumerate(questions):
        question_index = str(i)
        user_answer = user_answers.get(question_index, "").upper()
        correct_answer = question["answer"].upper()
        is_correct = user_answer == correct_answer

        if is_correct:
            score += 1
            feedback_entry = {
                "question": question["question"],
                "your_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": True,
                "explanation": question.get("explanation", ""),
                "revision_passage": None  # No need to retrieve passage for correct answers
            }
        else:
            # ── RAG MAGIC HAPPENS HERE ───────────────────────────────────
            # For each wrong answer, retrieve the specific passage from
            # the user's notes that covers this topic
            # This is what sets your evaluator apart!

            topic = question.get("topic", question["question"])
            relevant_chunks = retrieve_relevant_chunks(
                query=topic,
                user_id=user_id,
                top_k=2  # Just top 2 passages for focused revision
            )

            # Pick the most relevant passage to show the student
            revision_passage = None
            if relevant_chunks:
                # The first result is the most similar (ChromaDB returns sorted by similarity)
                best_chunk = relevant_chunks[0]
                revision_passage = {
                    "text": best_chunk["text"],
                    "source": best_chunk["source"],
                    "relevance_score": round(best_chunk["score"], 2)
                }

            # Track this as a weak topic (your existing evaluator logic)
            if topic not in weak_topics:
                weak_topics.append(topic)

            feedback_entry = {
                "question": question["question"],
                "your_answer": user_answer,
                "correct_answer": correct_answer,
                "is_correct": False,
                "explanation": question.get("explanation", ""),
                "revision_passage": revision_passage  # ← The RAG-powered revision hint
            }

        detailed_feedback.append(feedback_entry)

    # ── BUILD FINAL RESULT ────────────────────────────────────────────────
    total = len(questions)
    percentage = round((score / total) * 100, 1) if total > 0 else 0

    return {
        "score": score,
        "total": total,
        "percentage": percentage,
        "weak_topics": weak_topics,           # Your existing feature, unchanged
        "detailed_feedback": detailed_feedback, # Enhanced with revision passages
        "performance_label": _get_performance_label(percentage)
    }


def _get_performance_label(percentage: float) -> str:
    """Helper to give a friendly performance label based on score."""
    if percentage >= 90:
        return "Excellent! 🌟"
    elif percentage >= 70:
        return "Good Job! 👍"
    elif percentage >= 50:
        return "Needs Improvement 📚"
    else:
        return "Revise Thoroughly 🔄"
