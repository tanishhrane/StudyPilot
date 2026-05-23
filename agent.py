import json

from llm import call_llm

from tools.study_plan import create_study_plan
from tools.quiz import generate_quiz, evaluate_quiz
from tools.summarizer import summarize_text

from config import DEBUG


# ==========================================
# Helper Function
# ==========================================

def format_history(history):

    conversation = ""

    for msg in history:

        role = msg["role"]

        content = msg["content"]

        conversation += f"{role.capitalize()}: {content}\n"

    return conversation


# ==========================================
# Main Agent Function
# ==========================================

def run_agent(user_input, history):

    system_prompt = """
You are StudyPilot, an intelligent academic routing agent.

You MUST return ONLY valid JSON.
Do NOT include explanations.
Do NOT include markdown.
Do NOT include extra text.

Available tools:

1. create_study_plan
   Arguments:
   - topic (string)
   - days (integer)

2. generate_quiz
   Arguments:
   - topic (string)
   - num_questions (integer)

3. summarize_text
   Arguments:
   - text (string)

Return format:

{
    "tool_name": "",
    "arguments": {}
}
"""

    # ==========================================
    # Format Conversation History
    # ==========================================

    context = format_history(history)

    # ==========================================
    # Create Enriched Input
    # ==========================================

    enriched_input = f"""
Conversation so far:
{context}

Current user request:
{user_input}
"""

    # ==========================================
    # LLM Tool Decision
    # ==========================================

    decision = call_llm(
        system_prompt,
        enriched_input
    )

    try:

        parsed = json.loads(
            decision.strip()
        )

    except json.JSONDecodeError:

        if DEBUG:

            print("\n[DEBUG] Raw LLM Output:")

            print(decision)

        return {
            "tool": "error",
            "result": "Error: Agent failed to return valid JSON."
        }

    tool_name = parsed.get("tool_name")

    arguments = parsed.get(
        "arguments",
        {}
    )

    if DEBUG:

        print(f"\n[DEBUG] Tool Selected: {tool_name}")

        print(f"[DEBUG] Arguments: {arguments}")

    # ==========================================
    # STUDY PLAN TOOL
    # ==========================================

    if tool_name == "create_study_plan":

        topic = arguments.get("topic")

        days = arguments.get("days")

        if not topic or not days:

            return {
                "tool": "error",
                "result": "Error: Missing topic or days."
            }

        try:

            days = int(days)

        except (ValueError, TypeError):

            return {
                "tool": "error",
                "result": "Error: Days must be a number."
            }

        result = create_study_plan(
            topic,
            days
        )

        return {
            "tool": "create_study_plan",

            "result": f"""
==============================
📘 StudyPilot Result
==============================

{result}

==============================
"""
        }

    # ==========================================
    # QUIZ TOOL
    # ==========================================

    elif tool_name == "generate_quiz":

        topic = arguments.get("topic")

        num_questions = arguments.get(
            "num_questions"
        )

        if not topic or not num_questions:

            return {
                "tool": "error",
                "result": "Error: Missing topic or question count."
            }

        try:

            num_questions = int(
                num_questions
            )

        except (ValueError, TypeError):

            return {
                "tool": "error",
                "result": "Error: Number of questions must be a number."
            }

        result = generate_quiz(
            topic,
            num_questions
        )

        if "error" in result:

            return {
                "tool": "error",
                "result": result["error"]
            }

        # ==========================================
        # Build Quiz Output
        # ==========================================

        output = "\n==============================\n"

        output += "📝 Quiz Generated\n"

        output += "==============================\n\n"

        for question in result["questions"]:

            output += (
                f"Q{question['id']}. "
                f"{question['question']}\n"
            )

            for (
                option_key,
                option_value
            ) in question["options"].items():

                output += (
                    f"{option_key}) "
                    f"{option_value}\n"
                )

            output += "\n"

        output += "==============================\n"

        return {

            "tool": "generate_quiz",

            "result": output,

            "quiz_data": result
        }

    # ==========================================
    # SUMMARIZER TOOL
    # ==========================================

    elif tool_name == "summarize_text":

        text = arguments.get("text")

        if not text:

            return {
                "tool": "error",
                "result": "Error: Missing text to summarize."
            }

        result = summarize_text(text)

        return {

            "tool": "summarize_text",

            "result": f"""
==============================
📄 Summary
==============================

{result}

==============================
"""
        }

    # ==========================================
    # UNKNOWN TOOL
    # ==========================================

    else:

        return {
            "tool": "error",
            "result": "Error: Unknown tool selected."
        }