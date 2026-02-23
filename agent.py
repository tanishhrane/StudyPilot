import json
from llm import call_llm
from tools.study_plan import create_study_plan
from tools.quiz import generate_quiz
from tools.summarizer import summarize_text
from config import DEBUG


def run_agent(user_input):

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

    decision = call_llm(system_prompt, user_input)

    try:
        parsed = json.loads(decision.strip())
    except json.JSONDecodeError:
        if DEBUG:
            print("\n[DEBUG] Raw LLM Output:")
            print(decision)
        return "Error: Agent failed to return valid JSON."

    tool_name = parsed.get("tool_name")
    arguments = parsed.get("arguments", {})

    if DEBUG:
        print(f"\n[DEBUG] Tool Selected: {tool_name}")
        print(f"[DEBUG] Arguments: {arguments}")

    # ===============================
    # Study Plan Tool
    # ===============================
    if tool_name == "create_study_plan":

        topic = arguments.get("topic")
        days = arguments.get("days")

        if not topic or not days:
            return "Error: Missing topic or days."

        try:
            days = int(days)
        except (ValueError, TypeError):
            return "Error: Days must be a number."

        result = create_study_plan(topic, days)

        return f"""
==============================
📘 StudyPilot Result
==============================

{result}

==============================
"""

    # ===============================
    # Quiz Tool
    # ===============================
    elif tool_name == "generate_quiz":

        topic = arguments.get("topic")
        num_questions = arguments.get("num_questions")

        if not topic or not num_questions:
            return "Error: Missing topic or question count."

        try:
            num_questions = int(num_questions)
        except (ValueError, TypeError):
            return "Error: Number of questions must be a number."

        result = generate_quiz(topic, num_questions)

        return f"""
==============================
📝 Quiz Generated
==============================

{result}

==============================
"""

    # ===============================
    # Summarizer Tool
    # ===============================
    elif tool_name == "summarize_text":

        text = arguments.get("text")

        if not text:
            return "Error: Missing text to summarize."

        result = summarize_text(text)

        return f"""
==============================
📄 Summary
==============================

{result}

==============================
"""

    else:
        return "Error: Unknown tool selected."