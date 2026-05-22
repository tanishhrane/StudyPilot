import json
from llm import call_llm
from tools.study_plan import create_study_plan
from tools.quiz import generate_quiz, evaluate_quiz
from tools.summarizer import summarize_text
from config import DEBUG


# Stores latest generated quiz
latest_quiz = None


# Helper function to format history
def format_history(history):

    conversation = ""

    for msg in history:

        role = msg["role"]
        content = msg["content"]

        conversation += f"{role.capitalize()}: {content}\n"

    return conversation


def run_agent(user_input, history):

    global latest_quiz

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

    # Format past conversation
    context = format_history(history)

    # Inject memory into user input
    enriched_input = f"""
Conversation so far:
{context}

Current user request:
{user_input}
"""

    # LLM decides which tool to use
    decision = call_llm(system_prompt, enriched_input)

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

        if "error" in result:
            return result["error"]

        # Store latest generated quiz
        latest_quiz = result

        output = "\n==============================\n"
        output += "📝 Quiz Generated\n"
        output += "==============================\n\n"

        for question in result["questions"]:

            output += f"Q{question['id']}. {question['question']}\n"

            for option_key, option_value in question["options"].items():

                output += f"{option_key}) {option_value}\n"

            output += "\n"

        output += "==============================\n"

        output += "\nEnter your answers in any format like:\n"
        output += "1-B\n"
        output += "1) B\n"
        output += "Q1: B\n"

        return output

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