import json

from datetime import datetime

from llm import call_llm

from tools.study_plan import create_study_plan
from tools.quiz import generate_quiz
from tools.summarizer import summarize_text

from memory import (
    save_message,
    get_last_messages
)

from config import DEBUG


# ==========================================
# Main Agent Function
# ==========================================

def run_agent(user_input):

    system_prompt = """
You are StudyPilot, an intelligent academic routing agent.

You MUST return ONLY valid JSON.

Do NOT include explanations.
Do NOT include markdown.
Do NOT include extra text.

IMPORTANT RULES:

- Only extract information explicitly mentioned by the user.
- Do NOT invent arguments.
- Do NOT assume missing values.
- If days are not mentioned, do NOT generate days.
- If time_slot is not mentioned, do NOT generate time_slot.
- If start_date is not mentioned, do NOT generate start_date.

Available tools:

1. create_study_plan
   Arguments:
   - topic (string)
   - days (integer, optional)
   - start_date (string, optional)
   - time_slot (string, required)

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
    # MEMORY RETRIEVAL
    # ==========================================

    history = get_last_messages(
        query=user_input,
        limit=5
    )

    memory_context = ""

    for msg in history:

        memory_context += (
            f"{msg['role']}: "
            f"{msg['content']}\n"
        )

    enhanced_input = f"""
Past Relevant Memory:
{memory_context}

Current User Input:
{user_input}
"""

    # ==========================================
    # TOOL DECISION
    # ==========================================

    decision = call_llm(
        system_prompt,
        enhanced_input
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

            "result": (
                "Error: Agent failed "
                "to return valid JSON."
            )
        }

    tool_name = parsed.get("tool_name")

    arguments = parsed.get(
        "arguments",
        {}
    )

    if DEBUG:

        print(
            f"\n[DEBUG] Tool Selected: "
            f"{tool_name}"
        )

        print(
            f"[DEBUG] Arguments: "
            f"{arguments}"
        )

    # ==========================================
    # SAVE USER MESSAGE
    # ==========================================

    save_message(
        role="user",
        content=user_input
    )

    # ==========================================
    # STUDY PLAN TOOL
    # ==========================================

    if tool_name == "create_study_plan":

        topic = arguments.get("topic")

        days = arguments.get("days")

        start_date = arguments.get(
            "start_date"
        )

        time_slot = arguments.get(
            "time_slot"
        )

        # ==========================================
        # REQUIRED FIELD CHECK
        # ==========================================

        if not topic:

            return {

                "tool": "error",

                "result": (
                    "Error: Missing study topic."
                )
            }

        if not time_slot:

            return {

                "tool": "error",

                "result": (
                    "Please rewrite your prompt "
                    "with your preferred daily study time.\n\n"

                    "Example:\n\n"

                    "'Make a 7-day ML study plan "
                    "from 6 PM to 8 PM'"
                )
            }

        # ==========================================
        # DEFAULT DAYS
        # ==========================================

        if not days:

            days = 7

        # ==========================================
        # DEFAULT START DATE
        # ==========================================

        if not start_date:

            today = datetime.today()

            start_date = today.strftime(
                "%Y-%m-%d"
            )

        # ==========================================
        # VALIDATE DAYS
        # ==========================================

        try:

            days = int(days)

        except (ValueError, TypeError):

            return {

                "tool": "error",

                "result": (
                    "Error: Days must be a number."
                )
            }

        # ==========================================
        # GENERATE STUDY PLAN
        # ==========================================

        result = create_study_plan(

            topic,
            days,
            start_date,
            time_slot

        )

        save_message(
            role="assistant",
            content=result["formatted_output"]
        )

        return {

            "tool": "create_study_plan",

            "result": f"""
==============================
📘 StudyPilot Result
==============================

{result['formatted_output']}

==============================
""",

            "plan_json": result["plan_json"]

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

                "result": (
                    "Error: Missing topic "
                    "or question count."
                )
            }

        try:

            num_questions = int(
                num_questions
            )

        except (ValueError, TypeError):

            return {

                "tool": "error",

                "result": (
                    "Error: Number of questions "
                    "must be a number."
                )
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

        save_message(
            role="assistant",
            content=f"Generated quiz on {topic}"
        )

        return {

            "tool": "generate_quiz",

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

                "result": (
                    "Error: Missing text "
                    "to summarize."
                )
            }

        result = summarize_text(text)

        save_message(
            role="assistant",
            content=result
        )

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

            "result": (
                "Error: Unknown tool selected."
            )
        }