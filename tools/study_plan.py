import json

from datetime import datetime, timedelta

from llm import call_llm


# ==========================================
# STUDY PLAN GENERATOR
# ==========================================

def create_study_plan(

    topic,
    days,
    start_date,
    time_slot

):

    system_prompt = """
You are an academic planning assistant.

You MUST return ONLY valid JSON.

Do NOT include explanations.
Do NOT include markdown.
Do NOT include extra text.

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

IMPORTANT:
- start_time and end_time must contain ONLY time
- Do NOT include dates inside time fields
- Example valid time:
  "09:00 AM"
- Example invalid time:
  "2024-01-01 09:00:00"
"""

    user_prompt = f"""
Create a {days}-day study plan
for the topic: {topic}.

The plan should start from:
{start_date}

Daily study time:
{time_slot}

Requirements:

- Divide the topic progressively from beginner to advanced
- Each day should teach SPECIFIC concepts
- Mention exact subtopics to learn
- Include realistic coding/practice tasks
- Avoid vague tasks like:
  "Study concepts"
  "Review topic"
- Make tasks practical and actionable
- Include implementation/problem-solving tasks where possible
- Each day should feel different and meaningful
- Keep the progression logical
- Use sequential day numbers
- Return ONLY valid JSON
"""

    response = call_llm(

        system_prompt,
        user_prompt

    )

    print("\n========== RAW LLM RESPONSE ==========\n")
    print(response)
    print("\n======================================\n")

    try:

        parsed_response = json.loads(response)

        formatted_output = ""

        formatted_output += (
            f"📘 "
            f"{parsed_response['plan_title']}\n\n"
        )

        # ==========================================
        # SAFE DATE PARSING — FIXED
        # Handles all formats agent may pass:
        # "2024-01-15"
        # "Monday 2024-01-15"
        # "Monday 2024-01-15"
        # "today"
        # ==========================================

        clean_date = _parse_start_date(start_date)

        base_date = datetime.strptime(
            clean_date,
            "%Y-%m-%d"
        )

        # ==========================================
        # GENERATE DAYS
        # ==========================================

        for index, study_day in enumerate(
            parsed_response["days"]
        ):

            current_date = (
                base_date + timedelta(days=index)
            ).strftime("%Y-%m-%d")

            # ==========================================
            # UPDATE JSON DATE
            # ==========================================

            study_day["date"] = current_date

            # ==========================================
            # SAFE FIELD ACCESS
            # ==========================================

            day_number = study_day.get(
                "day_number",
                index + 1
            )

            start_time = study_day.get(
                "start_time",
                "09:00 AM"
            )

            end_time = study_day.get(
                "end_time",
                "12:00 PM"
            )

            topic_name = study_day.get(
                "topic",
                "Study Session"
            )

            tasks = study_day.get(
                "tasks",
                []
            )

            # ==========================================
            # FORMATTED OUTPUT
            # ==========================================

            formatted_output += (
                f"📅 Day {day_number} "
                f"— {current_date}\n"
                f"⏰ {start_time} to {end_time}\n"
                f"📖 {topic_name}\n\n"
            )

            for task in tasks:
                formatted_output += (
                    f"  ✅ {task.get('task', '')}\n"
                    f"     {task.get('description', '')}\n\n"
                )

            formatted_output += "—" * 30 + "\n\n"

        return {
            "formatted_output": formatted_output,
            "plan_json": parsed_response
        }

    except Exception as e:
        raise e


# ==========================================
# DATE PARSER HELPER — ADDED
# ==========================================

def _parse_start_date(start_date: str) -> str:
    """
    Cleans and normalizes start_date to 
    YYYY-MM-DD format regardless of what 
    the agent passes in.
    """

    if not start_date:
        return datetime.today().strftime("%Y-%m-%d")

    start_date = start_date.strip()

    # Handle "today"
    if start_date.lower() == "today":
        return datetime.today().strftime("%Y-%m-%d")

    # Try direct parse — "2024-01-15"
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        return start_date
    except ValueError:
        pass

    # Handle "Monday 2024-01-15" — day name + date
    try:
        parts = start_date.split(" ")
        for part in parts:
            try:
                datetime.strptime(part, "%Y-%m-%d")
                return part     # return only the date
            except ValueError:
                continue
    except Exception:
        pass

    # Handle "Monday, January 15 2024"
    for fmt in [
        "%B %d %Y",
        "%B %d, %Y",
        "%d %B %Y",
        "%A, %B %d %Y",
        "%A %B %d %Y"
    ]:
        try:
            parsed = datetime.strptime(start_date, fmt)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Final fallback — use today
    return datetime.today().strftime("%Y-%m-%d")