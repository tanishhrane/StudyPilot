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

        parsed_response = json.loads(
            response
        )

        formatted_output = ""

        formatted_output += (
            f"📘 "
            f"{parsed_response['plan_title']}\n\n"
        )

        # ==========================================
        # SMART DATE HANDLING
        # ==========================================

        start_date_lower = start_date.lower()

        today = datetime.today()

        # ==========================================
        # TODAY
        # ==========================================

        if start_date_lower == "today":

            parsed_date = today

        # ==========================================
        # TOMORROW
        # ==========================================

        elif start_date_lower == "tomorrow":

            parsed_date = (

                today +

                timedelta(days=1)

            )

        # ==========================================
        # WEEKDAY HANDLING
        # ==========================================

        elif start_date_lower in [

            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday"

        ]:

            weekdays = {

                "monday": 0,
                "tuesday": 1,
                "wednesday": 2,
                "thursday": 3,
                "friday": 4,
                "saturday": 5,
                "sunday": 6

            }

            target_day = weekdays[
                start_date_lower
            ]

            current_day = today.weekday()

            days_ahead = (

                target_day -

                current_day

            ) % 7

            if days_ahead == 0:

                days_ahead = 7

            parsed_date = (

                today +

                timedelta(days=days_ahead)

            )

        # ==========================================
        # NORMAL YYYY-MM-DD DATE
        # ==========================================

        else:

            parsed_date = datetime.strptime(

                start_date,

                "%Y-%m-%d"

            )

        start_date = parsed_date.strftime(
            "%Y-%m-%d"
        )

        base_date = datetime.strptime(
            start_date,
            "%Y-%m-%d"
        )

        # ==========================================
        # GENERATE DAYS
        # ==========================================

        for index, study_day in enumerate(

            parsed_response["days"]

        ):

            current_date = (

                base_date +

                timedelta(days=index)

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

                f"📅 Day "
                f"{day_number} "
                f"({current_date})\n"

            )

            formatted_output += (
                f"⏰ "
                f"{start_time} "
                f"- "
                f"{end_time}\n"
            )

            formatted_output += (
                f"📖 Topic: "
                f"{topic_name}\n"
            )

            formatted_output += (
                "Tasks:\n"
            )

            # ==========================================
            # TASKS
            # ==========================================

            for task in tasks:

                if isinstance(task, dict):

                    task_title = task.get(
                        "task",
                        "Study Task"
                    )

                    task_description = task.get(
                        "description",
                        ""
                    )

                    formatted_output += (
                        f"- {task_title}\n"
                    )

                    if task_description:

                        formatted_output += (
                            f"   • "
                            f"{task_description}\n"
                        )

                else:

                    formatted_output += (
                        f"- {task}\n"
                    )

            formatted_output += "\n"

        return {

            "formatted_output":
                formatted_output,

            "plan_json":
                parsed_response

        }

    except Exception as e:

        print("\n========== STUDY PLAN ERROR ==========\n")

        print(e)

        print("\n======================================\n")

        raise e