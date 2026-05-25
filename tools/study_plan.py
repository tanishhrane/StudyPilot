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

Create clean and structured study plans.

Keep responses:
- concise
- organized
- easy to follow
"""

    user_prompt = f"""
Create a structured {days}-day study plan
for the topic: {topic}.

The plan should start from:
{start_date}

Daily study time:
{time_slot}

Requirements:

- Divide into daily study goals
- Mention concepts to study
- Include practice/revision tasks
- Keep each day realistic
- Keep output clean and structured
"""

    response = call_llm(

        system_prompt,
        user_prompt

    )

    return response
