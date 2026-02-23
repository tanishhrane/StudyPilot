from llm import call_llm

def create_study_plan(topic,days):
  print("hello")
  system_prompt="You are an academic planning assistant."
  user_prompt = f"""
    Create a structured {days}-day study plan for the topic: {topic}.

    Requirements:
    - Break into daily goals
    - Include concepts to cover
    - Include practice suggestions
    - Keep it concise but structured
    """
  response=call_llm(system_prompt,user_prompt)
  return response
