from llm import call_llm


def summarize_text(text):

    system_prompt = "You are a concise academic summarizer."

    user_prompt = f"""
    Summarize the following content clearly in bullet points:

    {text}
    """

    response = call_llm(system_prompt, user_prompt)

    return response
