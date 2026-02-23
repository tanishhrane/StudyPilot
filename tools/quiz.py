from llm import call_llm


def generate_quiz(topic, num_questions):

    system_prompt = "You are an academic quiz generator."

    user_prompt = f"""
    Generate {num_questions} multiple choice questions on the topic: {topic}.

    Format:
    Question:
    A)
    B)
    C)
    D)
    Correct Answer:
    """

    response = call_llm(system_prompt, user_prompt)

    return response
