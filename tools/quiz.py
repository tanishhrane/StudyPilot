import json
from llm import call_llm


def generate_quiz(topic, num_questions):

    system_prompt = """
You are an academic quiz generator.

You MUST return ONLY valid JSON.

Format:

{
    "topic": "",
    "questions": [
        {
            "id": 1,
            "question": "",
            "options": {
                "A": "",
                "B": "",
                "C": "",
                "D": ""
            },
            "correct_answer": "",
            "subtopic": ""
        }
    ]
}
"""

    user_prompt = f"""
Generate {num_questions} multiple choice questions on the topic: {topic}.
"""

    response = call_llm(system_prompt, user_prompt)

    try:
        quiz_data = json.loads(response)
        return quiz_data

    except json.JSONDecodeError:
        return {
            "error": "Failed to generate valid quiz JSON."
        }


#  NEW FUNCTION
# VERY IMPORTANT FUNCTION FOR TODAY
def evaluate_quiz(quiz_data, user_answers):

    score = 0
    total = len(quiz_data["questions"])

    weak_topics = []

    results = []

    for question in quiz_data["questions"]:

        qid = question["id"]

        correct_answer = question["correct_answer"]

        user_answer = user_answers.get(qid)

        is_correct = user_answer == correct_answer

        if is_correct:
            score += 1
        else:
            weak_topics.append(question["subtopic"])

        results.append({
            "question_id": qid,
            "user_answer": user_answer,
            "correct_answer": correct_answer,
            "is_correct": is_correct
        })

    return {
        "score": score,
        "total": total,
        "weak_topics": list(set(weak_topics)),
        "results": results
    }