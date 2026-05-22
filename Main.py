import re

from agent import run_agent
import agent
from memory import save_message, get_last_messages
from tools.quiz import evaluate_quiz


# Parse user quiz answers



def parse_answers(answer_text):

    answers = {}

    # Split using BOTH commas and new lines
    parts = re.split(r'[,\n]+', answer_text)

    for part in parts:

        part = part.strip().upper()

        # Match formats:
        # 1-B
        # 1) B
        # Q1: B

        match = re.search(r'(\d+).*?([ABCD])', part)

        if match:

            qid = int(match.group(1))

            answer = match.group(2)

            answers[qid] = answer

    return answers


if __name__ == "__main__":

    print("StudyPilot Agent Ready.")

    while True:

        user_input = input("\nAsk StudyPilot (or type 'exit'): ").strip()

        if user_input.lower() == "exit":
            break

        # Quiz answer mode
        answer_pattern =  re.search(
                          r'(\bQ?\d+\s*[-.):]?\s*[ABCD]\b)',
                          user_input.upper()
                           )
        if agent.latest_quiz and answer_pattern:

            try:

                user_answers = parse_answers(user_input)

                result = evaluate_quiz(agent.latest_quiz, user_answers)

                print("\n==============================")
                print("📊 Quiz Results")
                print("==============================\n")

                print(f"Score: {result['score']} / {result['total']}\n")

                if result["weak_topics"]:

                    print("Weak Topics:")

                    for topic in result["weak_topics"]:
                        print(f"- {topic}")

                else:
                    print("Excellent performance!")

                print("\nDetailed Results:\n")

                for item in result["results"]:

                    status = "Correct" if item["is_correct"] else "Wrong"

                    print(
                        f"Question {item['question_id']} -> "
                        f"Your Answer: {item['user_answer']} | "
                        f"Correct Answer: {item['correct_answer']} | "
                        f"{status}"
                    )

                print("\n==============================\n")

                continue

            except Exception as e:

                print(f"Error evaluating quiz: {e}")

                continue

        # Save user message
        save_message("user", user_input)

        # Retrieve memory
        history = get_last_messages(limit=5)

        # Run agent
        result = run_agent(user_input, history)

        # Save assistant response
        if result:
            save_message("assistant", result)

        # Print response
        print("\nStudyPilot Response:\n")

        print(result)