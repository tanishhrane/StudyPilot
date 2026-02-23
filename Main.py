from agent import run_agent


if __name__ == "__main__":

    print("StudyPilot Agent Ready.")

    while True:
        user_input = input("\nAsk StudyPilot (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        result = run_agent(user_input)

        print("\nStudyPilot Response:\n")
        print(result)
