from agent import run_agent
from memory import save_message,get_last_messages


if __name__ == "__main__":

    print("StudyPilot Agent Ready.")

    while True:
        user_input = input("\nAsk StudyPilot (or type 'exit'): ")

        if user_input.lower() == "exit":
            break

        save_message("user", user_input)
        history=get_last_messages(limit=5)


        result = run_agent(user_input,history)
        if result:

            save_message("assistant",result)
            #If there is an error or no output, in that case as well error is saved, to avoid 
            #that we used if statement.


        print("\nStudyPilot Response:\n")
        print(result)
