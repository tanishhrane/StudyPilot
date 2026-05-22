import streamlit as st
import re
import agent

from agent import run_agent
from memory import save_message, get_last_messages
from tools.quiz import evaluate_quiz


# Page config
st.set_page_config(
    page_title="StudyPilot",
    page_icon="📘",
    layout="centered"
)


# Title
st.title("📘 StudyPilot")
st.subheader("Your Agentic AI Study Assistant")


# User input
user_input = st.text_area(
    "Ask StudyPilot:",
    placeholder="Example: Generate a quiz on machine learning with 3 questions"
)


# Generate button
if st.button("Generate"):

    if user_input.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    # Save user message
    save_message("user", user_input)

    # Get memory history
    history = get_last_messages(limit=5)

    # Run agent
    result = run_agent(user_input, history)

    # Save assistant response
    if result:
        save_message("assistant", str(result))

    # ==========================================
    # QUIZ UI
    # ==========================================

    if agent.latest_quiz:

        quiz = agent.latest_quiz

        st.markdown("---")

        st.header("📝 Quiz")

        user_answers = {}

        # Display questions properly
        for question in quiz["questions"]:

            st.subheader(
                f"Q{question['id']}. {question['question']}"
            )

            selected = st.radio(
                "Choose your answer:",
                options=list(question["options"].keys()),
                format_func=lambda x:
                    f"{x}) {question['options'][x]}",
                key=f"question_{question['id']}"
            )

            user_answers[question["id"]] = selected

            st.markdown("")

        # Submit quiz button
        if st.button("Submit Quiz"):

            evaluation = evaluate_quiz(
                quiz,
                user_answers
            )

            st.markdown("---")

            st.header("📊 Quiz Results")

            st.success(
                f"Score: {evaluation['score']} / {evaluation['total']}"
            )

            # Weak topics
            if evaluation["weak_topics"]:

                st.subheader("Weak Topics")

                for topic in evaluation["weak_topics"]:
                    st.write(f"- {topic}")

            else:
                st.success("Excellent performance!")

            # Detailed results
            st.subheader("Detailed Results")

            for item in evaluation["results"]:

                if item["is_correct"]:

                    st.success(
                        f"Question {item['question_id']} → Correct"
                    )

                else:

                    st.error(
                        f"Question {item['question_id']} → Wrong "
                        f"(Correct Answer: {item['correct_answer']})"
                    )

    # ==========================================
    # NORMAL RESPONSE UI
    # ==========================================

    else:

        st.markdown("---")

        st.header("🔎 Result")

        st.write(result)