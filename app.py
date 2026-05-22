import streamlit as st
import agent

from agent import run_agent
from memory import save_message, get_last_messages
from tools.quiz import evaluate_quiz


# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="StudyPilot",
    page_icon="📘",
    layout="centered"
)


# ==========================================
# SESSION STATE INITIALIZATION
# ==========================================

if "quiz" not in st.session_state:
    st.session_state.quiz = None

if "quiz_submitted" not in st.session_state:
    st.session_state.quiz_submitted = False

if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

if "current_mode" not in st.session_state:
    st.session_state.current_mode = "normal"

if "normal_result" not in st.session_state:
    st.session_state.normal_result = ""


# ==========================================
# HEADER
# ==========================================

st.title("📘 StudyPilot")
st.subheader("Your Agentic AI Study Assistant")


# ==========================================
# USER INPUT
# ==========================================

user_input = st.text_area(
    "Ask StudyPilot:",
    placeholder=(
        "Examples:\n"
        "- Generate a quiz on machine learning\n"
        "- Create a study plan for DSA\n"
        "- Summarize reinforcement learning"
    ),
    height=150
)


# ==========================================
# GENERATE BUTTON
# ==========================================

if st.button("Generate"):

    if user_input.strip() == "":
        st.warning("Please enter a prompt.")
        st.stop()

    # Reset old states
    st.session_state.quiz_submitted = False
    st.session_state.evaluation = None

    # Save user message
    save_message("user", user_input)

    # Retrieve history
    history = get_last_messages(limit=5)

    # Run agent
    result = run_agent(user_input, history)

    # Save assistant response
    if result:
        save_message("assistant", str(result))

    # ==========================================
    # QUIZ MODE
    # ==========================================

    if agent.latest_quiz:

        st.session_state.current_mode = "quiz"

        st.session_state.quiz = agent.latest_quiz

    # ==========================================
    # NORMAL MODE
    # ==========================================

    else:

    # Clear old quiz completely
    agent.latest_quiz = None

    st.session_state.current_mode = "normal"

    st.session_state.quiz = None

    st.session_state.quiz_submitted = False

    st.session_state.evaluation = None

    st.session_state.normal_result = result


# ==========================================
# QUIZ UI
# ==========================================

if (
    st.session_state.quiz
    and st.session_state.current_mode == "quiz"
):

    quiz = st.session_state.quiz

    st.markdown("---")

    st.header("📝 Quiz")

    user_answers = {}

    # Display questions
    for question in quiz["questions"]:

        st.subheader(
            f"Q{question['id']}. {question['question']}"
        )

        selected_answer = st.radio(
            "Choose your answer:",
            options=list(question["options"].keys()),

            format_func=lambda option:
                f"{option}) {question['options'][option]}",

            key=f"question_{question['id']}"
        )

        user_answers[question["id"]] = selected_answer

        st.markdown("")


    # ==========================================
    # SUBMIT QUIZ BUTTON
    # ==========================================

    if st.button("Submit Quiz"):

        evaluation = evaluate_quiz(
            quiz,
            user_answers
        )

        st.session_state.evaluation = evaluation

        st.session_state.quiz_submitted = True


# ==========================================
# QUIZ RESULTS
# ==========================================

if st.session_state.quiz_submitted:

    evaluation = st.session_state.evaluation

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

if (
    st.session_state.current_mode == "normal"
    and st.session_state.normal_result
):

    st.markdown("---")

    st.header("🔎 Result")

    st.write(st.session_state.normal_result)