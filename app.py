# ==========================================
# IMPORTS
# ==========================================

import streamlit as st

from agent import run_agent

from tools.quiz import evaluate_quiz

from tools.calendar_sync import (
    sync_study_plan_to_calendar
)

from memory import (
    save_message,
    get_weak_topics
)

from auth import (
    login_user,
    signup_user,
    google_login
)


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

if "user" not in st.session_state:
    st.session_state.user = None

if "plan_json" not in st.session_state:
    st.session_state.plan_json = None


# ==========================================
# HEADER
# ==========================================

st.title("📘 StudyPilot")

st.subheader(
    "Your Agentic AI Study Assistant"
)


# ==========================================
# AUTHENTICATION UI
# ==========================================

if st.session_state.user is None:

    st.markdown("---")

    st.subheader("🔐 Login / Signup")

    auth_mode = st.selectbox(
        "Choose Option",
        ["Login", "Signup"]
    )

    email = st.text_input("Email")

    password = st.text_input(
        "Password",
        type="password"
    )

    if auth_mode == "Login":

        if st.button("Login"):

            user = login_user(email, password)

            if user:
                st.session_state.user = user
                st.success("Login successful!")
                st.rerun()
            else:
                st.error("Invalid email or password.")

    else:

        if st.button("Create Account"):

            user = signup_user(email, password)

            if user:
                st.success("Account created successfully!")
            else:
                st.error("Signup failed.")

    st.markdown("---")

    st.subheader("Google Authentication")

    if st.button("Continue with Google"):

        user_info = google_login()

        if user_info:
            st.session_state.user = user_info
            st.success("Google Login Successful!")
            st.rerun()
        else:
            st.error("Google Login Failed")

    st.stop()


# ==========================================
# LOGGED IN USER
# ==========================================

if st.session_state.user:

    st.success("Logged in successfully")

    if "name" in st.session_state.user:
        st.write("Name:", st.session_state.user["name"])

    if "email" in st.session_state.user:
        st.write("Email:", st.session_state.user["email"])

    if "picture" in st.session_state.user:
        st.image(
            st.session_state.user["picture"],
            width=80
        )

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()


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

    # ==========================================
    # RESET STATES
    # ==========================================

    st.session_state.quiz           = None
    st.session_state.quiz_submitted = False
    st.session_state.evaluation     = None

    # ==========================================
    # RUN AGENT FIRST — to detect topic
    # ==========================================

    result = run_agent(user_input)

    tool_used = result.get("tool")

    # ==========================================
    # WEAK TOPICS BANNER (topic-scoped)
    # ==========================================

    current_topic = None

    if tool_used == "generate_quiz":
        current_topic = result.get(
            "quiz_data", {}
        ).get("topic")

    past_weak = get_weak_topics(
        session_id="default",
        topic=current_topic
    )

    if past_weak and tool_used == "generate_quiz":

        st.info(
            f"📌 Based on past quizzes, "
            f"you previously struggled with: "
            f"**{', '.join(past_weak)}**. "
            f"A focused quiz on these will be generated."
        )

    # ==========================================
    # QUIZ MODE
    # ==========================================

    if tool_used == "generate_quiz":

        st.session_state.current_mode = "quiz"
        st.session_state.quiz         = result["quiz_data"]

    # ==========================================
    # NORMAL MODE
    # ==========================================

    else:

        st.session_state.current_mode    = "normal"
        st.session_state.quiz            = None
        st.session_state.quiz_submitted  = False
        st.session_state.evaluation      = None
        st.session_state.normal_result   = result["result"]

        if tool_used == "create_study_plan":
            st.session_state.plan_json = result.get("plan_json")


# ==========================================
# QUIZ UI
# ==========================================

if (
    st.session_state.quiz
    and
    st.session_state.current_mode == "quiz"
):

    quiz = st.session_state.quiz

    st.markdown("---")

    st.header("📝 Quiz")

    user_answers = {}

    for question in quiz["questions"]:

        st.subheader(
            f"Q{question['id']}. "
            f"{question['question']}"
        )

        selected_answer = st.radio(

            "Choose your answer:",

            options=list(question["options"].keys()),

            format_func=lambda option:
                f"{option}) "
                f"{question['options'][option]}",

            key=f"question_{question['id']}"
        )

        user_answers[question["id"]] = selected_answer

        st.markdown("")

    if st.button("Submit Quiz"):

        evaluation = evaluate_quiz(quiz, user_answers)

        st.session_state.evaluation      = evaluation
        st.session_state.quiz_submitted  = True

        # ==========================================
        # SAVE WEAK TOPICS TO MEMORY (topic-scoped)
        # ==========================================

        if evaluation["weak_topics"]:

            save_message(
                role="system",
                content=(
                    f"User struggled with: "
                    f"{', '.join(evaluation['weak_topics'])}"
                ),
                session_id="default",
                weak_topics=evaluation["weak_topics"],
                topic=quiz.get("topic")
            )


# ==========================================
# QUIZ RESULTS
# ==========================================

if st.session_state.quiz_submitted:

    evaluation = st.session_state.evaluation

    st.markdown("---")

    st.header("📊 Quiz Results")

    st.success(
        f"Score: "
        f"{evaluation['score']} / "
        f"{evaluation['total']}"
    )

    if evaluation["weak_topics"]:

        st.subheader("Weak Topics")

        for topic in evaluation["weak_topics"]:
            st.write(f"- {topic}")

    else:

        st.success("Excellent performance!")

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
    and
    st.session_state.normal_result
):

    st.markdown("---")

    st.header("🔎 Result")

    st.write(st.session_state.normal_result)


# ==========================================
# CALENDAR SYNC BUTTON
# ==========================================

if st.session_state.plan_json:

    st.markdown("---")

    if st.button("📅 Sync Plan To Google Calendar"):

        sync_result = sync_study_plan_to_calendar(
            st.session_state.plan_json
        )

        st.success(sync_result)