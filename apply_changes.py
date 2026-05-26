# apply_changes.py
# Run this once: python apply_changes.py
# It will update all 4 files automatically

import os

# ==========================================
# FILE 1: memory.py
# ==========================================

memory_content = '''# memory.py
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid

# ==========================================
# INIT
# ==========================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="chat_memory",
    metadata={"hnsw:space": "cosine"}
)

# ==========================================
# SAVE MESSAGE
# ==========================================

def save_message(
    role,
    content,
    session_id="default",
    weak_topics=None
):

    vector = embedder.encode(content).tolist()

    collection.add(
        ids=[str(uuid.uuid4())],
        embeddings=[vector],
        documents=[content],
        metadatas=[{
            "role":        role,
            "session_id":  session_id,
            "weak_topics": ",".join(weak_topics or []),
            "timestamp":   datetime.now().isoformat()
        }]
    )

# ==========================================
# GET LAST MESSAGES
# ==========================================

def get_last_messages(
    query="study session",
    session_id="default",
    limit=5
):

    count = collection.count()

    if count == 0:
        return []

    query_vector = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(limit, count),
        where={"session_id": session_id}
    )

    messages = []

    for doc, meta in zip(
        results["documents"][0],
        results["metadatas"][0]
    ):
        messages.append({
            "content":    doc,
            "role":       meta["role"],
            "session_id": meta["session_id"]
        })

    return messages

# ==========================================
# GET WEAK TOPICS
# ==========================================

def get_weak_topics(session_id="default"):

    count = collection.count()

    if count == 0:
        return []

    results = collection.query(
        query_embeddings=[
            embedder.encode(
                "quiz wrong incorrect weak"
            ).tolist()
        ],
        n_results=min(10, count),
        where={"session_id": session_id}
    )

    weak = []

    for meta in results["metadatas"][0]:
        topics = meta.get("weak_topics", "")
        if topics:
            weak.extend(topics.split(","))

    return list(set(filter(None, weak)))
'''

# ==========================================
# FILE 2: tools/quiz.py
# ==========================================

quiz_content = '''# tools/quiz.py
import json
from llm import call_llm


# ==========================================
# QUIZ GENERATOR
# ==========================================

def generate_quiz(
    topic,
    num_questions,
    weak_topics=None
):

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

    if weak_topics:
        weak_instruction = f"""
IMPORTANT: The user has previously struggled
with these subtopics: {weak_topics}.

Prioritize questions on these weak areas.
Make those questions slightly harder than usual.
"""
    else:
        weak_instruction = ""

    user_prompt = f"""
Generate {num_questions} multiple choice questions
on the topic: {topic}.

{weak_instruction}
"""

    response = call_llm(
        system_prompt,
        user_prompt
    )

    try:
        quiz_data = json.loads(response)
        return quiz_data

    except json.JSONDecodeError:
        return {
            "error": (
                "Failed to generate "
                "valid quiz JSON."
            )
        }


# ==========================================
# QUIZ EVALUATOR
# ==========================================

def evaluate_quiz(quiz_data, user_answers):

    score = 0
    total = len(quiz_data["questions"])
    weak_topics = []
    results = []

    for question in quiz_data["questions"]:

        qid            = question["id"]
        correct_answer = question["correct_answer"]
        user_answer    = user_answers.get(qid)
        is_correct     = user_answer == correct_answer

        if is_correct:
            score += 1
        else:
            weak_topics.append(
                question["subtopic"]
            )

        results.append({
            "question_id":    qid,
            "user_answer":    user_answer,
            "correct_answer": correct_answer,
            "is_correct":     is_correct
        })

    return {
        "score":       score,
        "total":       total,
        "weak_topics": list(set(weak_topics)),
        "results":     results
    }
'''

# ==========================================
# FILE 3: agent.py
# ==========================================

agent_content = '''# agent.py
import json

from datetime import datetime

from llm import call_llm

from tools.study_plan import create_study_plan
from tools.quiz import generate_quiz
from tools.summarizer import summarize_text

from memory import (
    save_message,
    get_last_messages,
    get_weak_topics
)

from config import DEBUG


# ==========================================
# MAIN AGENT FUNCTION
# ==========================================

def run_agent(user_input):

    system_prompt = """
You are StudyPilot, an intelligent academic routing agent.

You MUST return ONLY valid JSON.

Do NOT include explanations.
Do NOT include markdown.
Do NOT include extra text.

IMPORTANT RULES:

- Only extract information explicitly mentioned by the user.
- Do NOT invent arguments.
- Do NOT assume missing values.
- If days are not mentioned, do NOT generate days.
- If time_slot is not mentioned, do NOT generate time_slot.
- If start_date is not mentioned, do NOT generate start_date.

Available tools:

1. create_study_plan
   Arguments:
   - topic (string)
   - days (integer, optional)
   - start_date (string, optional)
   - time_slot (string, required)

2. generate_quiz
   Arguments:
   - topic (string)
   - num_questions (integer)

3. summarize_text
   Arguments:
   - text (string)

Return format:

{
    "tool_name": "",
    "arguments": {}
}
"""

    # ==========================================
    # MEMORY RETRIEVAL
    # ==========================================

    history = get_last_messages(
        query=user_input,
        limit=5
    )

    memory_context = ""

    for msg in history:

        memory_context += (
            f"{msg[\'role\']}: "
            f"{msg[\'content\']}\n"
        )

    enhanced_input = f"""
Past Relevant Memory:
{memory_context}

Current User Input:
{user_input}
"""

    # ==========================================
    # TOOL DECISION
    # ==========================================

    decision = call_llm(
        system_prompt,
        enhanced_input
    )

    try:

        parsed = json.loads(
            decision.strip()
        )

    except json.JSONDecodeError:

        if DEBUG:

            print("\\n[DEBUG] Raw LLM Output:")
            print(decision)

        return {
            "tool": "error",
            "result": (
                "Error: Agent failed "
                "to return valid JSON."
            )
        }

    tool_name = parsed.get("tool_name")

    arguments = parsed.get(
        "arguments",
        {}
    )

    if DEBUG:

        print(
            f"\\n[DEBUG] Tool Selected: "
            f"{tool_name}"
        )

        print(
            f"[DEBUG] Arguments: "
            f"{arguments}"
        )

    # ==========================================
    # SAVE USER MESSAGE
    # ==========================================

    save_message(
        role="user",
        content=user_input
    )

    # ==========================================
    # STUDY PLAN TOOL
    # ==========================================

    if tool_name == "create_study_plan":

        topic = arguments.get("topic")

        days = arguments.get("days")

        start_date = arguments.get("start_date")

        time_slot = arguments.get("time_slot")

        if not topic:
            return {
                "tool": "error",
                "result": "Error: Missing study topic."
            }

        if not time_slot:
            return {
                "tool": "error",
                "result": (
                    "Please rewrite your prompt "
                    "with your preferred daily study time.\\n\\n"
                    "Example:\\n\\n"
                    "\'Make a 7-day ML study plan "
                    "from 6 PM to 8 PM\'"
                )
            }

        if not days:
            days = 7

        if not start_date:
            today = datetime.today()
            start_date = today.strftime("%A %Y-%m-%d")

        try:
            days = int(days)
        except (ValueError, TypeError):
            return {
                "tool": "error",
                "result": "Error: Number of days must be a number."
            }

        result = create_study_plan(
            topic,
            days,
            start_date,
            time_slot
        )

        save_message(
            role="assistant",
            content=result.get("formatted_output", "")
        )

        return {
            "tool": "create_study_plan",
            "result": result["formatted_output"],
            "plan_json": result["plan_json"]
        }

    # ==========================================
    # QUIZ TOOL
    # ==========================================

    elif tool_name == "generate_quiz":

        topic = arguments.get("topic")

        num_questions = arguments.get("num_questions")

        if not topic or not num_questions:
            return {
                "tool": "error",
                "result": (
                    "Error: Missing topic "
                    "or question count."
                )
            }

        try:
            num_questions = int(num_questions)
        except (ValueError, TypeError):
            return {
                "tool": "error",
                "result": (
                    "Error: Number of questions "
                    "must be a number."
                )
            }

        # ==========================================
        # FETCH WEAK TOPICS
        # ==========================================

        weak = get_weak_topics(
            session_id="default"
        )

        weak_topics_str = (
            ", ".join(weak) if weak else None
        )

        # ==========================================
        # GENERATE QUIZ WITH WEAK TOPICS
        # ==========================================

        result = generate_quiz(
            topic,
            num_questions,
            weak_topics_str
        )

        if "error" in result:
            return {
                "tool": "error",
                "result": result["error"]
            }

        save_message(
            role="assistant",
            content=f"Generated quiz on {topic}"
        )

        return {
            "tool": "generate_quiz",
            "quiz_data": result
        }

    # ==========================================
    # SUMMARIZER TOOL
    # ==========================================

    elif tool_name == "summarize_text":

        text = arguments.get("text")

        if not text:
            return {
                "tool": "error",
                "result": "Error: Missing text to summarize."
            }

        result = summarize_text(text)

        save_message(
            role="assistant",
            content=result
        )

        return {
            "tool": "summarize_text",
            "result": f"""
==============================
\\U0001f4c4 Summary
==============================

{result}

==============================
"""
        }

    # ==========================================
    # UNKNOWN TOOL
    # ==========================================

    else:

        return {
            "tool": "error",
            "result": "Error: Unknown tool selected."
        }
'''

# ==========================================
# FILE 4: app.py
# ==========================================

app_content = '''# app.py

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
    page_icon="\\U0001f4d8",
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

st.title("\\U0001f4d8 StudyPilot")

st.subheader(
    "Your Agentic AI Study Assistant"
)


# ==========================================
# AUTHENTICATION UI
# ==========================================

if st.session_state.user is None:

    st.markdown("---")

    st.subheader("\\U0001f510 Login / Signup")

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
        "Examples:\\n"
        "- Generate a quiz on machine learning\\n"
        "- Create a study plan for DSA\\n"
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

    st.session_state.quiz_submitted = False
    st.session_state.evaluation = None

    # ==========================================
    # WEAK TOPICS BANNER
    # ==========================================

    past_weak = get_weak_topics(
        session_id="default"
    )

    if past_weak:

        st.info(
            f"\\U0001f4cc Based on past quizzes, "
            f"you previously struggled with: "
            f"**{\', \'.join(past_weak)}**. "
            f"A focused quiz on these will be generated."
        )

    # ==========================================
    # RUN AGENT
    # ==========================================

    result = run_agent(user_input)

    # ==========================================
    # DETECT CURRENT TOOL
    # ==========================================

    tool_used = result.get("tool")

    # ==========================================
    # QUIZ MODE
    # ==========================================

    if tool_used == "generate_quiz":

        st.session_state.current_mode = "quiz"
        st.session_state.quiz = result["quiz_data"]

    # ==========================================
    # NORMAL MODE
    # ==========================================

    else:

        st.session_state.current_mode = "normal"
        st.session_state.quiz = None
        st.session_state.quiz_submitted = False
        st.session_state.evaluation = None
        st.session_state.normal_result = result["result"]

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

    st.header("\\U0001f4dd Quiz")

    user_answers = {}

    for question in quiz["questions"]:

        st.subheader(
            f"Q{question[\'id\']}. "
            f"{question[\'question\']}"
        )

        selected_answer = st.radio(

            "Choose your answer:",

            options=list(question["options"].keys()),

            format_func=lambda option:
                f"{option}) "
                f"{question[\'options\'][option]}",

            key=f"question_{question[\'id\']}"
        )

        user_answers[question["id"]] = selected_answer

        st.markdown("")

    if st.button("Submit Quiz"):

        evaluation = evaluate_quiz(quiz, user_answers)

        st.session_state.evaluation = evaluation
        st.session_state.quiz_submitted = True

        # ==========================================
        # SAVE WEAK TOPICS TO MEMORY
        # ==========================================

        if evaluation["weak_topics"]:

            save_message(
                role="system",
                content=(
                    f"User struggled with: "
                    f"{\', \'.join(evaluation[\'weak_topics\'])}"
                ),
                session_id="default",
                weak_topics=evaluation["weak_topics"]
            )


# ==========================================
# QUIZ RESULTS
# ==========================================

if st.session_state.quiz_submitted:

    evaluation = st.session_state.evaluation

    st.markdown("---")

    st.header("\\U0001f4ca Quiz Results")

    st.success(
        f"Score: "
        f"{evaluation[\'score\']} / "
        f"{evaluation[\'total\']}"
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
                f"Question {item[\'question_id\']} \\U00002192 Correct"
            )
        else:
            st.error(
                f"Question {item[\'question_id\']} \\U00002192 Wrong "
                f"(Correct Answer: {item[\'correct_answer\']})"
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

    st.header("\\U0001f50e Result")

    st.write(st.session_state.normal_result)


# ==========================================
# CALENDAR SYNC BUTTON
# ==========================================

if st.session_state.plan_json:

    st.markdown("---")

    if st.button("\\U0001f4c5 Sync Plan To Google Calendar"):

        sync_result = sync_study_plan_to_calendar(
            st.session_state.plan_json
        )

        st.success(sync_result)
'''

# ==========================================
# FILE 5: requirements.txt
# ==========================================

requirements_content = '''groq
python-dotenv
streamlit
pyrebase4
google-auth
google-auth-oauthlib
requests
google-api-python-client
google-auth-httplib2
streamlit-oauth
chromadb
sentence-transformers
'''

# ==========================================
# WRITE ALL FILES
# ==========================================

files = {
    "memory.py":         memory_content,
    "tools/quiz.py":     quiz_content,
    "agent.py":          agent_content,
    "app.py":            app_content,
    "requirements.txt":  requirements_content,
}

for filepath, content in files.items():

    # Make sure tools/ directory exists
    os.makedirs(
        os.path.dirname(filepath),
        exist_ok=True
    ) if os.path.dirname(filepath) else None

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"✅ Updated: {filepath}")

print("\n🎉 All files updated successfully!")
print("Now run: git add . && git commit -m 'Add RAG memory with ChromaDB' && git push")