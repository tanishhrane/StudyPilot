# ==========================================
# IMPORTS
# ==========================================

import streamlit as st
import tempfile
import os

from agent import run_agent
from tools.quiz import evaluate_quiz
from tools.calendar_sync import sync_study_plan_to_calendar

from memory import save_message, get_weak_topics

from auth import login_user, signup_user, google_login

# ── RAG IMPORTS ───────────────────────────────────────────────
# rag_engine  → PDF/DOCX/PPTX/TXT processing + FAISS storage/retrieval
# rag_tools   → RAG-powered quiz, evaluator, summarizer, study plan
from rag_engine import (
    store_document,
    list_user_documents,
    delete_user_document,
    has_documents
)
from tools.rag_tools import (
    generate_quiz_from_docs,
    evaluate_quiz_with_rag,
    summarize_document_rag,
    create_study_plan_from_docs
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

# ── Original states (unchanged) ───────────────────────────────
if "quiz"            not in st.session_state: st.session_state.quiz            = None
if "quiz_submitted"  not in st.session_state: st.session_state.quiz_submitted  = False
if "evaluation"      not in st.session_state: st.session_state.evaluation      = None
if "current_mode"    not in st.session_state: st.session_state.current_mode    = "normal"
if "normal_result"   not in st.session_state: st.session_state.normal_result   = ""
if "user"            not in st.session_state: st.session_state.user            = None
if "plan_json"       not in st.session_state: st.session_state.plan_json       = None

# ── New RAG states ────────────────────────────────────────────
if "rag_quiz"            not in st.session_state: st.session_state.rag_quiz            = None
if "rag_quiz_submitted"  not in st.session_state: st.session_state.rag_quiz_submitted  = False
if "rag_evaluation"      not in st.session_state: st.session_state.rag_evaluation      = None
if "rag_user_answers"    not in st.session_state: st.session_state.rag_user_answers    = {}
if "rag_summary_result"  not in st.session_state: st.session_state.rag_summary_result  = None


# ==========================================
# HEADER
# ==========================================

st.title("📘 StudyPilot")
st.subheader("Your Agentic AI Study Assistant")


# ==========================================
# AUTHENTICATION UI  (unchanged)
# ==========================================

if st.session_state.user is None:

    st.markdown("---")
    st.subheader("🔐 Login / Signup")

    auth_mode = st.selectbox("Choose Option", ["Login", "Signup"])
    email     = st.text_input("Email")
    password  = st.text_input("Password", type="password")

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
# LOGGED IN USER  (unchanged)
# ==========================================

if st.session_state.user:

    st.success("Logged in successfully")

    if "name"    in st.session_state.user: st.write("Name:",  st.session_state.user["name"])
    if "email"   in st.session_state.user: st.write("Email:", st.session_state.user["email"])
    if "picture" in st.session_state.user: st.image(st.session_state.user["picture"], width=80)

    if st.button("Logout"):
        st.session_state.user = None
        st.rerun()

    # ── Get Firebase user ID for RAG isolation ─────────────────
    # Each user's documents are stored separately using their Firebase UID.
    # We try "localId" first (email/password login), then "uid" (Google login),
    # then fall back to email so it always works.
    user_id = (
        st.session_state.user.get("localId")
        or st.session_state.user.get("uid")
        or st.session_state.user.get("email", "default_user")
    )


# ==========================================
# SIDEBAR — DOCUMENT UPLOAD
# ==========================================

st.sidebar.markdown("## 📂 Study Documents")
st.sidebar.markdown("Upload your notes to power AI features with your own material.")

uploaded_file = st.sidebar.file_uploader(
    "Upload file",
    type=["pdf", "docx", "pptx", "txt"],
    help="Supports PDF, Word documents, PowerPoint slides, and plain text"
)

if uploaded_file is not None:
    # Get a clean document name from the filename
    doc_name = os.path.splitext(uploaded_file.name)[0]

    if st.sidebar.button(f"📥 Index '{doc_name}'"):
        # Save the uploaded file to a temporary path on disk.
        # Streamlit gives us an in-memory file object; our extraction
        # functions need a real file path, so we write it to disk first.
        suffix = os.path.splitext(uploaded_file.name)[1]  # e.g. ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        with st.sidebar.status(f"Processing '{doc_name}'...", expanded=True) as status:
            try:
                st.sidebar.write("📄 Extracting text...")
                st.sidebar.write("✂️  Chunking...")
                st.sidebar.write("🔢 Embedding + storing...")
                n = store_document(tmp_path, user_id, doc_name)
                status.update(label=f"✅ Indexed {n} sections", state="complete")
                st.sidebar.success(f"'{doc_name}' is ready to use!")
            except ValueError as e:
                status.update(label="❌ Failed", state="error")
                st.sidebar.error(str(e))
            finally:
                os.unlink(tmp_path)  # Always delete the temp file

# Show already-indexed documents with delete buttons
existing_docs = list_user_documents(user_id)

if existing_docs:
    st.sidebar.markdown("**Indexed documents:**")
    for doc in existing_docs:
        c1, c2 = st.sidebar.columns([3, 1])
        c1.markdown(f"📄 {doc}")
        if c2.button("🗑️", key=f"del_{doc}", help=f"Remove {doc}"):
            delete_user_document(user_id, doc)
            st.rerun()
else:
    st.sidebar.info("No documents indexed yet.")


# ==========================================
# TABS
# ==========================================

tab_agent, tab_rag_quiz, tab_rag_summary, tab_rag_plan = st.tabs([
    "🤖 Ask StudyPilot",
    "🧠 Quiz from My Notes",
    "📝 Summarise My Notes",
    "📅 Plan from My Notes"
])


# ==========================================
# TAB 1 — ORIGINAL AGENT  (unchanged from your original app.py)
# ==========================================

with tab_agent:

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

    if st.button("Generate"):

        if user_input.strip() == "":
            st.warning("Please enter a prompt.")
            st.stop()

        st.session_state.quiz           = None
        st.session_state.quiz_submitted = False
        st.session_state.evaluation     = None

        result    = run_agent(user_input)
        tool_used = result.get("tool")

        # Weak topics banner (your original logic, unchanged)
        current_topic = None
        if tool_used == "generate_quiz":
            current_topic = result.get("quiz_data", {}).get("topic")

        past_weak = get_weak_topics(session_id="default", topic=current_topic)
        if past_weak:
            st.info(
                f"📌 Based on past quizzes, you previously struggled with: "
                f"**{', '.join(past_weak)}**. A focused quiz will be generated."
            )

        if tool_used == "generate_quiz":
            st.session_state.current_mode = "quiz"
            st.session_state.quiz         = result["quiz_data"]
        else:
            st.session_state.current_mode  = "normal"
            st.session_state.quiz          = None
            st.session_state.quiz_submitted = False
            st.session_state.evaluation    = None
            st.session_state.normal_result = result["result"]
            if tool_used == "create_study_plan":
                st.session_state.plan_json = result.get("plan_json")

    # Quiz UI (unchanged)
    if st.session_state.quiz and st.session_state.current_mode == "quiz":

        quiz = st.session_state.quiz
        st.markdown("---")
        st.header("📝 Quiz")
        user_answers = {}

        for question in quiz["questions"]:
            st.subheader(f"Q{question['id']}. {question['question']}")
            selected_answer = st.radio(
                "Choose your answer:",
                options=list(question["options"].keys()),
                format_func=lambda option: f"{option}) {question['options'][option]}",
                key=f"question_{question['id']}"
            )
            user_answers[question["id"]] = selected_answer
            st.markdown("")

        if st.button("Submit Quiz"):
            evaluation = evaluate_quiz(quiz, user_answers)
            st.session_state.evaluation     = evaluation
            st.session_state.quiz_submitted = True
            if evaluation["weak_topics"]:
                save_message(
                    role="system",
                    content=f"User struggled with: {', '.join(evaluation['weak_topics'])}",
                    session_id="default",
                    weak_topics=evaluation["weak_topics"],
                    topic=quiz.get("topic")
                )

    # Quiz results (unchanged)
    if st.session_state.quiz_submitted:
        evaluation = st.session_state.evaluation
        st.markdown("---")
        st.header("📊 Quiz Results")
        st.success(f"Score: {evaluation['score']} / {evaluation['total']}")

        if evaluation["weak_topics"]:
            st.subheader("Weak Topics")
            for topic in evaluation["weak_topics"]:
                st.write(f"- {topic}")
        else:
            st.success("Excellent performance!")

        st.subheader("Detailed Results")
        for item in evaluation["results"]:
            if item["is_correct"]:
                st.success(f"Question {item['question_id']} → Correct")
            else:
                st.error(f"Question {item['question_id']} → Wrong (Correct Answer: {item['correct_answer']})")

    # Normal result (unchanged)
    if st.session_state.current_mode == "normal" and st.session_state.normal_result:
        st.markdown("---")
        st.header("🔎 Result")
        st.write(st.session_state.normal_result)

    # Calendar sync (unchanged)
    if st.session_state.plan_json:
        st.markdown("---")
        if st.button("📅 Sync Plan To Google Calendar"):
            sync_result = sync_study_plan_to_calendar(st.session_state.plan_json)
            st.success(sync_result)


# ==========================================
# TAB 2 — RAG QUIZ  (new)
# ==========================================

with tab_rag_quiz:

    st.header("🧠 Quiz from My Notes")
    st.markdown("Generates questions **directly from your uploaded study material** — not generic AI knowledge.")

    if not has_documents(user_id):
        st.info("👈 Upload your study notes using the sidebar to get started.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            rag_topic = st.text_input("Topic", placeholder="e.g., Newton's Laws")
        with col2:
            rag_num_q = st.slider("Questions", min_value=3, max_value=10, value=5)
        with col3:
            rag_diff  = st.selectbox("Difficulty", ["easy", "medium", "hard"])

        if st.button("🎯 Generate Quiz from My Notes", type="primary"):
            if not rag_topic.strip():
                st.error("Please enter a topic.")
            else:
                st.session_state.rag_quiz           = None
                st.session_state.rag_quiz_submitted = False
                st.session_state.rag_evaluation     = None
                st.session_state.rag_user_answers   = {}

                with st.spinner("Searching your notes and crafting questions..."):
                    # Fetch weak topics from memory for this topic (same as original agent)
                    weak = get_weak_topics(session_id="default", topic=rag_topic)
                    weak_str = ", ".join(weak) if weak else None

                    result = generate_quiz_from_docs(
                        topic=rag_topic,
                        user_id=user_id,
                        num_questions=rag_num_q,
                        difficulty=rag_diff,
                        weak_topics=weak_str
                    )

                if "error" in result:
                    st.error(result["error"])
                else:
                    st.session_state.rag_quiz = result
                    st.success(f"✅ Generated {len(result['questions'])} questions from your notes!")

        # ── Display quiz ──────────────────────────────────────
        if st.session_state.rag_quiz and not st.session_state.rag_quiz_submitted:
            quiz = st.session_state.rag_quiz
            st.markdown("---")
            st.markdown(f"### 📝 {quiz.get('topic', 'Quiz')}")
            st.caption("All questions sourced from your uploaded study material")

            for i, q in enumerate(quiz["questions"]):
                st.subheader(f"Q{q['id']}. {q['question']}")
                options_display = [f"{k}: {v}" for k, v in q["options"].items()]
                selected = st.radio(
                    f"Answer for Q{q['id']}",
                    options=options_display,
                    key=f"rag_q_{i}",
                    label_visibility="collapsed"
                )
                if selected:
                    st.session_state.rag_user_answers[q["id"]] = selected[0]
                if q.get("source"):
                    st.caption(f"📄 From: {q['source']}")
                st.markdown("")

            if st.button("📊 Submit & Evaluate", type="primary", key="submit_rag_quiz"):
                answered = len(st.session_state.rag_user_answers)
                total    = len(quiz["questions"])
                if answered < total:
                    st.warning(f"Please answer all {total} questions. ({answered}/{total} done)")
                else:
                    with st.spinner("Evaluating and finding revision passages..."):
                        rag_result = evaluate_quiz_with_rag(
                            quiz_data=quiz,
                            user_answers=st.session_state.rag_user_answers,
                            user_id=user_id
                        )
                        st.session_state.rag_evaluation     = rag_result
                        st.session_state.rag_quiz_submitted = True

                        # Save weak topics to memory (same as original)
                        if rag_result["weak_topics"]:
                            save_message(
                                role="system",
                                content=f"User struggled with: {', '.join(rag_result['weak_topics'])}",
                                session_id="default",
                                weak_topics=rag_result["weak_topics"],
                                topic=quiz.get("topic")
                            )
                    st.rerun()

        # ── Results ───────────────────────────────────────────
        if st.session_state.rag_quiz_submitted and st.session_state.rag_evaluation:
            result = st.session_state.rag_evaluation

            st.markdown("---")
            st.header("📊 Quiz Results")

            c1, c2, c3 = st.columns(3)
            c1.metric("Score",       f"{result['score']} / {result['total']}")
            c2.metric("Percentage",  f"{result['percentage']}%")
            c3.metric("Performance", result["performance_label"])

            if result["weak_topics"]:
                st.subheader("⚠️ Topics to Revise")
                for t in result["weak_topics"]:
                    st.warning(f"📚 {t}")
            else:
                st.success("🌟 Excellent! No weak topics.")

            st.subheader("📝 Detailed Feedback")
            for item in result["results"]:
                icon = "✅" if item["is_correct"] else "❌"
                with st.expander(f"{icon}  Q{item['question_id']}: {item['question'][:70]}..."):
                    c1, c2 = st.columns(2)
                    c1.markdown(f"**Your answer:** `{item['user_answer']}`")
                    c2.markdown(f"**Correct answer:** `{item['correct_answer']}`")

                    # ── RAG revision passage for wrong answers ─
                    if not item["is_correct"] and item.get("revision_passage"):
                        p = item["revision_passage"]
                        st.markdown("**📖 From your notes:**")
                        st.info(
                            f"*Source: {p['source']}*\n\n{p['text']}\n\n"
                            f"*Relevance: {p['score'] * 100:.0f}%*"
                        )

            if st.button("🔄 Retake Quiz", key="retake_rag"):
                st.session_state.rag_quiz_submitted = False
                st.session_state.rag_evaluation     = None
                st.session_state.rag_user_answers   = {}
                st.rerun()


# ==========================================
# TAB 3 — RAG SUMMARISER  (new)
# ==========================================

with tab_rag_summary:

    st.header("📝 Summarise My Notes")
    st.markdown("Summarises your uploaded documents — handles large files that wouldn't fit in a single prompt.")

    if not has_documents(user_id):
        st.info("👈 Upload your study notes using the sidebar to get started.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            doc_options  = ["All uploaded documents"] + existing_docs
            selected_doc = st.selectbox("Document", doc_options)
            doc_for_rag  = "all" if selected_doc == "All uploaded documents" else selected_doc
        with c2:
            summary_type = st.selectbox(
                "Style",
                ["comprehensive", "bullet_points", "exam_focused"],
                format_func=lambda x: {
                    "comprehensive": "📖 Comprehensive",
                    "bullet_points": "• Bullet Points",
                    "exam_focused":  "🎓 Exam-Focused"
                }[x]
            )

        focus = st.text_input("Focus on a specific topic? (optional)", placeholder="e.g. Mitochondria")

        if st.button("📋 Generate Summary", type="primary"):
            with st.spinner("Retrieving key sections and summarising..."):
                try:
                    summary_result = summarize_document_rag(
                        user_id=user_id,
                        doc_name=doc_for_rag,
                        summary_type=summary_type,
                        focus_topic=focus.strip() if focus.strip() else None
                    )
                    st.session_state.rag_summary_result = summary_result
                except ValueError as e:
                    st.error(str(e))

        if st.session_state.rag_summary_result:
            res = st.session_state.rag_summary_result

            st.markdown("### 📊 Compression Stats")
            c1, c2, c3 = st.columns(3)
            c1.metric("Original", f"{res['original_word_count']} words")
            c2.metric("Summary",  f"{res['summary_word_count']} words")
            c3.metric("Ratio",    f"{res['compression_ratio']}x shorter")

            if res.get("key_points"):
                st.markdown("### ⚡ Key Takeaways")
                for p in res["key_points"]:
                    st.success(f"• {p}")

            st.markdown("### 📖 Full Summary")
            st.markdown(res["summary"])

            st.download_button(
                "⬇️ Download Summary",
                data=res["summary"],
                file_name=f"summary_{res['source']}_{summary_type}.txt",
                mime="text/plain"
            )


# ==========================================
# TAB 4 — RAG STUDY PLAN  (new)
# ==========================================

with tab_rag_plan:

    st.header("📅 Study Plan from My Notes")
    st.markdown("Generates a personalised study plan based on what's **actually in your uploaded material**.")

    if not has_documents(user_id):
        st.info("👈 Upload your study notes using the sidebar to get started.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            plan_topic = st.text_input("Topic", placeholder="e.g., Operating Systems")
            plan_days  = st.number_input("Number of days", min_value=1, max_value=30, value=7)
        with c2:
            plan_start = st.text_input("Start date", placeholder="today / tomorrow / Monday / 2025-06-01")
            plan_time  = st.text_input("Daily study time", placeholder="e.g., 6 PM to 8 PM")

        if st.button("📋 Generate Plan from My Notes", type="primary"):
            if not plan_topic.strip():
                st.error("Please enter a topic.")
            elif not plan_time.strip():
                st.error("Please enter a daily study time.")
            else:
                start = plan_start.strip() if plan_start.strip() else "today"
                with st.spinner("Reading your notes and building your personalised plan..."):
                    try:
                        plan_result = create_study_plan_from_docs(
                            topic=plan_topic,
                            user_id=user_id,
                            days=int(plan_days),
                            start_date=start,
                            time_slot=plan_time
                        )
                        st.markdown("---")
                        st.markdown(plan_result["formatted_output"])

                        # Store plan_json so calendar sync works here too
                        st.session_state.plan_json = plan_result["plan_json"]

                    except Exception as e:
                        st.error(f"Could not generate plan: {str(e)}")

        # Calendar sync works from any tab as long as plan_json is set
        if st.session_state.plan_json:
            st.markdown("---")
            if st.button("📅 Sync to Google Calendar", key="rag_cal_sync"):
                sync_result = sync_study_plan_to_calendar(st.session_state.plan_json)
                st.success(sync_result)
