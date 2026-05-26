# ============================================================
# rag_ui.py — Streamlit UI for RAG features
# Copy the relevant sections into your existing app.py
# Each section is clearly labeled with where it goes
# ============================================================

# ── IMPORTS TO ADD AT TOP OF YOUR app.py ────────────────────
import streamlit as st
import tempfile   # Built-in Python module: creates temporary files
                  # Why: When user uploads a PDF, Streamlit gives us the file data
                  #      in memory. We need to save it to disk so fitz can read it.
                  #      tempfile.NamedTemporaryFile() creates a temp file that
                  #      auto-deletes when we're done with it.
import os

from rag_engine import store_document, list_user_documents, delete_user_document
from rag_quiz import generate_quiz_from_document, evaluate_quiz_with_rag
from rag_summarizer import summarize_document, summarize_by_topics


# ============================================================
# SECTION A — DOCUMENT UPLOAD SIDEBAR
# Add this to your sidebar in app.py
# ============================================================

def render_document_upload_sidebar(user_id: str):
    """
    Shows a PDF uploader in the Streamlit sidebar.
    This is the entry point: user uploads their notes here,
    and they become available to Quiz Generator and Summariser.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Your Study Documents")

    # st.sidebar.file_uploader() — Streamlit widget for file uploads
    # type=["pdf"] — only allow PDF files
    # Returns None if no file uploaded, or a UploadedFile object if uploaded
    uploaded_file = st.sidebar.file_uploader(
        "Upload your study notes (PDF)",
        type=["pdf"],
        help="Upload your textbook chapters, lecture notes, or any study material"
    )

    if uploaded_file is not None:
        # Give the document a friendly name (remove .pdf extension)
        # .rsplit(".", 1)[0] — splits from right at ".", takes part before the dot
        doc_name = uploaded_file.name.rsplit(".", 1)[0]

        if st.sidebar.button(f"📥 Add '{doc_name}' to my notes"):
            # ── SAVE UPLOADED FILE TO DISK ────────────────────────────────
            # Streamlit's uploaded_file is an in-memory object (like a file opened in Python)
            # fitz (PyMuPDF) needs an actual file PATH on disk, not an in-memory object
            # Solution: write it to a temporary file, get its path, use that path

            # tempfile.NamedTemporaryFile() creates a real file on disk
            # delete=False: don't auto-delete (we'll delete it manually after processing)
            # suffix=".pdf": give it a .pdf extension so fitz recognizes it
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())  # Write PDF bytes to temp file
                tmp_path = tmp_file.name                   # Get the file path

            # ── PROCESS THE PDF WITH A PROGRESS SPINNER ──────────────────
            # st.spinner() shows an animated spinner while the code inside runs
            with st.spinner(f"Processing '{doc_name}'... This takes about 10-30 seconds"):
                try:
                    num_chunks = store_document(
                        pdf_path=tmp_path,
                        user_id=user_id,
                        doc_name=doc_name
                    )
                    # st.sidebar.success() shows a green success message
                    st.sidebar.success(f"✅ Added! ({num_chunks} sections indexed)")
                except ValueError as e:
                    # st.sidebar.error() shows a red error message
                    st.sidebar.error(f"❌ Error: {str(e)}")
                finally:
                    # os.unlink() deletes a file — always clean up temp files!
                    # 'finally' block runs even if an exception occurred
                    os.unlink(tmp_path)

    # ── SHOW EXISTING DOCUMENTS ───────────────────────────────────────────
    existing_docs = list_user_documents(user_id)

    if existing_docs:
        st.sidebar.markdown("**Your indexed documents:**")
        for doc in existing_docs:
            # Create two columns: one for doc name, one for delete button
            col1, col2 = st.sidebar.columns([3, 1])
            col1.markdown(f"📄 {doc}")

            # Each button needs a unique key — use f-string with doc name
            if col2.button("🗑️", key=f"del_{doc}"):
                delete_user_document(user_id, doc)
                # st.rerun() refreshes the Streamlit page to reflect the deletion
                st.rerun()
    else:
        st.sidebar.info("No documents yet. Upload your study notes above!")


# ============================================================
# SECTION B — RAG QUIZ GENERATOR UI
# Replace or enhance your existing quiz UI section in app.py
# ============================================================

def render_rag_quiz_section(user_id: str):
    """
    Full UI for the RAG-powered Quiz Generator.
    """
    st.header("🧠 Quiz Generator")
    st.markdown("*Questions generated from YOUR uploaded study notes*")

    # Check if user has uploaded documents
    existing_docs = list_user_documents(user_id)

    if not existing_docs:
        # st.warning() shows a yellow warning box
        st.warning("⚠️ Please upload your study notes in the sidebar first!")
        return

    # ── QUIZ SETTINGS ─────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        # st.text_input() — simple text box
        topic = st.text_input(
            "Topic to quiz on",
            placeholder="e.g., Newton's Laws, Photosynthesis, World War II"
        )

    with col2:
        # st.slider() — draggable slider, returns the selected value
        num_questions = st.slider("Number of questions", min_value=3, max_value=10, value=5)

    with col3:
        # st.selectbox() — dropdown menu, returns the selected option
        difficulty = st.selectbox("Difficulty", options=["easy", "medium", "hard"])

    # ── GENERATE BUTTON ────────────────────────────────────────────────────
    if st.button("🎯 Generate Quiz from My Notes", type="primary"):
        if not topic:
            st.error("Please enter a topic!")
            return

        with st.spinner("Reading your notes and creating questions..."):
            try:
                # This is where RAG happens — questions come from the user's PDF!
                questions = generate_quiz_from_document(
                    topic=topic,
                    user_id=user_id,
                    num_questions=num_questions,
                    difficulty=difficulty
                )

                # st.session_state — Streamlit's way of storing data between interactions
                # Without session_state, variables reset every time user clicks something
                st.session_state["current_quiz"] = questions
                st.session_state["quiz_topic"] = topic
                st.session_state["user_answers"] = {}

                st.success(f"✅ Generated {len(questions)} questions from your notes!")

            except ValueError as e:
                st.error(f"❌ {str(e)}")

    # ── DISPLAY QUIZ ───────────────────────────────────────────────────────
    # Check if we have a quiz in session_state to display
    if "current_quiz" in st.session_state and st.session_state["current_quiz"]:
        questions = st.session_state["current_quiz"]

        st.markdown(f"### 📝 Quiz: {st.session_state.get('quiz_topic', 'Your Topic')}")
        st.markdown(f"*All questions sourced from your uploaded notes*")

        for i, q in enumerate(questions):
            st.markdown(f"**Q{i+1}. {q['question']}**")

            # st.radio() — multiple choice selector
            # The options dict has keys A/B/C/D, .items() gives (key, value) pairs
            options_display = [f"{key}: {val}" for key, val in q["options"].items()]

            # key= must be unique per widget — use question index
            selected = st.radio(
                f"Select answer for Q{i+1}",
                options=options_display,
                key=f"q_{i}",
                label_visibility="collapsed"  # Hides the label (we already showed the question)
            )

            if selected:
                # Extract just the letter (A, B, C, D) from "A: Some option text"
                chosen_letter = selected[0]
                st.session_state["user_answers"][str(i)] = chosen_letter

            # Show which document this question came from
            if "source" in q:
                st.caption(f"📄 Source: {q['source']}")

            st.markdown("---")

        # ── SUBMIT BUTTON ──────────────────────────────────────────────────
        if st.button("📊 Submit & Evaluate", type="primary"):
            user_answers = st.session_state.get("user_answers", {})

            if len(user_answers) < len(questions):
                st.warning(f"Please answer all {len(questions)} questions before submitting!")
                return

            with st.spinner("Evaluating your answers and finding revision passages..."):
                result = evaluate_quiz_with_rag(
                    questions=questions,
                    user_answers=user_answers,
                    user_id=user_id
                )

            # ── SHOW RESULTS ───────────────────────────────────────────────
            st.markdown("## 📊 Your Results")

            # Display score prominently using columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Score", f"{result['score']}/{result['total']}")
            col2.metric("Percentage", f"{result['percentage']}%")
            col3.metric("Performance", result['performance_label'])

            # Show weak topics (your existing feature)
            if result["weak_topics"]:
                st.markdown("### ⚠️ Topics to Revise")
                for topic in result["weak_topics"]:
                    # st.warning() for each weak topic — clear visual indicator
                    st.warning(f"📚 {topic}")

            # ── DETAILED FEEDBACK WITH RAG REVISION PASSAGES ───────────────
            st.markdown("### 📝 Detailed Feedback")

            for feedback in result["detailed_feedback"]:
                # st.expander() — collapsible section, keeps UI clean
                with st.expander(
                    f"{'✅' if feedback['is_correct'] else '❌'} {feedback['question'][:80]}..."
                ):
                    st.write(f"**Your answer:** {feedback['your_answer']}")
                    st.write(f"**Correct answer:** {feedback['correct_answer']}")
                    st.write(f"**Explanation:** {feedback['explanation']}")

                    # ── THIS IS THE RAG MAGIC ──────────────────────────────
                    # For wrong answers, show the exact passage from their notes
                    if not feedback["is_correct"] and feedback.get("revision_passage"):
                        passage = feedback["revision_passage"]
                        st.markdown("**📖 Relevant passage from your notes:**")
                        # st.info() shows a blue info box — good for highlighting passages
                        st.info(
                            f"*From '{passage['source']}':*\n\n{passage['text'][:400]}..."
                            f"\n\n*(Relevance: {passage['relevance_score']*100:.0f}%)*"
                        )


# ============================================================
# SECTION C — RAG SUMMARISER UI
# Replace or enhance your existing summariser UI in app.py
# ============================================================

def render_rag_summarizer_section(user_id: str):
    """
    Full UI for the RAG-powered Smart Summariser.
    """
    st.header("📝 Smart Summariser")
    st.markdown("*Summarise your uploaded PDFs — even 200-page textbooks*")

    existing_docs = list_user_documents(user_id)

    if not existing_docs:
        st.warning("⚠️ Please upload your study notes in the sidebar first!")
        return

    # ── SUMMARISER SETTINGS ────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        # Add "All Documents" option to summarise across everything
        doc_options = ["All uploaded documents"] + existing_docs
        selected_doc = st.selectbox("Which document to summarise?", options=doc_options)
        doc_name = "all" if selected_doc == "All uploaded documents" else selected_doc

    with col2:
        summary_type = st.selectbox(
            "Summary type",
            options=["comprehensive", "bullet_points", "exam_focused"],
            format_func=lambda x: {
                "comprehensive": "📖 Comprehensive (detailed)",
                "bullet_points": "• Bullet Points (quick revision)",
                "exam_focused":  "🎓 Exam-Focused (definitions + tips)"
            }[x]
            # format_func: transforms the option value into a display label
            # lambda x: ... — an anonymous function that takes x and returns the formatted string
        )

    # Optional topic focus
    focus_topic = st.text_input(
        "Focus on a specific topic? (optional)",
        placeholder="e.g., Leave blank for full summary, or type 'Mitochondria'"
    )

    # ── GENERATE SUMMARY ───────────────────────────────────────────────────
    if st.button("📋 Generate Summary", type="primary"):
        with st.spinner("Reading your notes and generating summary..."):
            try:
                result = summarize_document(
                    user_id=user_id,
                    doc_name=doc_name,
                    summary_type=summary_type,
                    focus_topic=focus_topic if focus_topic else None
                )

                # ── DISPLAY SUMMARY ────────────────────────────────────────
                # Show compression stats — impressive for interviewers!
                st.markdown("### 📊 Summary Stats")
                col1, col2, col3 = st.columns(3)
                col1.metric("Original Length", f"{result['original_word_count']} words")
                col2.metric("Summary Length", f"{result['summary_word_count']} words")
                col3.metric("Compression", f"{result['compression_ratio']}x shorter")

                # Quick Takeaways — show key points as chips/tags
                if result["key_points"]:
                    st.markdown("### ⚡ Quick Takeaways")
                    for point in result["key_points"]:
                        st.success(f"• {point}")  # Green boxes for key points

                # Full summary
                st.markdown("### 📖 Full Summary")
                # st.markdown() renders markdown formatting from the LLM response
                st.markdown(result["summary"])

                # Download button — users love this!
                # st.download_button() creates a button that downloads text as a file
                st.download_button(
                    label="⬇️ Download Summary as .txt",
                    data=result["summary"],
                    file_name=f"summary_{doc_name}_{summary_type}.txt",
                    mime="text/plain"
                )

            except ValueError as e:
                st.error(f"❌ {str(e)}")

    # ── MULTI-TOPIC SUMMARY (Bonus Feature) ───────────────────────────────
    st.markdown("---")
    st.markdown("### 🗂️ Topic-by-Topic Summary")
    st.markdown("Enter multiple topics to get individual summaries for each")

    topics_input = st.text_area(
        "Enter topics (one per line)",
        placeholder="Photosynthesis\nCell Division\nDNA Replication"
    )

    if st.button("📚 Summarise All Topics"):
        if not topics_input.strip():
            st.error("Please enter at least one topic!")
            return

        # .strip().split("\n") — remove whitespace, then split by newline into a list
        topics = [t.strip() for t in topics_input.strip().split("\n") if t.strip()]

        with st.spinner(f"Generating summaries for {len(topics)} topics..."):
            topic_summaries = summarize_by_topics(user_id=user_id, topics=topics)

        for topic_name, summary in topic_summaries.items():
            # st.expander() for each topic — keeps UI clean and organised
            with st.expander(f"📘 {topic_name}"):
                st.markdown(summary)
