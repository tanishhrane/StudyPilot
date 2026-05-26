# ============================================================
# RAG SETUP GUIDE for StudyPilot
# Read this BEFORE running any code
# ============================================================

# ── NEW PACKAGES TO INSTALL ──────────────────────────────────
# Run this command in your terminal (inside your venv):
#
#   pip install pymupdf chromadb sentence-transformers groq python-dotenv
#
# What each package does:
#   pymupdf          → Read PDFs (imported as 'fitz' in code)
#   chromadb         → Local vector database (stores embeddings on disk)
#   sentence-transformers → Convert text to vectors (the embedding model)
#   groq             → Call Llama via Groq API
#   python-dotenv    → Read your .env file (GROQ_API_KEY)
#
# Note: sentence-transformers will download the embedding model (~80MB)
# the first time you run the code. After that it's cached locally.


# ── ADD TO YOUR .env FILE ────────────────────────────────────
# Open your .env file and make sure this line is there:
#
#   GROQ_API_KEY=your_groq_api_key_here
#
# Get your free Groq API key at: https://console.groq.com


# ── NEW FILES TO ADD TO YOUR PROJECT ────────────────────────
# Your project structure should look like this after adding RAG:
#
# STUDYPILOT/
# ├── tools/
# │   ├── calendar_sync.py
# │   ├── quiz.py              ← your existing file (unchanged)
# │   ├── study_plan.py        ← your existing file (unchanged)
# │   └── summarizer.py        ← your existing file (unchanged)
# ├── rag_engine.py            ← NEW: the core RAG brain
# ├── rag_quiz.py              ← NEW: RAG-powered quiz generator
# ├── rag_summarizer.py        ← NEW: RAG-powered summariser
# ├── rag_ui.py                ← NEW: Streamlit UI components
# ├── chroma_store/            ← AUTO-CREATED: ChromaDB stores data here
# ├── agent.py
# ├── app.py                   ← MODIFY: add RAG UI sections here
# ├── auth.py
# ├── config.py
# ├── llm.py
# ├── memory.py
# ├── Main.py
# └── .env


# ── HOW TO INTEGRATE INTO YOUR app.py ────────────────────────
# Add these imports at the top of your app.py:
#
#   from rag_ui import (
#       render_document_upload_sidebar,
#       render_rag_quiz_section,
#       render_rag_summarizer_section
#   )
#
# Then in your main app logic:
#
#   # In your sidebar section (where user is logged in):
#   if st.session_state.get("user"):
#       user_id = st.session_state["user"]["uid"]   # Firebase user ID
#       render_document_upload_sidebar(user_id)
#
#   # In your Quiz Generator tab/section:
#   render_rag_quiz_section(user_id)
#
#   # In your Summariser tab/section:
#   render_rag_summarizer_section(user_id)


# ── ADD TO YOUR requirements.txt ─────────────────────────────
# Add these lines to your requirements.txt:
#
#   pymupdf>=1.23.0
#   chromadb>=0.4.0
#   sentence-transformers>=2.2.0
#   groq>=0.4.0
#   python-dotenv>=1.0.0


# ── ADD chroma_store/ TO YOUR .gitignore ─────────────────────
# The chroma_store/ folder contains your vector database.
# It can get large and is user-specific, so don't commit it to Git.
# Open your .gitignore and add:
#
#   chroma_store/


# ── FIRST RUN CHECKLIST ──────────────────────────────────────
# 1. ✅ pip install pymupdf chromadb sentence-transformers groq python-dotenv
# 2. ✅ Add GROQ_API_KEY to your .env file
# 3. ✅ Add rag_engine.py, rag_quiz.py, rag_summarizer.py, rag_ui.py to project root
# 4. ✅ Import and call render functions in app.py
# 5. ✅ Run: streamlit run app.py (or Main.py, whichever starts your app)
# 6. ✅ Upload a PDF in the sidebar
# 7. ✅ Go to Quiz Generator → enter a topic → click Generate Quiz from My Notes
# 8. ✅ Go to Summariser → select document → Generate Summary


# ── HOW TO EXPLAIN RAG IN INTERVIEWS ─────────────────────────
#
# "I built a full RAG pipeline in StudyPilot from scratch.
#
#  When a student uploads their study notes, the system:
#  1. Extracts text using PyMuPDF
#  2. Chunks it into 500-character overlapping segments
#  3. Converts chunks to 384-dimensional vectors using sentence-transformers
#  4. Stores vectors in ChromaDB, keyed by Firebase user ID so data stays isolated
#
#  When generating a quiz, instead of relying on the LLM's training data,
#  we embed the topic query, perform cosine similarity search to retrieve the
#  top-5 most relevant chunks, and pass them as context to Llama via Groq.
#  This grounds the LLM entirely in the student's own notes.
#
#  The evaluator goes further — for each wrong answer, it runs a second
#  retrieval to surface the exact passage from the student's notes that
#  explains the correct concept, giving precise, contextual feedback.
#
#  The summariser uses the same pipeline to handle documents larger than
#  any LLM's context window, achieving a typical 5-10x compression ratio."
