# ============================================================
# rag_engine.py — The BRAIN of your RAG system
# This file is shared by both the Quiz Generator and Summariser
# Both tools import from here. You build this once, use everywhere.
# ============================================================

# ── IMPORTS ─────────────────────────────────────────────────
# What: fitz is the Python name for PyMuPDF, a library to read PDF files
# Why:  We need to extract raw text from PDFs the user uploads
import fitz  # pip install pymupdf

# What: os lets Python talk to your operating system (read env vars, check paths)
# Why:  We use it to read your GROQ_API_KEY from the .env file
import os

# What: chromadb is a lightweight vector database that runs locally
# Why:  We store our embedded chunks here so we can search them later
# Think of it like a smart dictionary where keys are "meaning" not exact words
import chromadb  # pip install chromadb

# What: SentenceTransformer converts text into a list of numbers (a vector/embedding)
# Why:  Computers can't compare meaning of sentences directly, but they CAN compare
#       lists of numbers. Similar sentences → similar numbers → easy to find related chunks
from sentence_transformers import SentenceTransformer  # pip install sentence-transformers

# What: Groq is the SDK to call the Groq API (which runs your Llama model)
# Why:  After we retrieve relevant chunks, we feed them to Llama to generate the answer
from groq import Groq  # pip install groq

# What: dotenv reads your .env file and loads variables like GROQ_API_KEY into os.environ
# Why:  You should NEVER hardcode API keys in your code (security risk)
from dotenv import load_dotenv  # pip install python-dotenv

# What: re is Python's built-in regex (pattern matching) module
# Why:  We use it to clean up text extracted from PDFs (remove weird characters)
import re

# What: List, Dict, Tuple are type hints from Python's typing module
# Why:  Makes code readable and helps your IDE give better suggestions
from typing import List, Dict, Tuple


# ── LOAD ENVIRONMENT VARIABLES ───────────────────────────────
# load_dotenv() reads your .env file (which has GROQ_API_KEY=your_key_here)
# After this line, os.environ["GROQ_API_KEY"] will work
load_dotenv()


# ============================================================
# STEP 1 — INITIALIZE MODELS & DATABASE
# ============================================================

# Create a SentenceTransformer model
# "all-MiniLM-L6-v2" is a small but powerful model that converts text → vectors
# It downloads automatically the first time (~80MB), then cached locally
# This model maps sentences to a 384-dimensional vector space
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# Create a ChromaDB client — this is your local vector database
# PersistentClient means data is SAVED to disk (not lost when your app restarts)
# The folder "chroma_store" will be created in your project root
chroma_client = chromadb.PersistentClient(path="./chroma_store")

# Initialize the Groq client using your API key from .env
# This is what calls Llama via Groq's servers
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


# ============================================================
# STEP 2 — PDF TEXT EXTRACTION
# ============================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Opens a PDF file and extracts all the text from every page.

    fitz.open()  — Opens the PDF file and loads it into memory
    doc.load_page(i) — Loads a single page by index (0 = first page)
    page.get_text() — Extracts all text from that page as a plain string

    Args:
        pdf_path: The file path to the PDF (e.g., "/tmp/notes.pdf")

    Returns:
        A single big string with all the text from all pages combined
    """
    doc = fitz.open(pdf_path)      # Open the PDF
    full_text = ""

    for i in range(len(doc)):      # Loop through every page
        page = doc.load_page(i)    # Load page i
        full_text += page.get_text()  # Extract text and append

    doc.close()  # Always close the file after reading

    # Clean up the text: remove extra whitespace, fix weird line breaks
    # re.sub(pattern, replacement, string) — replaces all matches of pattern
    full_text = re.sub(r'\s+', ' ', full_text).strip()

    return full_text


# ============================================================
# STEP 3 — CHUNKING (Split text into small pieces)
# ============================================================

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Splits a long piece of text into smaller overlapping chunks.

    WHY CHUNKS?
    - LLMs have token limits. You can't feed a 100-page book into a prompt.
    - Vector search works better on small, focused pieces of text.
    - Overlap ensures we don't cut a sentence in half and lose meaning at boundaries.

    HOW OVERLAP WORKS:
    chunk_size=500, overlap=50 means:
    - Chunk 1: characters 0   → 500
    - Chunk 2: characters 450 → 950   (starts 50 chars before chunk 1 ends)
    - Chunk 3: characters 900 → 1400
    ...so context is never lost at boundaries.

    Args:
        text:       The full extracted text from the PDF
        chunk_size: How many characters per chunk (500 is a good starting point)
        overlap:    How many characters to repeat between chunks (prevents cutoffs)

    Returns:
        A list of strings, each being one chunk
    """
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]   # Python slicing: text[start:end] extracts a substring
        chunks.append(chunk)
        start += chunk_size - overlap  # Move forward, but back up by 'overlap' chars

    return chunks


# ============================================================
# STEP 4 — EMBEDDING + STORING IN VECTOR DATABASE
# ============================================================

def store_document(pdf_path: str, user_id: str, doc_name: str) -> int:
    """
    The full pipeline: PDF → text → chunks → embeddings → ChromaDB

    DETAILED FLOW:
    1. Extract text from PDF using fitz
    2. Split text into chunks
    3. Convert each chunk into a vector (embedding) using SentenceTransformer
    4. Store (chunk_text, embedding, metadata) in ChromaDB

    ChromaDB concepts:
    - Collection: Like a table in a database. Each user gets their own collection.
    - Document: The actual text chunk we're storing
    - Embedding: The vector representation of that chunk
    - Metadata: Extra info stored alongside (like page/source name)
    - ID: A unique identifier for each chunk

    Args:
        pdf_path:  Path to the uploaded PDF file
        user_id:   Firebase user ID — keeps each user's data separate
        doc_name:  A friendly name for this document (e.g., "Physics Chapter 3")

    Returns:
        Number of chunks stored (useful for showing progress in Streamlit)
    """

    # Step 1: Extract all text from PDF
    print(f"📄 Extracting text from {doc_name}...")
    text = extract_text_from_pdf(pdf_path)

    if not text.strip():
        raise ValueError("No text found in PDF. It might be a scanned image PDF.")

    # Step 2: Split into chunks
    print(f"✂️  Splitting into chunks...")
    chunks = chunk_text(text)
    print(f"   Created {len(chunks)} chunks")

    # Step 3: Create embeddings for ALL chunks at once
    # embedding_model.encode() takes a list of strings and returns a numpy array
    # Each row in the array is a 384-dimensional vector for one chunk
    print(f"🔢 Generating embeddings...")
    embeddings = embedding_model.encode(chunks)
    # embeddings.tolist() converts numpy array → plain Python list (ChromaDB needs this)
    embeddings_list = embeddings.tolist()

    # Step 4: Get or create a ChromaDB collection for this user
    # Collection name format: "studypilot_<user_id>" — keeps users isolated
    # get_or_create_collection() — creates if doesn't exist, returns existing if it does
    collection_name = f"studypilot_{user_id[:20]}"  # ChromaDB has name length limits
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        # metadata tells ChromaDB which similarity metric to use
        # "cosine" measures the angle between vectors — best for text similarity
        metadata={"hnsw:space": "cosine"}
    )

    # Step 5: Store each chunk with its embedding and metadata
    # collection.add() takes parallel lists: documents, embeddings, metadatas, ids
    # They must all be the same length — each index represents one chunk
    ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": doc_name, "chunk_index": i} for i in range(len(chunks))]

    # upsert() = update if exists + insert if new (safer than add() which errors on duplicates)
    collection.upsert(
        documents=chunks,           # The actual text chunks
        embeddings=embeddings_list, # Their vector representations
        metadatas=metadatas,        # Extra info (source file, chunk number)
        ids=ids                     # Unique IDs for each chunk
    )

    print(f"✅ Stored {len(chunks)} chunks from '{doc_name}' in vector DB")
    return len(chunks)


# ============================================================
# STEP 5 — RETRIEVAL (Find relevant chunks for a query)
# ============================================================

def retrieve_relevant_chunks(query: str, user_id: str, top_k: int = 5) -> List[Dict]:
    """
    Given a question/topic, finds the most relevant chunks from the vector DB.

    HOW SIMILARITY SEARCH WORKS:
    1. Convert the query into a vector using the same embedding model
    2. ChromaDB computes cosine similarity between query vector and ALL stored vectors
    3. Returns the top_k most similar chunks

    Cosine similarity = 1 means identical meaning, 0 means completely unrelated

    Args:
        query:   The question or topic to search for (e.g., "Newton's laws of motion")
        user_id: Firebase user ID — search only THIS user's documents
        top_k:   How many chunks to return (5 is usually enough for good context)

    Returns:
        List of dicts, each with 'text', 'source', and 'score'
    """
    collection_name = f"studypilot_{user_id[:20]}"

    # Check if this user has any documents stored
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception:
        # get_collection() raises an error if collection doesn't exist
        print("No documents found for this user. Please upload a PDF first.")
        return []

    # Convert the query into a vector (same model used during storage — MUST match)
    query_embedding = embedding_model.encode([query]).tolist()

    # collection.query() — performs the similarity search
    # query_embeddings: the vector to search for
    # n_results: how many top results to return
    # include: what info to return alongside the results
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k, collection.count()),  # Can't ask for more than we have
        include=["documents", "metadatas", "distances"]
    )

    # results is a dict with parallel lists — index 0 = best match, etc.
    # results["documents"][0] → list of chunk texts
    # results["metadatas"][0] → list of metadata dicts
    # results["distances"][0] → list of distances (lower = more similar for cosine)

    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "score": 1 - results["distances"][0][i]  # Convert distance → similarity score
        })

    return chunks


# ============================================================
# STEP 6 — GENERATION (Call Llama via Groq with retrieved context)
# ============================================================

def call_llama_with_context(system_prompt: str, user_prompt: str) -> str:
    """
    Calls the Llama model via Groq API.

    This is the 'G' in RAG — Generation.
    We pass a system prompt (instructions) and user prompt (question + retrieved context).

    groq_client.chat.completions.create() — same API style as OpenAI
    model: "llama3-8b-8192" — 8B parameter Llama 3, 8192 token context window
    messages: a list of role/content dicts (standard chat format)
    temperature: 0 = deterministic (good for quizzes), 1 = creative (good for summaries)

    Args:
        system_prompt: Instructions for the model (what role it should play)
        user_prompt:   The actual question + retrieved context chunks

    Returns:
        The model's response as a plain string
    """
    response = groq_client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.3,   # Low temperature = more focused, less random
        max_tokens=2048    # Maximum length of the response
    )

    # response.choices[0].message.content — standard way to extract text from API response
    return response.choices[0].message.content


# ============================================================
# STEP 7 — LIST & DELETE (for Streamlit UI management)
# ============================================================

def list_user_documents(user_id: str) -> List[str]:
    """
    Returns a list of document names the user has uploaded.
    Useful for showing "Your uploaded documents: [Physics Ch3, Math Notes]" in Streamlit.
    """
    collection_name = f"studypilot_{user_id[:20]}"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        # collection.get() with no args returns ALL stored items
        results = collection.get(include=["metadatas"])
        # Extract unique source names from metadata
        sources = list(set(m["source"] for m in results["metadatas"]))
        return sources
    except Exception:
        return []


def delete_user_document(user_id: str, doc_name: str) -> bool:
    """
    Deletes all chunks belonging to a specific document.
    Useful if user wants to remove a document and re-upload.

    collection.delete() with where filter — deletes only chunks matching the filter
    """
    collection_name = f"studypilot_{user_id[:20]}"
    try:
        collection = chroma_client.get_collection(name=collection_name)
        # where= is a filter: only delete chunks where metadata "source" == doc_name
        collection.delete(where={"source": doc_name})
        return True
    except Exception:
        return False
