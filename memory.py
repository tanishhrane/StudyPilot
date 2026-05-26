# memory.py
import chromadb
from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid

# ── Init ──────────────────────────────────────
embedder = SentenceTransformer("all-MiniLM-L6-v2")

chroma_client = chromadb.Client()

collection = chroma_client.get_or_create_collection(
    name="chat_memory",
    metadata={"hnsw:space": "cosine"}
)

# ── Save message ──────────────────────────────
def save_message(role: str, content: str,
                 session_id: str = "default",
                 weak_topics: list = None):

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

# ── Retrieve relevant messages ─────────────────
def get_last_messages(query: str = "study session",
                      session_id: str = "default",
                      limit: int = 5) -> list[dict]:

    query_vector = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[query_vector],
        n_results=limit,
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

# ── Get weak topics ────────────────────────────
def get_weak_topics(session_id: str = "default") -> list[str]:

    results = collection.query(
        query_embeddings=[
            embedder.encode("quiz wrong incorrect weak").tolist()
        ],
        n_results=10,
        where={"session_id": session_id}
    )

    weak = []
    for meta in results["metadatas"][0]:
        topics = meta.get("weak_topics", "")
        if topics:
            weak.extend(topics.split(","))

    return list(set(filter(None, weak)))