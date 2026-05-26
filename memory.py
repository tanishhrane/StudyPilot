# memory.py
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
