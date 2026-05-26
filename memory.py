from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

from sentence_transformers import SentenceTransformer
from datetime import datetime
import uuid

# ==========================================
# EMBEDDING MODEL
# ==========================================

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

# ==========================================
# QDRANT CLIENT
# ==========================================

client = QdrantClient(":memory:")

COLLECTION = "chat_memory"

# ==========================================
# CREATE COLLECTION
# ==========================================

existing_collections = [
    collection.name
    for collection in client.get_collections().collections
]

if COLLECTION not in existing_collections:

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )

# ==========================================
# SAVE MESSAGE
# ==========================================

def save_message(
    role: str,
    content: str,
    session_id: str = "default",
    topic_tags: list = None,
    weak_topics: list = None
):

    vector = embedder.encode(content).tolist()

    payload = {
        "role": role,
        "content": content,
        "session_id": session_id,
        "topic_tags": topic_tags or [],
        "weak_topics": weak_topics or [],
        "timestamp": datetime.now().isoformat()
    }

    client.upsert(
        collection_name=COLLECTION,
        points=[
            PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=payload
            )
        ]
    )

# ==========================================
# GET RELEVANT MESSAGES
# ==========================================

def get_last_messages(
    query: str,
    session_id: str = "default",
    limit: int = 5
):

    query_vector = embedder.encode(
        query
    ).tolist()

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=limit
    )

    filtered_results = []

    for result in results:

        payload = result.payload

        if payload.get("session_id") == session_id:

            filtered_results.append(payload)

    return filtered_results

# ==========================================
# GET WEAK TOPICS
# ==========================================

def get_weak_topics(
    session_id: str = "default"
):

    query_vector = embedder.encode(
        "quiz wrong incorrect weak"
    ).tolist()

    results = client.search(
        collection_name=COLLECTION,
        query_vector=query_vector,
        limit=10
    )

    weak_topics = []

    for result in results:

        payload = result.payload

        if payload.get("session_id") == session_id:

            weak_topics.extend(
                payload.get(
                    "weak_topics",
                    []
                )
            )

    return list(set(weak_topics))