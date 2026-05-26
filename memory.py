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

client = QdrantClient(
    path="qdrant_data"
)

COLLECTION = "chat_memory"

# ==========================================
# CREATE COLLECTION
# ==========================================

collections = client.get_collections().collections

collection_names = [
    c.name
    for c in collections
]

if COLLECTION not in collection_names:

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
    role,
    content,
    session_id="default"
):

    vector = embedder.encode(
        content
    ).tolist()

    payload = {
        "role": role,
        "content": content,
        "session_id": session_id,
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
    query,
    session_id="default",
    limit=5
):

    query_vector = embedder.encode(
        query
    ).tolist()

    hits = client.search_batch(
        collection_name=COLLECTION,
        requests=[
            {
                "vector": query_vector,
                "limit": limit
            }
        ]
    )[0]

    results = []

    for hit in hits:

        payload = hit.payload

        if payload.get(
            "session_id"
        ) == session_id:

            results.append(payload)

    return results

# ==========================================
# GET WEAK TOPICS
# ==========================================

def get_weak_topics(
    session_id="default"
):

    query_vector = embedder.encode(
        "weak incorrect wrong quiz"
    ).tolist()

    hits = client.search_batch(
        collection_name=COLLECTION,
        requests=[
            {
                "vector": query_vector,
                "limit": 10
            }
        ]
    )[0]

    weak_topics = []

    for hit in hits:

        payload = hit.payload

        if payload.get(
            "session_id"
        ) == session_id:

            weak_topics.extend(
                payload.get(
                    "weak_topics",
                    []
                )
            )

    return list(
        set(weak_topics)
    )