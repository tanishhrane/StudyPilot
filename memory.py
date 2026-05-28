import os
import pickle
import uuid

import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from datetime import datetime

# ==========================================
# CONSTANTS
# ==========================================

EMBEDDING_DIM = 384
INDEX_PATH    = "faiss_index.bin"
META_PATH     = "faiss_metadata.pkl"

# ==========================================
# INIT EMBEDDER
# ==========================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# LOAD OR CREATE FAISS INDEX
# ==========================================

def _load_index():

    if (
        os.path.exists(INDEX_PATH)
        and
        os.path.exists(META_PATH)
    ):
        index = faiss.read_index(INDEX_PATH)

        with open(META_PATH, "rb") as f:
            metadata = pickle.load(f)

    else:
        index    = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []

    return index, metadata


def _save_index(index, metadata):

    faiss.write_index(index, INDEX_PATH)

    with open(META_PATH, "wb") as f:
        pickle.dump(metadata, f)


# Load once at module level
_index, _metadata = _load_index()


# ==========================================
# EMBED HELPER
# ==========================================

def _embed(text: str) -> np.ndarray:

    vector = embedder.encode(
        text,
        normalize_embeddings=True
    )

    return vector.astype("float32").reshape(1, -1)


# ==========================================
# SAVE MESSAGE
# ==========================================

def save_message(
    role,
    content,
    session_id="default",
    weak_topics=None,
    topic=None          # ✅ stores which quiz topic produced weak topics
):

    global _index, _metadata

    vector = _embed(content)

    _index.add(vector)

    _metadata.append({
        "id":          str(uuid.uuid4()),
        "role":        role,
        "content":     content,
        "session_id":  session_id,
        "weak_topics": weak_topics or [],
        "topic":       topic,             # ✅ new field
        "timestamp":   datetime.now().isoformat()
    })

    _save_index(_index, _metadata)


# ==========================================
# GET LAST MESSAGES
# ==========================================

def get_last_messages(
    query="study session",
    session_id="default",
    limit=5
):

    global _index, _metadata

    if _index.ntotal == 0:
        return []

    query_vector = _embed(query)

    k = min(limit * 4, _index.ntotal)

    scores, indices = _index.search(
        query_vector, k
    )

    results = []

    for idx, score in zip(indices[0], scores[0]):

        if idx == -1:
            continue

        entry = _metadata[idx]

        if entry["session_id"] != session_id:
            continue

        results.append({
            "content":    entry["content"],
            "role":       entry["role"],
            "session_id": entry["session_id"],
            "score":      float(score)
        })

        if len(results) == limit:
            break

    return results


# ==========================================
# GET WEAK TOPICS (topic-scoped) ✅ UPDATED
# ==========================================

def get_weak_topics(session_id="default", topic=None):  # ✅ topic param added

    global _index, _metadata

    if _index.ntotal == 0:
        return []

    # ✅ topic-aware query
    query_text = (
        f"{topic} weak incorrect subtopic"
        if topic
        else "quiz wrong incorrect weak"
    )

    query_vector = _embed(query_text)

    k = min(20, _index.ntotal)

    scores, indices = _index.search(
        query_vector, k
    )

    weak = []

    for idx, score in zip(indices[0], scores[0]):

        if idx == -1:
            continue

        entry = _metadata[idx]

        if entry["session_id"] != session_id:
            continue

        # ✅ Only include weak topics from matching quiz topic
        if topic and entry.get("topic") != topic:
            continue

        weak.extend(entry.get("weak_topics", []))

    return list(set(filter(None, weak)))