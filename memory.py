import os
import pickle
import uuid
import numpy as np
import faiss

from sentence_transformers import SentenceTransformer
from datetime import datetime

# ==========================================
# INIT
# ==========================================

EMBEDDING_DIM   = 384
INDEX_PATH      = "faiss_index.bin"
METADATA_PATH   = "faiss_metadata.pkl"

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ==========================================
# LOAD OR CREATE INDEX
# ==========================================

def _load_index():
    if (
        os.path.exists(INDEX_PATH)
        and
        os.path.exists(METADATA_PATH)
    ):
        index = faiss.read_index(INDEX_PATH)

        with open(METADATA_PATH, "rb") as f:
            metadata = pickle.load(f)

    else:
        # IndexFlatIP = cosine similarity
        # (with normalized vectors)
        index    = faiss.IndexFlatIP(EMBEDDING_DIM)
        metadata = []   # list of dicts, one per vector

    return index, metadata


def _save_index(index, metadata):
    faiss.write_index(index, INDEX_PATH)

    with open(METADATA_PATH, "wb") as f:
        pickle.dump(metadata, f)


# Load once at module level
_index, _metadata = _load_index()


# ==========================================
# EMBED HELPER
# ==========================================

def _embed(text: str) -> np.ndarray:
    vector = embedder.encode(
        text,
        normalize_embeddings=True   # required for cosine via IP
    )
    return vector.astype("float32").reshape(1, -1)


# ==========================================
# SAVE MESSAGE
# ==========================================

def save_message(
    role,
    content,
    session_id="default",
    weak_topics=None
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

    # Safety: nothing stored yet
    if _index.ntotal == 0:
        return []

    query_vector = _embed(query)

    # Search top-k across all vectors
    k = min(limit * 4, _index.ntotal)   # oversample then filter

    scores, indices = _index.search(query_vector, k)

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
# GET WEAK TOPICS
# ==========================================

def get_weak_topics(session_id="default"):
    global _index, _metadata

    if _index.ntotal == 0:
        return []

    query_vector = _embed(
        "quiz wrong incorrect weak"
    )

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

        weak.extend(entry.get("weak_topics", []))

    return list(set(filter(None, weak)))