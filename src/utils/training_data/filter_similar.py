"""Embedding-based deduplication using FAISS."""

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss


def dedup_by_cosine_similarity(
    texts: list[str],
    threshold: float = 0.91,
    k: int = 10,
    model_name: str = 'all-MiniLM-L6-v2',
) -> tuple[set[int], list[tuple[int, int, float]]]:
    """Return (indices_to_remove, duplicate_pairs) using embedding + FAISS dedup."""
    model = SentenceTransformer(model_name)
    print(f"Encoding {len(texts)} texts with {model_name}...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True)

    embeddings_norm = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings_norm)
    index = faiss.IndexFlatIP(embeddings_norm.shape[1])
    index.add(embeddings_norm)
    similarities, indices = index.search(embeddings_norm, k)

    duplicate_pairs = []
    for i, (sims, idxs) in enumerate(zip(similarities, indices)):
        for sim, j in zip(sims, idxs):
            if i < j and sim > threshold:
                duplicate_pairs.append((i, j, float(sim)))

    indices_to_remove = set(j for i, j, sim in duplicate_pairs)
    print(f"Found {len(duplicate_pairs)} duplicate pairs, {len(indices_to_remove)} indices to remove")
    return indices_to_remove, duplicate_pairs
