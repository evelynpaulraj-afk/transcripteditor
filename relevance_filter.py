# relevance_filter.py
from sentence_transformers import SentenceTransformer, util

_model = None

def get_embedder():
    global _model
    if _model is None:
        # Small, free, runs locally — no API cost, no rate limit
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def filter_by_relevance(segments: list, brief: str, keep_ratio: float = 0.7) -> tuple[list, list]:
    """
    Ranks segments by embedding similarity to the brief and returns
    (kept_segments, dropped_segments). Dropped segments never reach the LLM at all —
    they're marked included=False immediately, saving tokens on every downstream pass.
    """
    if len(segments) < 8:
        # Too few segments for filtering to be worth the risk — always run the full pipeline
        return segments, []

    model = get_embedder()
    brief_embedding = model.encode(brief, convert_to_tensor=True)
    texts = [s["text"] for s in segments]
    segment_embeddings = model.encode(texts, convert_to_tensor=True)

    similarities = util.cos_sim(brief_embedding, segment_embeddings)[0]

    scored = list(zip(segments, similarities.tolist()))
    scored.sort(key=lambda x: x[1], reverse=True)

    keep_count = max(8, int(len(segments) * keep_ratio))
    kept = [s for s, _ in scored[:keep_count]]
    dropped = [s for s, _ in scored[keep_count:]]

    # Restore original chronological order for what's kept — ranking was only for selection,
    # not for changing sequence
    kept_ids = {s["id"] for s in kept}
    kept_in_order = [s for s in segments if s["id"] in kept_ids]

    return kept_in_order, dropped