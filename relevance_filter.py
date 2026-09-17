from sentence_transformers import SentenceTransformer, util

_model = None


def get_embedder():
    global _model

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def filter_by_relevance(
    segments: list,
    brief: str,
    keep_ratio: float = 0.5,
    neighbor_window: int = 1
) -> tuple[list, list]:

    """
    Selects transcript segments relevant to the user's editing request.

    IMPORTANT:
    - Original transcript text is never modified.
    - Timestamp information is preserved.
    - Neighboring sentences are included to preserve context.
    """

    if len(segments) < 8:
        return segments, []

    model = get_embedder()

    brief_embedding = model.encode(
        brief,
        convert_to_tensor=True
    )

    texts = [s["text"] for s in segments]

    segment_embeddings = model.encode(
        texts,
        convert_to_tensor=True
    )

    similarities = util.cos_sim(
        brief_embedding,
        segment_embeddings
    )[0]

    scored = list(
        zip(
            range(len(segments)),
            segments,
            similarities.tolist()
        )
    )

    scored.sort(
        key=lambda x: x[2],
        reverse=True
    )

    keep_count = max(
        8,
        int(len(segments) * keep_ratio)
    )

    selected_indices = set()

    # Select the most relevant sentences
    for index, segment, score in scored[:keep_count]:

        selected_indices.add(index)

        # Add nearby sentences for continuity
        for offset in range(
            -neighbor_window,
            neighbor_window + 1
        ):

            neighbor_index = index + offset

            if 0 <= neighbor_index < len(segments):
                selected_indices.add(neighbor_index)

    # Restore chronological order
    kept = [
        segments[i]
        for i in sorted(selected_indices)
    ]

    kept_ids = {
        s["id"]
        for s in kept
    }

    dropped = [
        s
        for s in segments
        if s["id"] not in kept_ids
    ]

    return kept, dropped