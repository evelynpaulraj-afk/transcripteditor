def build_context(
    brief,
    transcript_segments,
    aligned_shots
):

    lines = []

    lines.append(
        "USER EDITING REQUEST:"
    )

    lines.append(brief)

    lines.append(
        "\nEXACT TRANSCRIPT:"
    )

    for segment in transcript_segments:

        lines.append(
            f"[{segment['start']:.2f}-"
            f"{segment['end']:.2f}] "
            f"{segment['text']}"
        )

    lines.append(
        "\nVISUAL CONTEXT:"
    )

    transcript_ids = {
        s["id"]
        for s in transcript_segments
    }

    for shot in aligned_shots:

        related = [
            s
            for s in shot["segments"]
            if s["id"] in transcript_ids
        ]

        if not related:
            continue

        lines.append(
            f"\nSHOT {shot['shot_id']} "
            f"[{shot['start']:.2f}-"
            f"{shot['end']:.2f}]"
        )

        if "visual" in shot:

            visual = shot["visual"]

            lines.append(
                f"Visual: {visual}"
            )

    return "\n".join(lines)