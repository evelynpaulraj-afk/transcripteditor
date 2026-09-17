def segment_overlaps_shot(segment, shot):

    return (
        segment["start"] < shot["end"]
        and segment["end"] > shot["start"]
    )


def align_segments_to_shots(segments, shots):

    aligned = []

    for shot in shots:

        matching_segments = [
            segment
            for segment in segments
            if segment_overlaps_shot(segment, shot)
        ]

        aligned.append({
            "shot_id": shot["shot_id"],
            "start": shot["start"],
            "end": shot["end"],
            "segments": matching_segments
        })

    return aligned