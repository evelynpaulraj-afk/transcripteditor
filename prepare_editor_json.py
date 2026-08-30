import json
import sys

def prepare_for_editor(whisper_json_path: str, video_path: str, output_path: str):
    with open(whisper_json_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    editor_segments = []
    for i, seg in enumerate(raw["segments"]):
        editor_segments.append({
            "id": f"seg-{i:04d}",
            "order": i,          # starts in chronological order
            "included": True,    # nothing cut yet
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"],
            "words": seg.get("words", []),
        })

    project = {
        "video_path": video_path,
        "segments": editor_segments,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(editor_segments)} editable segments to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python prepare_editor_json.py <transcript.json> <video_path> <project.json>")
        sys.exit(1)
    prepare_for_editor(sys.argv[1], sys.argv[2], sys.argv[3])