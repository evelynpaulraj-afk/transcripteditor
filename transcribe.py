from faster_whisper import WhisperModel
import json
import sys

#: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

def transcribe(video_path: str, output_path: str, model_size: str = "medium"):
    # device="cpu" works everywhere; switch to "cuda" if you have an NVIDIA GPU
    # compute_type controls speed/accuracy tradeoff: "int8" is fastest on CPU
    model = WhisperModel(model_size, device="cpu", compute_type="int8")

    segments, info = model.transcribe(
        video_path,
        word_timestamps=True,   # gives per-word start/end, not just per-segment
        vad_filter=True,        # skips silence, avoids hallucinated text on dead air
    )

    print(f"Detected language: {info.language} (p={info.language_probability:.2f})")

    output = {"segments": []}
    for seg in segments:
        segment_data = {
            "id": seg.id,
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
            "words": [
                {"word": w.word, "start": round(w.start, 2), "end": round(w.end, 2)}
                for w in (seg.words or [])
            ],
        }
        output["segments"].append(segment_data)
        print(f"[{segment_data['start']:>7.2f}s -> {segment_data['end']:>7.2f}s] {segment_data['text']}")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\nSaved transcript to {output_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python transcribe.py test.mp4 [output_json]")
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "transcript.json"
    transcribe(video_path, output_path)