# resegment_sentences.py
import json
import sys

SENTENCE_ENDERS = {".", "!", "?"}


def load_transcript(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resegment_by_sentence(transcript: dict) -> dict:
    # Flatten every word across all original Whisper segments into one stream —
    # we're discarding Whisper's original segment boundaries entirely and
    # rebuilding our own based on punctuation.
    all_words = []
    for seg in transcript["segments"]:
        all_words.extend(seg.get("words", []))

    if not all_words:
        raise ValueError("No word-level timestamps found — re-run transcribe.py with word_timestamps=True")

    sentences = []
    current_words = []

    for w in all_words:
        current_words.append(w)
        stripped = w["word"].strip()
        if stripped and stripped[-1] in SENTENCE_ENDERS:
            sentences.append(current_words)
            current_words = []

    if current_words:  # trailing words with no terminal punctuation
        sentences.append(current_words)

    new_segments = []
    for i, words in enumerate(sentences):
        text = "".join(w["word"] for w in words).strip()
        new_segments.append({
            "id": f"seg-{i:04d}",
            "start": round(words[0]["start"], 2),
            "end": round(words[-1]["end"], 2),
            "text": text,
            "words": words,
        })

    return {"segments": new_segments}


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python resegment_sentences.py <transcript.json> <sentences.json>")
        sys.exit(1)

    transcript = load_transcript(sys.argv[1])
    resegmented = resegment_by_sentence(transcript)
    with open(sys.argv[2], "w", encoding="utf-8") as f:
        json.dump(resegmented, f, indent=2, ensure_ascii=False)
    print(f"Rebuilt {len(resegmented['segments'])} sentence-level segments")