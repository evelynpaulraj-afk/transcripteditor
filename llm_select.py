# llm_select.py
import json
import sys
import os
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = "gemini-3.6-flash"
NUM_VOTES = 3  # self-consistency: how many times to run the selection pass


def load_project(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)


def extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text[text.find("{"):text.rfind("}") + 1])


def call_model(prompt: str) -> dict:
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    return extract_json(response.choices[0].message.content)


def estimate_tokens(text: str) -> int:
    # Rough heuristic (~4 chars/token) — good enough for a relative before/after comparison,
    # not meant to match the provider's exact tokenizer.
    return max(1, len(text) // 4)


def compact_json(obj) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


# ---------- PASS -1: LLM-based compression (runs once per video, cached in project.json) ----------
# Instead of a hand-maintained filler-word list, the model itself decides what's disposable —
# consistent with letting the LLM own every other editorial judgment in this pipeline.

def build_compression_prompt(segments: list) -> str:
    compact = [{"id": s["id"], "text": s["text"]} for s in segments]
    return f"""Below are transcript sentences from spoken, unscripted speech. For each one, produce a
compressed version by removing disfluencies, filler words, and false starts — anything that
carries no information and exists only because it's spontaneous speech.

Keep all technical content, meaning, and factual detail fully intact. Do NOT paraphrase, do NOT
summarize, do NOT change word choice for style — only remove words that add zero information.
If a sentence has nothing removable, return it unchanged.

Example:
  Input:  "Um, yeah, so like, the way it works is you basically, you take the video, right, and it,
           uh, it listens to it and it kind of figures out what words are being said."
  Output: "The way it works is you take the video and it listens to it and figures out what words
           are being said."

Respond with ONLY:
{{"compressed": [{{"id": "seg-0000", "text": "..."}}, {{"id": "seg-0001", "text": "..."}}]}}

SENTENCES:
{compact_json(compact)}
"""


def compress_segments_with_llm(segments: list) -> None:
    """Mutates segments in place, adding a 'text_compressed' field. Skips segments that already
    have one cached, so re-running on the same project.json doesn't waste API calls."""
    to_compress = [s for s in segments if "text_compressed" not in s]
    if not to_compress:
        print("All segments already have cached compression — skipping LLM compression pass")
        return

    prompt = build_compression_prompt(to_compress)
    orig_size = estimate_tokens(compact_json([{"id": s["id"], "text": s["text"]} for s in to_compress]))

    result = call_model(prompt)
    by_id = {s["id"]: s for s in to_compress}
    valid_ids = set(by_id.keys())
    seen = set()
    warnings = []

    for entry in result.get("compressed", []):
        sid = entry.get("id")
        new_text = entry.get("text", "").strip()
        if sid not in valid_ids or not new_text:
            continue
        original = by_id[sid]["text"]
        # Sanity check: flag (but don't reject) suspiciously aggressive compression,
        # since that could indicate the model dropped real content, not just filler.
        if len(new_text) < len(original) * 0.4:
            warnings.append((sid, original, new_text))
        by_id[sid]["text_compressed"] = new_text
        seen.add(sid)

    # Defensive: anything the model skipped just falls back to its own original text
    for s in to_compress:
        if s["id"] not in seen:
            s["text_compressed"] = s["text"]

    comp_size = estimate_tokens(compact_json([{"id": s["id"], "text": s["text_compressed"]} for s in to_compress]))
    saved_pct = 100 * (orig_size - comp_size) / orig_size if orig_size else 0
    print(f"LLM compression: ~{orig_size} -> ~{comp_size} tokens ({saved_pct:.0f}% reduction) across {len(to_compress)} sentences")

    if warnings:
        print(f"Compression sanity check: {len(warnings)} sentence(s) shrank by more than 60% — review these:")
        for sid, orig, comp in warnings[:5]:
            print(f"  [{sid}] BEFORE: {orig}")
            print(f"  [{sid}] AFTER:  {comp}")


def debug_compression_sample(segments: list, n: int = 5):
    """Print before/after for a few sentences that actually changed, so you can eyeball quality."""
    print("\n--- Compression debug sample ---")
    shown = 0
    for s in segments:
        original = s["text"]
        compressed = s.get("text_compressed", original)
        if original != compressed:
            print(f"  BEFORE: {original}")
            print(f"  AFTER:  {compressed}")
            print()
            shown += 1
        if shown >= n:
            break
    if shown == 0:
        print("  No sentences were changed by compression at all.")
    print("--- end sample ---\n")


# ---------- PASS 0: group segments into coherent "beats" ----------

def build_grouping_prompt(segments: list) -> str:
    compact = [{"id": s["id"], "text": s["text_compressed"]} for s in segments]
    return f"""Below is a transcript, broken into individual sentences, in original spoken order.

Group these sentences into "beats" — each beat is one complete, self-contained thought or point.

EXAMPLE of correct grouping:
Input sentences:
  seg-A: "The first thing I want to talk about is how transcription works."
  seg-B: "It takes a video file and runs it through Whisper to get timestamps."
  seg-C: "The way it works is you take the video and it listens to it."
  seg-D: "Oh random side note, I had coffee this morning."
Correct beats: [["seg-A", "seg-B"], ["seg-C"], ["seg-D"]]
Why: A+B are one beat because B is the payoff for A's setup. C is its OWN beat, separate from
A+B, even though it restates a similar idea — it's a distinct, independently-cuttable attempt at
explaining the same thing, not part of the same sentence flow. D is unrelated, its own beat.

Rules:
- A setup sentence MUST be grouped with the sentence(s) that deliver its explanation.
- A filler restatement of an already-explained point is its OWN separate beat, never merged into
  the beat it's restating — this keeps it independently cuttable.
- Every id must appear in exactly one beat. Beats must contain only CONSECUTIVE ids.
- Do not reorder or omit anything here — that happens in a later step.

Respond with ONLY:
{{"beats": [{{"ids": ["seg-0000", "seg-0001"]}}, {{"ids": ["seg-0002"]}}]}}

TRANSCRIPT:
{compact_json(compact)}
"""


def group_into_beats(segments: list) -> list:
    prompt = build_grouping_prompt(segments)
    result = call_model(prompt)
    valid_ids = {s["id"] for s in segments}
    beats = []
    seen = set()
    invalid_count = 0

    for b in result["beats"]:
        raw_ids = b["ids"]
        ids = [i for i in raw_ids if i in valid_ids and i not in seen]
        invalid_count += len(raw_ids) - len(ids)
        if ids:
            beats.append(ids)
            seen.update(ids)

    for s in segments:
        if s["id"] not in seen:
            beats.append([s["id"]])
            seen.add(s["id"])

    if invalid_count:
        print(f"  Hallucination check: {invalid_count} invalid/duplicate id(s) in grouping pass")

    return beats


# ---------- PASS 1: select & reorder whole beats (with reasoning + self-consistency) ----------

def build_selection_prompt(beats: list, segment_by_id: dict, brief: str, target_length, allow_reorder: bool) -> str:
    compact = []
    for i, ids in enumerate(beats):
        text = " ".join(segment_by_id[i2]["text_compressed"] for i2 in ids)
        duration = sum(segment_by_id[i2]["end"] - segment_by_id[i2]["start"] for i2 in ids)
        compact.append({"beat_id": f"beat-{i:03d}", "duration_sec": round(duration, 2), "text": text})

    total_duration = sum(b["duration_sec"] for b in compact)
    length_instruction = (
        f"Target final length: ~{target_length}s. Full footage is {total_duration:.0f}s."
        if target_length else f"Full footage is {total_duration:.0f}s. Keep only what's relevant."
    )
    reorder_instruction = (
        "You MAY reorder beats for a clearer or more compelling story."
        if allow_reorder else "Keep beats in original relative order."
    )

    return f"""You are an expert video editor. Each "beat" below is already a complete, indivisible
thought — treat it as a single unit. Decide which beats to KEEP and in what SEQUENCE.

BRIEF: "{brief}"
{length_instruction}
{reorder_instruction}

Rules:
- Cut filler, tangents, and repeated points, unless the brief says otherwise.
- You may not split a beat apart — keep or cut it entirely, as one block.
- For each beat you keep, give a short reason (under 12 words) grounded in its actual content —
  not a generic justification. This is a coherence check, not decoration.

Respond with ONLY:
{{"final_sequence": [
  {{"beat_id": "beat-002", "reason": "delivers the concrete mechanism the brief asks to lead with"}},
  {{"beat_id": "beat-000", "reason": "..."}}
]}}

BEATS:
{compact_json(compact)}
"""


def run_selection_once(beats, segment_by_id, brief, target_length, allow_reorder, beat_by_label):
    prompt = build_selection_prompt(beats, segment_by_id, brief, target_length, allow_reorder)
    result = call_model(prompt)
    entries = result.get("final_sequence", [])
    chosen = [(e["beat_id"], e.get("reason", "")) for e in entries if e.get("beat_id") in beat_by_label]
    return chosen


def select_segments(project: dict, brief: str, target_length=None, allow_reorder: bool = False) -> dict:
    segments = project["segments"]
    segment_by_id = {s["id"]: s for s in segments}

    compress_segments_with_llm(segments)
    debug_compression_sample(segments)

    beats = group_into_beats(segments)
    print(f"Grouped {len(segments)} sentences into {len(beats)} beats")
    beat_by_label = {f"beat-{i:03d}": ids for i, ids in enumerate(beats)}

    # ---- Self-consistency: run selection NUM_VOTES times, keep majority-agreed beats ----
    print(f"Running selection pass {NUM_VOTES}x for self-consistency check...")
    run_results = []
    reasons_by_run = []
    for v in range(NUM_VOTES):
        chosen = run_selection_once(beats, segment_by_id, brief, target_length, allow_reorder, beat_by_label)
        run_results.append([label for label, _ in chosen])
        reasons_by_run.append({label: reason for label, reason in chosen})

    vote_counts = Counter()
    for run in run_results:
        for label in set(run):
            vote_counts[label] += 1

    majority_threshold = (NUM_VOTES // 2) + 1
    majority_beats = {label for label, count in vote_counts.items() if count >= majority_threshold}

    disagreements = {label: count for label, count in vote_counts.items() if count < NUM_VOTES}
    if disagreements:
        print("Self-consistency disagreements:")
        for label, count in sorted(disagreements.items()):
            verdict = "kept (majority)" if label in majority_beats else "cut (not majority)"
            print(f"  {label}: selected in {count}/{NUM_VOTES} runs -> {verdict}")
    else:
        print("Self-consistency check: all runs agreed on every beat")

    # Use the run whose order best covers the majority-approved beats
    best_run_idx = max(range(NUM_VOTES), key=lambda i: sum(1 for b in run_results[i] if b in majority_beats))
    final_labels = [b for b in run_results[best_run_idx] if b in majority_beats]
    final_reasons = reasons_by_run[best_run_idx]

    print("\nFinal selection with reasoning:")
    for label in final_labels:
        print(f"  {label}: {final_reasons.get(label, '(no reason recorded)')}")

    final_sequence = []
    for label in final_labels:
        final_sequence.extend(beat_by_label[label])

    for seg in segments:
        seg["included"] = False
    for new_order, sid in enumerate(final_sequence):
        segment_by_id[sid]["included"] = True
        segment_by_id[sid]["order"] = new_order

    kept_duration = sum(segment_by_id[sid]["end"] - segment_by_id[sid]["start"] for sid in final_sequence)
    print(f"\nKept {len(final_labels)}/{len(beats)} beats -> {len(final_sequence)} segments (~{kept_duration:.0f}s)")
    return project


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print('Usage: python llm_select.py <project.json> <output.json> "<brief>" [target_length_seconds] [--reorder]')
        sys.exit(1)
    project_path, output_path, brief = sys.argv[1], sys.argv[2], sys.argv[3]
    remaining = sys.argv[4:]
    allow_reorder = "--reorder" in remaining
    remaining = [a for a in remaining if a != "--reorder"]
    target_length = float(remaining[0]) if remaining else None

    project = load_project(project_path)
    updated = select_segments(project, brief, target_length, allow_reorder)
    save_project(updated, output_path)
    print(f"Saved decisions to {output_path}")