# llm_select.py
import json
import sys
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

MODEL = "gemini-3.6-flash"


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


# ---------- PASS 0: group segments into coherent "beats" ----------

def build_grouping_prompt(segments: list) -> str:
    compact = [{"id": s["id"], "text": s["text"]} for s in segments]
    return f"""Below is a transcript, broken into individual sentences, in original spoken order.

Group these sentences into "beats" — each beat is one complete, self-contained thought or point.

EXAMPLE of correct grouping:
Input sentences:
  seg-A: "The first thing I want to talk about is how transcription works."
  seg-B: "It takes a video file and runs it through Whisper to get timestamps."
  seg-C: "Um, yeah, so like, the way it works is you basically take the video and it listens to it."
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
{json.dumps(compact, indent=2)}
"""


def group_into_beats(segments: list) -> list:
    result = call_model(build_grouping_prompt(segments))
    valid_ids = {s["id"] for s in segments}
    beats = []
    seen = set()
    for b in result["beats"]:
        ids = [i for i in b["ids"] if i in valid_ids and i not in seen]
        if ids:
            beats.append(ids)
            seen.update(ids)
    # Defensive: any id the model dropped becomes its own beat, appended in original order
    for s in segments:
        if s["id"] not in seen:
            beats.append([s["id"]])
            seen.add(s["id"])
    return beats


# ---------- PASS 1: select & reorder whole beats ----------

def build_selection_prompt(beats: list, segment_by_id: dict, brief: str, target_length, allow_reorder: bool) -> str:
    compact = []
    for i, ids in enumerate(beats):
        text = " ".join(segment_by_id[i2]["text"] for i2 in ids)
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

Respond with ONLY: {{"final_sequence": ["beat-002", "beat-000"]}}

BEATS:
{json.dumps(compact, indent=2)}
"""


def select_segments(project: dict, brief: str, target_length=None, allow_reorder: bool = False) -> dict:
    segments = project["segments"]
    segment_by_id = {s["id"]: s for s in segments}

    beats = group_into_beats(segments)
    print(f"Grouped {len(segments)} sentences into {len(beats)} beats")

    result = call_model(build_selection_prompt(beats, segment_by_id, brief, target_length, allow_reorder))
    beat_by_label = {f"beat-{i:03d}": ids for i, ids in enumerate(beats)}
    chosen_labels = [b for b in result["final_sequence"] if b in beat_by_label]

    final_sequence = []
    for label in chosen_labels:
        final_sequence.extend(beat_by_label[label])  # expand beat -> its member segment ids, still in order

    for seg in segments:
        seg["included"] = False
    for new_order, sid in enumerate(final_sequence):
        segment_by_id[sid]["included"] = True
        segment_by_id[sid]["order"] = new_order

    kept_duration = sum(segment_by_id[sid]["end"] - segment_by_id[sid]["start"] for sid in final_sequence)
    print(f"Kept {len(chosen_labels)}/{len(beats)} beats -> {len(final_sequence)} segments (~{kept_duration:.0f}s)")
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