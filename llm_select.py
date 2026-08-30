# llm_select.py
import json
import sys
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama",
)

MODEL = "llama3.1:8b"


def load_project(project_path: str) -> dict:
    with open(project_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)


def build_prompt(segments: list, brief: str, target_length: float | None, allow_reorder: bool) -> str:
    compact = [
        {
            "id": s["id"],
            "duration_sec": round(s["end"] - s["start"], 2),
            "text": s["text"],
        }
        for s in segments
    ]

    total_duration = sum(s["duration_sec"] for s in compact)
    length_instruction = (
        f"The final edited video should be approximately {target_length} seconds long. "
        f"The full unedited footage is {total_duration:.0f} seconds long."
        if target_length
        else f"The full unedited footage is {total_duration:.0f} seconds long. Keep only what's relevant to the brief."
    )

    reorder_instruction = (
        "You MAY reorder segments if it tells a clearer or more compelling story than the "
        "original chronological order — for example, moving a strong point earlier, or grouping "
        "related ideas together even if they weren't spoken back-to-back."
        if allow_reorder
        else "Keep all kept segments in their original relative order — do not reorder them."
    )

    return f"""You are an expert video editor. You are given a timestamped transcript, broken into
sequential segments, from a piece of raw footage. Your job is to decide which segments to KEEP
and in what SEQUENCE they should play in the final edited video, based on the editor's brief below.

EDITOR'S BRIEF: "{brief}"

{length_instruction}

{reorder_instruction}

Rules:
- Cut filler, false starts, off-topic tangents, and repeated points, unless the brief says otherwise.
- Prefer keeping complete thoughts together — don't split a segment's idea awkwardly.
- Do not invent or alter text — you are only selecting and sequencing existing segment ids.

Respond with ONLY a JSON object in this exact shape, and nothing else — no explanation, no markdown fences:
{{"final_sequence": ["seg-0003", "seg-0000", "seg-0007"]}}

This list should contain ONLY the ids you are keeping, in the exact order they should play.
Every id you omit is treated as cut.

TRANSCRIPT SEGMENTS:
{json.dumps(compact, indent=2)}
"""


def extract_json(raw_text: str) -> dict:
    text = raw_text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    start = text.find("{")
    end = text.rfind("}")
    return json.loads(text[start:end + 1])


def select_segments(project: dict, brief: str, target_length: float | None = None, allow_reorder: bool = False) -> dict:
    segments = project["segments"]
    prompt = build_prompt(segments, brief, target_length, allow_reorder)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )

    raw = response.choices[0].message.content
    parsed = extract_json(raw)
    final_sequence = parsed["final_sequence"]

    valid_ids = {s["id"] for s in segments}
    final_sequence = [sid for sid in final_sequence if sid in valid_ids]  # drop hallucinated ids defensively

    segment_by_id = {s["id"]: s for s in segments}

    # Everything starts cut; only ids in final_sequence get included + a new order
    for seg in segments:
        seg["included"] = False

    for new_order, sid in enumerate(final_sequence):
        segment_by_id[sid]["included"] = True
        segment_by_id[sid]["order"] = new_order

    kept_duration = sum(segment_by_id[sid]["end"] - segment_by_id[sid]["start"] for sid in final_sequence)
    print(f"Kept {len(final_sequence)}/{len(segments)} segments (~{kept_duration:.0f}s of edited footage)")
    if allow_reorder:
        original_order = [s["id"] for s in sorted(segments, key=lambda s: s["start"]) if s["id"] in final_sequence]
        if original_order != final_sequence:
            print("Segments were reordered from their original chronological position.")

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