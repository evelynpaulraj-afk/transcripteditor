# llm_select.py

import json
import sys
import os
import hashlib
from collections import Counter

from dotenv import load_dotenv
from openai import OpenAI

from relevance_filter import filter_by_relevance


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

MODEL = "gemini-3.5-flash-lite"

# Self-consistency:
# Run the final selection multiple times and keep majority-approved beats.
NUM_VOTES = 1


# ============================================================
# FILE HELPERS
# ============================================================

def load_project(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)


def load_visual_context(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# JSON / LLM HELPERS
# ============================================================

def extract_json(raw_text: str) -> dict:
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "").replace("```", "").strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        raise ValueError(f"No JSON found:\n{text}")

    text = text[start:end + 1]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        print("\n----- Invalid JSON returned by Gemini -----")
        print(text)
        print("------------------------------------------")
        raise


CACHE_DIR = "llm_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def call_model(prompt):

    key = hashlib.md5(prompt.encode()).hexdigest()
    cache_file = os.path.join(CACHE_DIR, key + ".json")

    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        response_format={"type":"json_object"},
        temperature=0,
        max_tokens=8192,
    )

    raw = response.choices[0].message.content

    with open("gemini_response.txt", "w", encoding="utf-8") as f:
        f.write(raw)

    print("\n========== RAW GEMINI RESPONSE ==========")
    print(raw)
    print("=========================================\n")

    result = extract_json(raw)

    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def estimate_tokens(text: str) -> int:
    """
    Rough token estimate.

    This is NOT the provider's actual tokenizer.
    It is only used for relative before/after comparisons.
    """

    return max(
        1,
        len(text) // 4
    )


def compact_json(obj) -> str:

    return json.dumps(
        obj,
        separators=(",", ":"),
        ensure_ascii=False
    )


# ============================================================
# PASS -1
# LLM TRANSCRIPT COMPRESSION
# ============================================================

def build_compression_prompt(segments: list) -> str:

    compact = [
        {
            "id": s["id"],
            "text": s["text"]
        }
        for s in segments
    ]

    return f"""
Below are transcript sentences from spoken, unscripted speech.

For each sentence, produce a compressed version by removing:

- disfluencies
- filler words
- false starts
- meaningless repetitions
- speech noise that carries no information

IMPORTANT:

The compressed version is ONLY for reasoning.

The original transcript is the source of truth for video editing.

Therefore:

- Keep all technical content.
- Keep all factual information.
- Keep all important meaning.
- Do NOT paraphrase.
- Do NOT summarize.
- Do NOT change terminology.
- Do NOT invent information.
- Do NOT remove meaningful words.
- Only remove words that add no information.

If nothing can safely be removed, return the sentence unchanged.

Example:

Input:
"Um, yeah, so like, the way it works is you basically, you take the video, right,
and it, uh, it listens to it and it kind of figures out what words are being said."

Output:
"The way it works is you take the video and it listens to it and figures out
what words are being said."

Return valid JSON only.
Do not include explanations, markdown, comments, or trailing commas.

Respond with ONLY:

{{
    "compressed": [{{"id": "seg-0000","text": "..."}}]
}}

SENTENCES:

{compact_json(compact)}
"""


def compress_segments_with_llm(segments: list) -> None:
    """
    Adds 'text_compressed' to each segment.

    IMPORTANT:
    The original 'text' field is NEVER changed.
    """

    to_compress = [
        s
        for s in segments
        if "text_compressed" not in s
    ]

    if not to_compress:

        print(
            "All segments already have cached compression "
            "— skipping LLM compression pass"
        )

        return

    prompt = build_compression_prompt(
        to_compress
    )

    original_payload = [
        {
            "id": s["id"],
            "text": s["text"]
        }
        for s in to_compress
    ]

    compressed_input_size = estimate_tokens(
        compact_json(original_payload)
    )
    
    result = call_model(prompt)

    by_id = {
        s["id"]: s
        for s in to_compress
    }

    valid_ids = set(by_id.keys())

    seen = set()

    warnings = []

    for entry in result.get(
        "compressed",
        []
    ):

        sid = entry.get("id")

        new_text = entry.get(
            "text",
            ""
        ).strip()

        if sid not in valid_ids:
            continue

        if not new_text:
            continue

        original = by_id[sid]["text"]

        # Warn if compression appears unusually aggressive.
        if len(new_text) < len(original) * 0.4:

            warnings.append(
                (
                    sid,
                    original,
                    new_text
                )
            )

        # IMPORTANT:
        # We add a new field.
        # We DO NOT modify the original transcript.
        by_id[sid]["text_compressed"] = new_text

        seen.add(sid)

    # Defensive fallback.
    # If the model forgot a segment, use the original text.
    for s in to_compress:

        if s["id"] not in seen:

            s["text_compressed"] = s["text"]

    compressed_payload = [
        {
            "id": s["id"],
            "text": s["text_compressed"]
        }
        for s in to_compress
    ]

    compressed_size = estimate_tokens(
        compact_json(compressed_payload)
    )
    # --------------------------------------------------------
    # Compression Metrics
    # --------------------------------------------------------

    tokens_saved = compressed_input_size - compressed_size

    compression_ratio = (
        compressed_input_size / compressed_size
        if compressed_size else 0
    )

    compression_percent = (
        tokens_saved / compressed_input_size * 100
        if compressed_input_size else 0
    )

    avg_before = (
        compressed_input_size / len(to_compress)
        if to_compress else 0
    )

    avg_after = (
        compressed_size / len(to_compress)
        if to_compress else 0
    )

    print("\n========== Prompt Compression Metrics ==========")
    print(f"Original tokens           : {compressed_input_size}")
    print(f"Compressed tokens         : {compressed_size}")
    print(f"Tokens saved              : {tokens_saved}")
    print(f"Compression ratio         : {compression_ratio:.2f}")
    print(f"Compression reduction (%) : {compression_percent:.2f}%")
    print(f"Average tokens/segment before : {avg_before:.2f}")
    print(f"Average tokens/segment after  : {avg_after:.2f}")

    print("\nSentence-level Compression")
    print("-" * 60)

    for s in to_compress:

        before = estimate_tokens(s["text"])

        after = estimate_tokens(
            s["text_compressed"]
        )

        reduction = (
            (before - after) / before * 100
            if before else 0
        )

        print(
            f'{s["id"]:>10} | '
            f'{before:3d} -> {after:3d} tokens | '
            f'{reduction:5.1f}%'
        )

    print("=" * 60)
    
    print(
        f"\nLLM compression summary:"
        f"\n{compressed_input_size} -> {compressed_size} tokens"
        f"\nReduction: {compression_percent:.2f}%"
        f"\nProcessed sentences: {len(to_compress)}"
    )

    if warnings:

        print(
            f"Compression sanity check: "
            f"{len(warnings)} sentence(s) shrank "
            f"by more than 60% — review these:"
        )

        for sid, original, compressed in warnings[:5]:

            print(
                f"  [{sid}] BEFORE: {original}"
            )

            print(
                f"  [{sid}] AFTER:  {compressed}"
            )


def debug_compression_sample(
    segments: list,
    n: int = 5
):

    print(
        "\n--- Compression debug sample ---"
    )

    shown = 0

    for s in segments:

        original = s["text"]

        compressed = s.get(
            "text_compressed",
            original
        )

        if original != compressed:

            print(
                f"  BEFORE: {original}"
            )

            print(
                f"  AFTER:  {compressed}"
            )

            print()

            shown += 1

        if shown >= n:
            break

    if shown == 0:

        print(
            "  No sentences were changed "
            "by compression at all."
        )

    print(
        "--- end sample ---\n"
    )


# ============================================================
# VISUAL CONTEXT
# ============================================================

def segment_overlaps_shot(
    segment,
    shot
):
    """
    Returns True when a transcript segment overlaps
    a visual shot in time.
    """

    return (
        segment["start"] < shot["end"]
        and segment["end"] > shot["start"]
    )


def find_visual_context_for_segments(
    segments,
    visual_data
):

    shots = visual_data

    relevant_shots = []

    seen = set()

    for shot in shots:

        for segment in segments:

            if segment_overlaps_shot(
                segment,
                shot
            ):

                shot_id = shot.get(
                    "shot_id"
                )

                if shot_id not in seen:

                    relevant_shots.append(
                        shot
                    )

                    seen.add(
                        shot_id
                    )

                break

    return relevant_shots


def format_visual_context(
    visual_shots
):
    """
    Converts visual metadata into compact LLM context.

    Visual information is deliberately kept compact.
    """

    if not visual_shots:

        return "No visual context available."

    lines = []

    for shot in visual_shots:

        visual = shot.get(
            "analysis",
            {}
        )

        shot_id = shot.get(
            "shot_id",
            "unknown"
        )

        start = shot.get(
            "start",
            0
        )

        end = shot.get(
            "end",
            0
        )

        lines.append(
            f"SHOT {shot_id} "
            f"[{start:.2f}-{end:.2f}]"
        )

        people = visual.get(
            "people",
            []
        )

        if people:
            lines.append(
                f"  People: {people}"
            )

        actions = visual.get(
            "actions",
            []
        )

        if actions:
            lines.append(
                f"  Actions: {actions}"
            )

        objects = visual.get(
            "objects",
            []
        )

        if objects:
            lines.append(
                f"  Objects: {objects}"
            )

        visual_content = visual.get(
            "visual_content",
            []
        )

        if visual_content:
            lines.append(
                f"  Visual content: "
                f"{visual_content}"
            )

        screen_content = visual.get(
            "screen_or_slide_content",
            []
        )

        if screen_content:
            lines.append(
                f"  Screen/slide content: "
                f"{screen_content}"
            )

        camera = visual.get(
            "camera_framing",
            ""
        )

        if camera:
            lines.append(
                f"  Camera: {camera}"
            )

        relevance = visual.get(
            "editing_relevance",
            ""
        )

        if relevance:
            lines.append(
                f"  Editing relevance: "
                f"{relevance}"
            )

        lines.append("")

    return "\n".join(lines)


# ============================================================
# PASS 0
# GROUP SENTENCES INTO COHERENT BEATS
# ============================================================

def build_grouping_prompt(
    segments: list
) -> str:

    compact = [
        {
            "id": s["id"],
            "text": s.get(
                "text_compressed",
                s["text"]
            )
        }
        for s in segments
    ]

    return f"""
Below is a transcript broken into individual sentences
in original spoken order.

Group these sentences into "beats".

A beat is one complete, self-contained thought or point
that should be treated as one indivisible editing unit.

Example:

Input:

seg-A:
"The first thing I want to talk about is how transcription works."

seg-B:
"It takes a video file and runs it through Whisper to get timestamps."

seg-C:
"The way it works is you take the video and it listens to it."

seg-D:
"Oh random side note, I had coffee this morning."

Correct:

[
    ["seg-A", "seg-B"],
    ["seg-C"],
    ["seg-D"]
]

Why:

A+B are one beat because B provides the explanation/payoff
for A.

C is its own beat because it is a separate attempt at
explaining the concept.

D is unrelated and therefore its own beat.

Rules:

- A setup sentence MUST be grouped with the sentence(s)
  that provide its explanation.
- A filler/redundant restatement should remain independently
  cuttable rather than being merged with the original point.
- Every ID must appear in exactly one beat.
- Beats must contain only consecutive transcript IDs.
- Do not reorder sentences.
- Do not omit sentences.
- Do not create IDs.
- Do not modify the meaning.

Respond with ONLY:

{{
    "beats": [
        {{
            "ids": [
                "seg-0000",
                "seg-0001"
            ]
        }},
        {{
            "ids": [
                "seg-0002"
            ]
        }}
    ]
}}

TRANSCRIPT:

{compact_json(compact)}
"""


def group_into_beats(
    segments: list
) -> list:

    prompt = build_grouping_prompt(
        segments
    )

    result = call_model(
        prompt
    )

    valid_ids = {
        s["id"]
        for s in segments
    }

    beats = []

    seen = set()

    invalid_count = 0

    for b in result.get(
        "beats",
        []
    ):

        raw_ids = b.get(
            "ids",
            []
        )

        ids = [
            i
            for i in raw_ids
            if i in valid_ids
            and i not in seen
        ]

        invalid_count += (
            len(raw_ids)
            - len(ids)
        )

        if ids:

            beats.append(
                ids
            )

            seen.update(
                ids
            )

    # Defensive fallback:
    # Anything the model forgot becomes its own beat.
    for s in segments:

        sid = s["id"]

        if sid not in seen:

            beats.append(
                [sid]
            )

            seen.add(
                sid
            )

    if invalid_count:

        print(
            f"Hallucination check: "
            f"{invalid_count} invalid/duplicate "
            f"ID(s) in grouping pass"
        )

    return beats


# ============================================================
# PASS 1
# BUILD SELECTION PROMPT
# ============================================================

def build_selection_prompt(
    beats: list,
    segment_by_id: dict,
    brief: str,
    target_length,
    allow_reorder: bool,
    visual_context=None
) -> str:

    compact = []

    for i, ids in enumerate(beats):

        text = " ".join(
            segment_by_id[i2].get(
                "text_compressed",
                segment_by_id[i2]["text"]
            )
            for i2 in ids
        )

        duration = sum(
            segment_by_id[i2]["end"]
            - segment_by_id[i2]["start"]
            for i2 in ids
        )

        start = min(
            segment_by_id[i2]["start"]
            for i2 in ids
        )

        end = max(
            segment_by_id[i2]["end"]
            for i2 in ids
        )

        compact.append(
            {
                "beat_id":
                    f"beat-{i:03d}",

                "start":
                    round(start, 2),

                "end":
                    round(end, 2),

                "duration_sec":
                    round(duration, 2),

                "text":
                    text
            }
        )

    total_duration = sum(
        b["duration_sec"]
        for b in compact
    )

    if target_length:

        length_instruction = (
            f"Target final length: approximately "
            f"{target_length}s. "
            f"Full available footage represented here "
            f"is approximately {total_duration:.0f}s."
        )

    else:

        length_instruction = (
            f"Full available footage represented here "
            f"is approximately {total_duration:.0f}s. "
            f"Keep only what is relevant."
        )

    if allow_reorder:

        reorder_instruction = (
            "You MAY reorder beats if doing so creates "
            "a clearer or more compelling story."
        )

    else:

        reorder_instruction = (
            "Keep beats in their original relative order."
        )

    # --------------------------------------------------------
    # VISUAL CONTEXT
    # --------------------------------------------------------

    if visual_context:

        visual_section = format_visual_context(
            visual_context
        )

    else:

        visual_section = (
            "No visual context was supplied. "
            "Base decisions on transcript content only."
        )

    return f"""
You are an expert video editor.

Your job is to decide which transcript beats should be
KEPT and in what SEQUENCE.

Each beat is an indivisible editing unit.

IMPORTANT ARCHITECTURE:

The transcript timestamps are the authoritative source
for actual video editing.

The compressed transcript text is ONLY a reasoning aid.

Visual context is supporting information about what is
actually visible at corresponding timestamps.

You must select beats using their IDs.

You must NOT invent transcript content.

You must NOT invent visual information.

BRIEF:

"{brief}"

{length_instruction}

{reorder_instruction}


RULES:

1. Cut filler, tangents, irrelevant material, and
   unnecessary repetition unless the brief requires them.

2. Keep information that directly supports the brief.

3. A beat must be either completely kept or completely cut.

4. You may NOT split a beat.

5. For every beat you keep, provide a short reason under
   12 words.

6. The reason must be grounded in the actual transcript
   and/or visual context.

7. Do not select a beat merely because it contains words
   similar to the brief. Consider its actual usefulness.

8. Visual context can help determine whether a shot supports
   what the speaker is saying.

9. Do not treat visual context as a replacement for the
   transcript.

10. The original transcript order should be preserved unless
    reordering is explicitly allowed.

11. Return ONLY valid beat IDs that appear below.

12. Do not return segment IDs.

13. Do not return timestamps as editing instructions.

Your output only decides which beats are kept.
The backend will map those beats back to their original
timestamped transcript segments.


Respond with ONLY:

{{
    "final_sequence": [
        {{
            "beat_id": "beat-002",
            "reason": "directly explains the requested mechanism"
        }},
        {{
            "beat_id": "beat-005",
            "reason": "graph visually supports the explanation"
        }}
    ]
}}


BEATS:

{compact_json(compact)}


VISUAL CONTEXT:

{visual_section}
"""


# ============================================================
# PASS 1
# ONE SELECTION RUN
# ============================================================

def run_selection_once(
    beats,
    segment_by_id,
    brief,
    target_length,
    allow_reorder,
    beat_by_label,
    visual_context=None
):

    prompt = build_selection_prompt(
        beats=beats,
        segment_by_id=segment_by_id,
        brief=brief,
        target_length=target_length,
        allow_reorder=allow_reorder,
        visual_context=visual_context
    )

    result = call_model(
        prompt
    )

    entries = result.get(
        "final_sequence",
        []
    )

    chosen = []

    seen_labels = set()

    for entry in entries:

        label = entry.get(
            "beat_id"
        )

        reason = entry.get(
            "reason",
            ""
        )

        if label not in beat_by_label:
            continue

        # Prevent duplicate beat selection.
        if label in seen_labels:
            continue

        chosen.append(
            (
                label,
                reason
            )
        )

        seen_labels.add(
            label
        )

    return chosen


# ============================================================
# MAIN SELECTION PIPELINE
# ============================================================

def select_segments(
    project: dict,
    brief: str,
    target_length=None,
    allow_reorder: bool = False,
    visual_context=None
) -> dict:

    all_segments = project["segments"]

    print(
        f"\nTotal transcript segments: "
        f"{len(all_segments)}"
    )

    # --------------------------------------------------------
    # STEP 1: EMBEDDING RELEVANCE FILTER
    # --------------------------------------------------------

    kept_segments, dropped_segments = (
        filter_by_relevance(
            all_segments,
            brief
        )
    )

    if dropped_segments:

        dropped_ids = [
            s["id"]
            for s in dropped_segments
        ]

        print(
            f"\nRelevance pre-filter: "
            f"dropped "
            f"{len(dropped_segments)}/"
            f"{len(all_segments)} "
            f"low-relevance segments "
            f"before sending to LLM."
        )

        for s in dropped_segments:

            s["included"] = False

    segments = kept_segments

    print(
        f"Segments entering LLM pipeline: "
        f"{len(segments)}"
    )

    # --------------------------------------------------------
    # STEP 2: VISUAL CONTEXT
    # --------------------------------------------------------

    relevant_visual = []

    if visual_context:

        relevant_visual = (
            find_visual_context_for_segments(
                segments,
                visual_context
            )
        )

        print(
            f"Visual context: "
            f"{len(relevant_visual)} "
            f"relevant shots matched "
            f"to transcript."
        )

    else:

        print(
            "Visual context: "
            "not provided."
        )

    # --------------------------------------------------------
    # STEP 3: LLM TRANSCRIPT COMPRESSION
    # --------------------------------------------------------

    segment_by_id = {
        s["id"]: s
        for s in segments
    }

    BATCH_SIZE = 20

    for i in range(0, len(segments), BATCH_SIZE):
        batch = segments[i:i+BATCH_SIZE]
        compress_segments_with_llm(batch)

    debug_compression_sample(
        segments
    )

    # --------------------------------------------------------
    # STEP 4: GROUP INTO BEATS
    # --------------------------------------------------------

    beats = group_into_beats(
        segments
    )

    print(
        f"Grouped {len(segments)} "
        f"sentences into "
        f"{len(beats)} beats"
    )

    beat_by_label = {
        f"beat-{i:03d}": ids
        for i, ids in enumerate(beats)
    }

    # --------------------------------------------------------
    # STEP 5: SELF-CONSISTENCY
    # --------------------------------------------------------

    print(
        f"\nRunning selection pass "
        f"{NUM_VOTES}x for "
        f"self-consistency check..."
    )

    run_results = []

    reasons_by_run = []

    for v in range(NUM_VOTES):

        print(
            f"  Selection run "
            f"{v + 1}/{NUM_VOTES}"
        )

        chosen = run_selection_once(
            beats,
            segment_by_id,
            brief,
            target_length,
            allow_reorder,
            beat_by_label,
            relevant_visual
        )

        run_results.append(
            [
                label
                for label, _ in chosen
            ]
        )

        reasons_by_run.append(
            {
                label: reason
                for label, reason in chosen
            }
        )

    # --------------------------------------------------------
    # STEP 6: COUNT VOTES
    # --------------------------------------------------------

    vote_counts = Counter()

    for run in run_results:

        for label in set(run):

            vote_counts[label] += 1

    majority_threshold = (
        NUM_VOTES // 2
    ) + 1

    majority_beats = {
        label
        for label, count
        in vote_counts.items()
        if count >= majority_threshold
    }

    # --------------------------------------------------------
    # SHOW DISAGREEMENTS
    # --------------------------------------------------------

    disagreements = {
        label: count
        for label, count
        in vote_counts.items()
        if count < NUM_VOTES
    }

    if disagreements:

        print(
            "\nSelf-consistency disagreements:"
        )

        for label, count in sorted(
            disagreements.items()
        ):

            if label in majority_beats:

                verdict = (
                    "kept (majority)"
                )

            else:

                verdict = (
                    "cut (not majority)"
                )

            print(
                f"  {label}: "
                f"selected in "
                f"{count}/{NUM_VOTES} "
                f"runs -> {verdict}"
            )

    else:

        print(
            "\nSelf-consistency check: "
            "all runs agreed on every beat"
        )

    # --------------------------------------------------------
    # STEP 7: FIND BEST RUN
    # --------------------------------------------------------

    best_run_idx = max(
        range(NUM_VOTES),
        key=lambda i:
        sum(
            1
            for beat in run_results[i]
            if beat in majority_beats
        )
    )

    final_labels = [
        beat
        for beat in run_results[best_run_idx]
        if beat in majority_beats
    ]

    final_reasons = reasons_by_run[
        best_run_idx
    ]

    # --------------------------------------------------------
    # FINAL SELECTION DISPLAY
    # --------------------------------------------------------

    print(
        "\nFinal selection with reasoning:"
    )

    for label in final_labels:

        print(
            f"  {label}: "
            f"{final_reasons.get(label, '(no reason recorded)')}"
        )

    # --------------------------------------------------------
    # STEP 8: MAP BEATS BACK TO ORIGINAL SEGMENT IDS
    # --------------------------------------------------------

    final_sequence = []

    for label in final_labels:

        final_sequence.extend(
            beat_by_label[label]
        )

    # --------------------------------------------------------
    # STEP 9: UPDATE EDITING FLAGS
    # --------------------------------------------------------

    # IMPORTANT:
    #
    # We only change included/order.
    #
    # We DO NOT change:
    #   text
    #   words
    #   start
    #   end
    #
    # Therefore your video editing pipeline still uses
    # the original transcript timestamps.

    for seg in segments:

        seg["included"] = False

    for new_order, sid in enumerate(
        final_sequence
    ):

        segment_by_id[sid][
            "included"
        ] = True

        segment_by_id[sid][
            "order"
        ] = new_order

    # --------------------------------------------------------
    # STEP 10: DURATION
    # --------------------------------------------------------

    kept_duration = sum(
        segment_by_id[sid]["end"]
        - segment_by_id[sid]["start"]
        for sid in final_sequence
    )

    print(
        f"\nKept "
        f"{len(final_labels)}/"
        f"{len(beats)} beats "
        f"-> "
        f"{len(final_sequence)} segments "
        f"(~{kept_duration:.0f}s)"
    )

    # --------------------------------------------------------
    # STORE OPTIONAL METADATA
    # --------------------------------------------------------

    project["editing_metadata"] = {
        "brief": brief,
        "visual_context_used": bool(
            visual_context
        ),
        "relevant_visual_shots": len(
            relevant_visual
        ),
        "num_votes": NUM_VOTES,
        "majority_threshold": majority_threshold,
        "selected_beats": final_labels
    }

    return project


# ============================================================
# COMMAND LINE INTERFACE
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 4:

        print(
            "Usage:\n"
            "python llm_select.py "
            "<project.json> "
            "<output.json> "
            "\"<brief>\" "
            "[target_length_seconds] "
            "[--reorder] "
            "[--visual-context visual_context.json]"
        )

        sys.exit(1)

    project_path = sys.argv[1]

    output_path = sys.argv[2]

    brief = sys.argv[3]

    remaining = sys.argv[4:]

    # --------------------------------------------------------
    # REORDER FLAG
    # --------------------------------------------------------

    allow_reorder = (
        "--reorder"
        in remaining
    )

    remaining = [
        arg
        for arg in remaining
        if arg != "--reorder"
    ]

    # --------------------------------------------------------
    # TARGET LENGTH
    # --------------------------------------------------------

    target_length = None

    if remaining:

        first_arg = remaining[0]

        # Only treat it as target length if it is numeric.
        try:

            target_length = float(
                first_arg
            )

            remaining = remaining[1:]

        except ValueError:

            target_length = None

    # --------------------------------------------------------
    # VISUAL CONTEXT FILE
    # --------------------------------------------------------

    visual_context = None

    if "--visual-context" in remaining:

        index = remaining.index(
            "--visual-context"
        )

        if index + 1 >= len(
            remaining
        ):

            print(
                "Error: "
                "--visual-context requires "
                "a JSON file path."
            )

            sys.exit(1)

        visual_context_path = (
            remaining[index + 1]
        )

        print(
            f"Loading visual context from: "
            f"{visual_context_path}"
        )

        visual_context = load_visual_context(
            visual_context_path
        )

    # --------------------------------------------------------
    # LOAD PROJECT
    # --------------------------------------------------------

    project = load_project(
        project_path
    )

    # --------------------------------------------------------
    # RUN PIPELINE
    # --------------------------------------------------------

    updated = select_segments(
        project,
        brief,
        target_length,
        allow_reorder,
        visual_context
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    save_project(
        updated,
        output_path
    )

    print(
        f"\nSaved decisions to "
        f"{output_path}"
    )