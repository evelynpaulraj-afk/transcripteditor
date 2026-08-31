# inspect_decisions.py
import json
import sys

def inspect(project_path: str):
    with open(project_path, "r", encoding="utf-8") as f:
        project = json.load(f)

    segments = project["segments"]

    # Original chronological order = sorted by start time (how they were recorded)
    original_order = [s["id"] for s in sorted(segments, key=lambda s: s["start"])]

    # Final order = sorted by the 'order' field (what the LLM/user decided)
    kept = [s for s in segments if s["included"]]
    final_order = sorted(kept, key=lambda s: s["order"])

    print("=== FINAL EDIT (in playback order) ===\n")
    for i, s in enumerate(final_order):
        orig_position = original_order.index(s["id"])
        moved = " <-- MOVED" if orig_position != original_order.index(final_order[i-1]["id"]) + 1 and i > 0 else ""
        print(f"{i+1}. ({s['id']}, originally #{orig_position+1} in recording) {s['start']:.1f}s-{s['end']:.1f}s: {s['text']}{moved}")

    print(f"\n=== CUT ({len(segments) - len(kept)} segments) ===\n")
    for s in segments:
        if not s["included"]:
            print(f"  ({s['id']}) {s['start']:.1f}s-{s['end']:.1f}s: {s['text']}")

if __name__ == "__main__":
    inspect(sys.argv[1])