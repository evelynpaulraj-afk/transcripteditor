# export_video.py
import json
import subprocess
import sys
import os
import tempfile
import shutil


def load_project(project_path: str) -> dict:
    with open(project_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_included_segments_in_order(project: dict) -> list:
    included = [s for s in project["segments"] if s["included"]]
    included.sort(key=lambda s: s["order"])
    return included


def build_keep_ranges(segments: list, pad: float = 0.08, max_pause: float = 1.2) -> list:
    """
    Merge consecutive included segments into continuous ranges when they were
    adjacent in the ORIGINAL video AND the gap between them is short (a natural
    breath, not dead air). Gaps longer than max_pause are treated as a real cut
    boundary, even if nothing was excluded there — this trims long pauses.
    """
    ranges = []
    for seg in segments:
        orig_index = int(seg["id"].split("-")[1])

        if ranges:
            prev = ranges[-1]
            contiguous_id = prev["_last_orig_index"] + 1 == orig_index
            gap = seg["start"] - prev["end"]  # actual silence duration in the source video

            if contiguous_id and gap <= max_pause:
                prev["end"] = seg["end"]
                prev["_last_orig_index"] = orig_index
                continue  # merged — skip creating a new range

        ranges.append({
            "start": seg["start"],
            "end": seg["end"],
            "_last_orig_index": orig_index,
        })

    for i, r in enumerate(ranges):
        r["start"] = max(0, r["start"] - pad) if i > 0 else r["start"]
        r["end"] = r["end"] + pad if i < len(ranges) - 1 else r["end"]
        del r["_last_orig_index"]

    return ranges


def cut_segment(video_path: str, start: float, end: float, out_path: str):
    duration = end - start
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-ss", str(start),
        "-t", str(duration),
        "-c:v", "libx264",
        "-c:a", "aac",
        "-avoid_negative_ts", "make_zero",
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed cutting [{start}-{end}]:\n{result.stderr}")


def concatenate_clips(clip_paths: list, output_path: str, temp_dir: str):
    list_file = os.path.join(temp_dir, "concat_list.txt")
    with open(list_file, "w", encoding="utf-8") as f:
        for clip in clip_paths:
            safe_path = os.path.abspath(clip).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", list_file,
        "-c", "copy",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed concatenating clips:\n{result.stderr}")


def export_video(project_path: str, output_path: str):
    project = load_project(project_path)
    video_path = project["video_path"]
    segments = get_included_segments_in_order(project)

    if not segments:
        raise ValueError("No included segments found — nothing to export.")

    ranges = build_keep_ranges(segments)
    print(f"Exporting {len(ranges)} continuous ranges (from {len(segments)} kept segments)")

    temp_dir = tempfile.mkdtemp(prefix="transcript_editor_")
    clip_paths = []

    try:
        for i, r in enumerate(ranges):
            clip_path = os.path.join(temp_dir, f"clip_{i:04d}.mp4")
            print(f"  [{i+1}/{len(ranges)}] Cutting {r['start']:.2f}s -> {r['end']:.2f}s")
            cut_segment(video_path, r["start"], r["end"], clip_path)
            clip_paths.append(clip_path)

        print("Concatenating clips...")
        concatenate_clips(clip_paths, output_path, temp_dir)
        print(f"\nDone. Final edited video saved to: {output_path}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python export_video.py <project.json> <output.mp4>")
        sys.exit(1)
    export_video(sys.argv[1], sys.argv[2])