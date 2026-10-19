import os
import json
import cv2
import time
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ============================================================
# DEVELOPMENT SETTINGS
# ============================================================

TEST_MODE = True        # False for the final experiment
# ============================================================
# CACHE SETTINGS
# ============================================================

USE_CACHED_VISUAL_CONTEXT = True

MAX_TEST_SHOTS = 10

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY not found. Add it to your .env file."
    )

client = genai.Client(api_key=API_KEY)

# Current Gemini model
MODEL_NAME = "gemini-3.5-flash-lite"

# Temporary directory for extracted frames
FRAME_DIR = "frames"

# Create a visual analysis request for every N seconds
SHOT_DURATION = 20.0

# Number of representative frames sent to Gemini for each shot
FRAMES_PER_SHOT = 1 if TEST_MODE else 3

# Small delay between API requests
API_DELAY = 1

# Number of retries for a failed Gemini request
MAX_RETRIES = 5


# ============================================================
# SHOT DETECTION
# ============================================================

def detect_shots(video_path):
    """
    Divide the video into fixed-duration windows.

    Returns:
        [
            {
                "shot_id": 0,
                "start": 0.0,
                "end": 5.0
            },
            ...
        ]
    """

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    duration = frame_count / fps

    shots = []

    shot_id = 0
    start = 0.0

    while start < duration:
        end = min(start + SHOT_DURATION, duration)

        shots.append({
            "shot_id": shot_id,
            "start": round(start, 3),
            "end": round(end, 3)
        })

        shot_id += 1
        start = end

    cap.release()

    return shots


# ============================================================
# FRAME EXTRACTION
# ============================================================

def extract_frames(video_path, shots):
    """
    Extract 3 representative frames from every shot.

    Frames are selected around:
        20%, 50%, 80%

    Returns:
        shots with an additional "frames" field.
    """

    os.makedirs(FRAME_DIR, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 0:
        cap.release()
        raise RuntimeError("Could not determine video FPS.")

    for shot in shots:

        start = shot["start"]
        end = shot["end"]

        duration = end - start

        # Representative points inside the shot
        if TEST_MODE:
            percentages = [0.50]          # middle frame only
        else:
            percentages = [0.20, 0.50, 0.80]

        frame_paths = []

        for index, percentage in enumerate(percentages):

            timestamp = start + duration * percentage

            frame_number = int(timestamp * fps)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            success, frame = cap.read()

            if not success:
                continue

            filename = (
                f"shot_{shot['shot_id']:04d}"
                f"_frame_{index}.jpg"
            )

            path = os.path.join(FRAME_DIR, filename)

            cv2.imwrite(
                path,
                frame,
                [cv2.IMWRITE_JPEG_QUALITY, 65]
            )

            frame_paths.append(path)

        shot["frames"] = frame_paths

    cap.release()

    return shots


# ============================================================
# GEMINI VISUAL ANALYSIS
# ============================================================

def analyze_shot(shot):
    gemini_start = time.time()
    """
    Send the representative frames of one shot to Gemini.

    Gemini receives:
        - 3 representative images
        - one visual-analysis instruction

    The response is expected to be JSON with the schema used
    by the downstream llm_select.py pipeline.
    """

    prompt = f"""
    You are performing visual analysis for an AI video editor.

    Analyze the following video shot using the provided representative
    frames.

    Shot information:
    - Shot ID: {shot["shot_id"]}
    - Start time: {shot["start"]} seconds
    - End time: {shot["end"]} seconds

    Return ONLY valid JSON.
    Do not include markdown.
    Do not include explanations outside the JSON.

    Use exactly this schema:

    {{
    "people": [],
    "actions": [],
    "objects": [],
    "visual_content": [],
    "screen_or_slide_content": [],
    "camera_framing": "",
    "editing_relevance": ""
    }}

    Instructions:

    1. people
    - List visible people or relevant characteristics.
    - Example: ["one person", "man", "woman", "speaker"]

    2. actions
    - Describe visible actions.
    - Example: ["speaking", "walking", "typing"]

    3. objects
    - List important visible objects.
    - Example: ["laptop", "microphone", "phone"]

    4. visual_content
    - Describe important visual information in the scene.
    - Include meaningful environmental or visual details.

    5. screen_or_slide_content
    - If a screen, presentation, chart, webpage, document, or slide
        is visible, describe its important content.
    - If none is visible, return [].

    6. camera_framing
    - Describe the framing.
    - Example: "medium shot", "close-up", "wide shot",
        "screen recording", "two-person shot"

    7. editing_relevance
    - Explain briefly what makes this shot relevant for video editing.
    - Focus on things such as speech context, visual changes,
        demonstrations, slides, screen content, actions, or scene changes.

    Be factual.
    Do not invent information that is not visible.
    Keep each field concise.
    """

    # Build multimodal Interactions API input
    inputs = []

    for frame_path in shot.get("frames", []):

        with open(frame_path, "rb") as f:
            image_bytes = f.read()

        inputs.append(
            types.Part.from_bytes(
                data=image_bytes,
                mime_type="image/jpeg"
            )
        )

    inputs.append(prompt)

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            print(
                f"  Gemini analysis attempt "
                f"{attempt}/{MAX_RETRIES}..."
            )

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=inputs
            )

            response_text = response.text

            # Remove accidental markdown fences if Gemini adds them
            if response_text.startswith("```"):
                response_text = response_text.replace(
                    "```json", "", 1
                ).replace(
                    "```", "", 1
                ).strip()

            data = json.loads(response_text)

            # Ensure all expected fields exist
            result = {
                "people": data.get("people", []),
                "actions": data.get("actions", []),
                "objects": data.get("objects", []),
                "visual_content": data.get("visual_content", []),
                "screen_or_slide_content": data.get(
                    "screen_or_slide_content", []
                ),
                "camera_framing": data.get(
                    "camera_framing", ""
                ),
                "editing_relevance": data.get(
                    "editing_relevance", ""
                )
            }
            print(f"Gemini took {time.time()-gemini_start:.2f} seconds")
            return result

        except Exception as e:

            last_error = e

            print(
                f"  Gemini request failed: {e}"
            )

            if attempt < MAX_RETRIES:
                if "429" in str(e):
                    wait_time = 15
                else:
                    wait_time = attempt * 5

                print(
                    f"  Retrying in {wait_time} seconds..."
                )

                time.sleep(wait_time)

    print(
        f"  WARNING: visual analysis failed for "
        f"shot {shot['shot_id']}"
    )
    
    return {
        "people": [],
        "actions": [],
        "objects": [],
        "visual_content": [],
        "screen_or_slide_content": [],
        "camera_framing": "",
        "editing_relevance": f"Visual analysis failed: {last_error}"
    }


# ============================================================
# MAIN
# ============================================================
def main():

    if len(sys.argv) < 3:
        print(
            "Usage:\n"
            "python generate_visual_context.py "
            "<video_path> <output_json>"
        )
        sys.exit(1)

    video_path = sys.argv[1]
    output_path = sys.argv[2]

    if not os.path.exists(video_path):
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    print("=" * 60)
    print("GEMINI VISUAL CONTEXT GENERATION")
    print("=" * 60)

    print(f"Video: {video_path}")
    print(f"Model: {MODEL_NAME}")
    print(f"Shot duration: {SHOT_DURATION}s")
    print(f"Frames per shot: {FRAMES_PER_SHOT}")
    print()

    # --------------------------------------------------------
    # 1. Detect shots
    # --------------------------------------------------------

    print("Detecting shots...")

    t = time.time()
    shots = detect_shots(video_path)
    if TEST_MODE:
        shots = shots[:MAX_TEST_SHOTS]
    print(f"Shot detection took {time.time()-t:.2f} seconds")

    print(f"Detected {len(shots)} shots.")
    print()

    # --------------------------------------------------------
    # 2. Extract representative frames
    # --------------------------------------------------------

    print("Extracting representative frames...")

    t = time.time()
    shots = extract_frames(video_path, shots)
    print(f"Frame extraction took {time.time()-t:.2f} seconds")

    total_frames = sum(
        len(shot.get("frames", []))
        for shot in shots
    )

    print(
        f"Extracted {total_frames} representative frames."
    )
    print()

    # --------------------------------------------------------
    # 3. Cache / Resume Support
    # --------------------------------------------------------

    if USE_CACHED_VISUAL_CONTEXT and os.path.exists(output_path):

        print("=" * 60)
        print("Using cached visual context")
        print("=" * 60)
        print(f"Loaded: {output_path}")
        print()

        with open(output_path, "r", encoding="utf-8") as f:
            visual_context = json.load(f)

        print(f"Loaded {len(visual_context)} analyzed shots.")

        print("Skipping Gemini visual analysis.")
        print()

        return

    visual_context = []
    completed_shots = set()

    if os.path.exists(output_path):

        try:

            with open(output_path, "r", encoding="utf-8") as f:
                visual_context = json.load(f)

            completed_shots = {
                shot["shot_id"]
                for shot in visual_context
            }

            print(
                f"Resuming from {len(completed_shots)} completed shots."
            )

        except Exception:

            print("Existing JSON could not be read.")

            visual_context = []
            completed_shots = set()

    # --------------------------------------------------------
    # 4. Decide what to process
    # --------------------------------------------------------

    shots_to_process = (
        shots[:MAX_TEST_SHOTS]
        if TEST_MODE
        else shots
    )

    print(
        f"Processing {len(shots_to_process)} shots..."
    )
    print()

    start_time = time.time()

    # --------------------------------------------------------
    # 5. Analyze shots
    # --------------------------------------------------------
    processed_count = len(completed_shots)

    for index, shot in enumerate(shots_to_process):

        if shot["shot_id"] in completed_shots:
            print(
                f"Skipping Shot {shot['shot_id']} (already completed)"
            )
            continue

        print(
            f"[{index + 1}/{len(shots_to_process)}] "
            f"Shot {shot['shot_id']} "
            f"({shot['start']}s - {shot['end']}s)"
        )

        analysis = analyze_shot(shot)

        visual_context.append({
            "shot_id": shot["shot_id"],
            "start": shot["start"],
            "end": shot["end"],
            "analysis": analysis
        })

        # Save after every successful shot
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                visual_context,
                f,
                indent=2,
                ensure_ascii=False
            )

        print(f"  People: {analysis['people']}")
        print(f"  Actions: {analysis['actions']}")
        print(f"  Framing: {analysis['camera_framing']}")
        print()

        # ETA
        # Update processed count
        processed_count += 1

        # ETA
        elapsed = time.time() - start_time
        avg = elapsed / max(processed_count - len(completed_shots), 1)
        remaining = len(shots_to_process) - processed_count
        eta = remaining * avg

        print(
            f"Completed: {processed_count}/{len(shots_to_process)} | "
            f"ETA: {eta/60:.1f} minutes"
        )

        if index < len(shots_to_process) - 1:
            time.sleep(API_DELAY)

    # --------------------------------------------------------
    # 6. Final save
    # --------------------------------------------------------

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            visual_context,
            f,
            indent=2,
            ensure_ascii=False
        )

    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(
        f"Saved visual context to: {output_path}"
    )


if __name__ == "__main__":
    main()