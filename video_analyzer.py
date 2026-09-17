import os
import json
import time
import cv2
import numpy as np
import pandas as pd
import pytesseract
import whisper

from PIL import Image
from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from google import genai
from google.genai import types


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_PATH = "dailyvlog.mp4"

GEMINI_API_KEY = "AQ.Ab8RN6KC4_auxeKuzbjkaKc2o5haDoND6NS2B7jKN4L_L1vZow"

GEMINI_MODEL = "gemini-3.6-flash"

# Number of shots to send to Gemini
NUM_KEY_SHOTS = 8

# Scene detection sensitivity
SCENE_THRESHOLD = 27.0

# Whisper model
WHISPER_MODEL = "base"

# Models
VISUAL_MODEL = "clip-ViT-B-32"
TEXT_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

# Windows
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"



if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH

import pytesseract

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ============================================================
# LOAD MODELS
# ============================================================

print("\nLoading Whisper...")

whisper_model = whisper.load_model(
    WHISPER_MODEL
)

print("Whisper loaded.")


print("\nLoading visual embedding model...")

visual_model = SentenceTransformer(
    VISUAL_MODEL
)

print("Visual model loaded.")


print("\nLoading text embedding model...")

text_model = SentenceTransformer(
    TEXT_MODEL
)

print("Text model loaded.")


# ============================================================
# GEMINI
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 1. SHOT DETECTION
# ============================================================

def detect_shots(video_path):

    print("\n========================================")
    print("SHOT DETECTION")
    print("========================================")

    video = open_video(video_path)

    scene_manager = SceneManager()

    scene_manager.add_detector(
        ContentDetector(
            threshold=SCENE_THRESHOLD
        )
    )

    scene_manager.detect_scenes(video)

    scenes = scene_manager.get_scene_list()

    shots = []

    for i, (start, end) in enumerate(scenes):

        shots.append({

            "shot_id": i,

            "start": start.get_seconds(),

            "end": end.get_seconds()

        })

    # If no shots detected, use entire video
    if not shots:

        duration = video.duration.get_seconds()

        shots = [{

            "shot_id": 0,

            "start": 0.0,

            "end": duration

        }]

    print(
        f"\nDetected {len(shots)} shots."
    )

    return shots


# ============================================================
# 2. WHISPER TRANSCRIPTION
# ============================================================

def transcribe_video(video_path):

    print("\n========================================")
    print("TRANSCRIPTION")
    print("========================================")

    result = whisper_model.transcribe(
        video_path,
        fp16=False
    )

    segments = []

    for segment in result["segments"]:

        segments.append({

            "start": float(
                segment["start"]
            ),

            "end": float(
                segment["end"]
            ),

            "text": segment["text"].strip()

        })

    print(
        f"Transcribed {len(segments)} segments."
    )

    return segments


# ============================================================
# 3. ALIGN TRANSCRIPT WITH SHOTS
# ============================================================

def align_transcript_to_shots(
    shots,
    transcript_segments
):

    shot_transcripts = []

    for shot in shots:

        shot_start = shot["start"]
        shot_end = shot["end"]

        text_parts = []

        for segment in transcript_segments:

            seg_start = segment["start"]
            seg_end = segment["end"]

            # Check temporal overlap
            overlap = (

                seg_start < shot_end

                and

                seg_end > shot_start

            )

            if overlap:

                text_parts.append(
                    segment["text"]
                )

        shot_text = " ".join(
            text_parts
        )

        shot_transcripts.append(
            shot_text
        )

    return shot_transcripts


# ============================================================
# 4. EXTRACT REPRESENTATIVE FRAME
# ============================================================

def extract_frame(
    video_path,
    start,
    end
):

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():

        return None

    middle = (
        start + end
    ) / 2

    cap.set(
        cv2.CAP_PROP_POS_MSEC,
        middle * 1000
    )

    success, frame = cap.read()

    cap.release()

    if not success:

        return None

    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    return Image.fromarray(
        frame
    )


# ============================================================
# 5. OCR
# ============================================================

def extract_ocr(frame):

    if frame is None:

        return ""

    try:

        text = pytesseract.image_to_string(
            frame
        )

        return " ".join(
            text.split()
        )

    except Exception as e:

        print(
            f"OCR error: {e}"
        )

        return ""


# ============================================================
# 6. VISUAL EMBEDDING
# ============================================================

def get_visual_embedding(frame):

    if frame is None:

        return None

    try:

        embedding = visual_model.encode(
            frame,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding

    except Exception as e:

        print(
            f"Visual embedding error: {e}"
        )

        return None


# ============================================================
# 7. TEXT EMBEDDING
# ============================================================

def get_text_embedding(text):

    if not text:

        return None

    if not text.strip():

        return None

    try:

        embedding = text_model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        return embedding

    except Exception as e:

        print(
            f"Text embedding error: {e}"
        )

        return None


# ============================================================
# 8. NORMALIZATION
# ============================================================

def normalize(values):

    values = np.asarray(
        values,
        dtype=float
    )

    if len(values) == 0:

        return values

    minimum = values.min()
    maximum = values.max()

    if (
        maximum - minimum
        < 1e-8
    ):

        return np.zeros_like(
            values
        )

    return (
        values - minimum
    ) / (
        maximum - minimum
    )


# ============================================================
# 9. VISUAL CHANGE SCORE
# ============================================================

def calculate_visual_change(
    embeddings
):

    scores = []

    for i in range(
        len(embeddings)
    ):

        if i == 0:

            scores.append(0.0)

            continue

        current = embeddings[i]
        previous = embeddings[i - 1]

        if (
            current is None
            or previous is None
        ):

            scores.append(0.0)

            continue

        similarity = np.dot(
            current,
            previous
        )

        change = 1.0 - similarity

        scores.append(
            change
        )

    return normalize(
        scores
    )


# ============================================================
# 10. TRANSCRIPT CHANGE SCORE
# ============================================================

def calculate_transcript_change(
    embeddings
):

    scores = []

    for i in range(
        len(embeddings)
    ):

        if i == 0:

            scores.append(0.0)

            continue

        current = embeddings[i]
        previous = embeddings[i - 1]

        if (
            current is None
            or previous is None
        ):

            scores.append(0.0)

            continue

        similarity = np.dot(
            current,
            previous
        )

        change = 1.0 - similarity

        scores.append(
            change
        )

    return normalize(
        scores
    )


# ============================================================
# 11. OCR IMPORTANCE SCORE
# ============================================================

def calculate_ocr_score(
    ocr_texts
):

    important_words = [

        "buy",
        "shop",
        "sale",
        "offer",
        "discount",
        "free",
        "limited",
        "price",
        "₹",
        "$",
        "%",
        "new",
        "now",
        "available",
        "download",
        "subscribe",
        "link",
        "website",
        "order"

    ]

    scores = []

    for text in ocr_texts:

        if not text:

            scores.append(0.0)

            continue

        lower = text.lower()

        word_count = len(
            text.split()
        )

        text_score = min(
            word_count / 15,
            1.0
        )

        keyword_score = 0.0

        for word in important_words:

            if word.lower() in lower:

                keyword_score += 0.15

        keyword_score = min(
            keyword_score,
            1.0
        )

        score = (

            0.5 * text_score

            +

            0.5 * keyword_score

        )

        scores.append(
            score
        )

    return normalize(
        scores
    )


# ============================================================
# 12. DURATION SCORE
# ============================================================

def calculate_duration_score(
    durations
):

    durations = np.asarray(
        durations,
        dtype=float
    )

    median = np.median(
        durations
    )

    differences = np.abs(
        durations - median
    )

    scores = differences / max(
        median,
        0.001
    )

    return normalize(
        scores
    )


# ============================================================
# 13. SELECT DIVERSE SHOTS
# ============================================================

def select_diverse_shots(
    df,
    visual_embeddings,
    number
):

    if len(df) <= number:

        return df.sort_values(
            "shot_id"
        )

    selected = []

    remaining = set(
        df.index.tolist()
    )

    # Start with highest importance
    first = df[
        "key_shot_score"
    ].idxmax()

    selected.append(
        first
    )

    remaining.remove(
        first
    )

    while (

        len(selected)
        <
        number

        and

        remaining

    ):

        best_candidate = None
        best_value = -999

        for candidate in remaining:

            candidate_embedding = (
                visual_embeddings[
                    candidate
                ]
            )

            if candidate_embedding is None:

                continue

            similarities = []

            for selected_idx in selected:

                selected_embedding = (
                    visual_embeddings[
                        selected_idx
                    ]
                )

                if selected_embedding is None:

                    continue

                similarity = np.dot(

                    candidate_embedding,

                    selected_embedding

                )

                similarities.append(
                    similarity
                )

            if similarities:

                max_similarity = max(
                    similarities
                )

            else:

                max_similarity = 0

            diversity = (
                1.0
                -
                max_similarity
            )

            importance = df.loc[
                candidate,
                "key_shot_score"
            ]

            value = (

                0.65 * importance

                +

                0.35 * diversity

            )

            if value > best_value:

                best_value = value

                best_candidate = candidate

        if best_candidate is None:

            break

        selected.append(
            best_candidate
        )

        remaining.remove(
            best_candidate
        )

    return df.loc[
        selected
    ].sort_values(
        "shot_id"
    )


# ============================================================
# 14. COMPLETE KEY-SHOT ANALYSIS
# ============================================================

def find_key_shots(
    video_path,
    shots,
    shot_transcripts,
    number=8
):

    print("\n========================================")
    print("KEY SHOT ANALYSIS")
    print("========================================")

    frames = []
    visual_embeddings = []
    text_embeddings = []
    ocr_texts = []
    durations = []

    for i, shot in enumerate(shots):

        start = shot["start"]
        end = shot["end"]

        duration = (
            end - start
        )

        print(
            f"\n[{i + 1}/{len(shots)}] "
            f"Shot {i}"
        )

        print(
            f"Time: "
            f"{start:.2f} - "
            f"{end:.2f}"
        )

        # Frame
        frame = extract_frame(
            video_path,
            start,
            end
        )

        frames.append(
            frame
        )

        # OCR
        ocr = extract_ocr(
            frame
        )

        ocr_texts.append(
            ocr
        )

        # Visual embedding
        visual_embedding = (
            get_visual_embedding(
                frame
            )
        )

        visual_embeddings.append(
            visual_embedding
        )

        # Transcript embedding
        text_embedding = (
            get_text_embedding(
                shot_transcripts[i]
            )
        )

        text_embeddings.append(
            text_embedding
        )

        durations.append(
            duration
        )

    # Scores
    visual_change = (
        calculate_visual_change(
            visual_embeddings
        )
    )

    transcript_change = (
        calculate_transcript_change(
            text_embeddings
        )
    )

    ocr_score = (
        calculate_ocr_score(
            ocr_texts
        )
    )

    duration_score = (
        calculate_duration_score(
            durations
        )
    )

    # ========================================================
    # FINAL SCORE
    # ========================================================

    final_score = (

        0.35 * visual_change

        +

        0.30 * transcript_change

        +

        0.25 * ocr_score

        +

        0.10 * duration_score

    )

    # ========================================================
    # DATAFRAME
    # ========================================================

    records = []

    for i, shot in enumerate(shots):

        records.append({

            "shot_id":
                i,

            "start":
                shot["start"],

            "end":
                shot["end"],

            "duration":
                durations[i],

            "transcript":
                shot_transcripts[i],

            "ocr":
                ocr_texts[i],

            "visual_change":
                visual_change[i],

            "transcript_change":
                transcript_change[i],

            "ocr_score":
                ocr_score[i],

            "duration_score":
                duration_score[i],

            "key_shot_score":
                final_score[i]

        })

    df = pd.DataFrame(
        records
    )

    # ========================================================
    # SELECT DIVERSE KEY SHOTS
    # ========================================================

    key_shots = select_diverse_shots(

        df,

        visual_embeddings,

        number

    )

    # Save everything
    df.to_csv(
        "all_shot_scores.csv",
        index=False
    )

    key_shots.to_csv(
        "key_shots.csv",
        index=False
    )

    # ========================================================
    # PRINT
    # ========================================================

    print("\n========================================")
    print("SELECTED KEY SHOTS")
    print("========================================")

    for _, row in key_shots.iterrows():

        print(
            f"\nShot {int(row['shot_id'])}"
        )

        print(
            f"Time: "
            f"{row['start']:.2f}s → "
            f"{row['end']:.2f}s"
        )

        print(
            f"Score: "
            f"{row['key_shot_score']:.3f}"
        )

        if row["transcript"]:

            print(
                f"Transcript: "
                f"{row['transcript']}"
            )

        if row["ocr"]:

            print(
                f"OCR: "
                f"{row['ocr']}"
            )

    return key_shots


# ============================================================
# 15. GEMINI ANALYSIS
# ============================================================

def analyze_shot_with_gemini(
    video_path,
    shot
):

    start = float(
        shot["start"]
    )

    end = float(
        shot["end"]
    )

    shot_id = int(
        shot["shot_id"]
    )

    frame = extract_frame(
        video_path,
        start,
        end
    )

    if frame is None:

        return None

    # Convert PIL → JPEG bytes
    import io

    buffer = io.BytesIO()

    frame.save(
        buffer,
        format="JPEG"
    )

    image_bytes = (
        buffer.getvalue()
    )

    prompt = """
Analyze this video shot for an AI video editor.

Return ONLY valid JSON.

Use exactly this structure:

{
  "shot_id": 0,
  "action": "",
  "description": "",
  "people": [],
  "objects": [],
  "framing": "",
  "camera_movement": "",
  "visual_importance": "",
  "editing_relevance": ""
}

Definitions:

action:
The main thing happening in the shot.

description:
One concise description of the shot.

people:
People visible in the shot.

objects:
Important visible objects.

framing:
Examples: close-up, medium shot, wide shot, screen recording.

camera_movement:
Examples: static, pan, tilt, zoom, handheld.

visual_importance:
Explain briefly why this shot may matter visually.

editing_relevance:
Explain whether this shot contains information that could matter when editing the video.
"""

    prompt = prompt.replace(
        '"shot_id": 0',
        f'"shot_id": {shot_id}'
    )

    try:

        response = client.models.generate_content(

            model=GEMINI_MODEL,

            contents=[

                types.Part.from_bytes(

                    data=image_bytes,

                    mime_type="image/jpeg"

                ),

                prompt

            ],

            config=types.GenerateContentConfig(

                temperature=0,

                response_mime_type="application/json"

            )

        )

        return json.loads(
            response.text
        )

    except Exception as e:

        print(
            f"Gemini failed for "
            f"shot {shot_id}: {e}"
        )

        return None


# ============================================================
# 16. ANALYZE SELECTED SHOTS WITH GEMINI
# ============================================================

def analyze_key_shots_with_gemini(
    video_path,
    key_shots
):

    print("\n========================================")
    print("GEMINI ANALYSIS")
    print("========================================")

    results = []

    for index, (_, shot) in enumerate(
        key_shots.iterrows()
    ):

        print(
            f"\nGemini "
            f"[{index + 1}/{len(key_shots)}]"
        )

        print(
            f"Shot {int(shot['shot_id'])}"
        )

        result = (
            analyze_shot_with_gemini(
                video_path,
                shot
            )
        )

        if result is not None:

            # Add timing information
            result["start"] = float(
                shot["start"]
            )

            result["end"] = float(
                shot["end"]
            )

            result["key_shot_score"] = float(
                shot["key_shot_score"]
            )

            results.append(
                result
            )

        # Small delay
        time.sleep(0.5)

    # Save
    with open(
        "gemini_results.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False
        )

    print(
        "\nSaved gemini_results.json"
    )

    return results


# ============================================================
# 17. MAIN PIPELINE
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("AI VIDEO ANALYSIS PIPELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # Check video
    # --------------------------------------------------------

    if not os.path.exists(
        VIDEO_PATH
    ):

        print(
            f"\nERROR:"
            f"\nVideo not found:"
            f"\n{VIDEO_PATH}"
        )

        return

    # --------------------------------------------------------
    # STEP 1
    # --------------------------------------------------------

    shots = detect_shots(
        VIDEO_PATH
    )

    # --------------------------------------------------------
    # STEP 2
    # --------------------------------------------------------

    transcript_segments = (
        transcribe_video(
            VIDEO_PATH
        )
    )

    # --------------------------------------------------------
    # STEP 3
    # --------------------------------------------------------

    shot_transcripts = (
        align_transcript_to_shots(
            shots,
            transcript_segments
        )
    )

    # --------------------------------------------------------
    # STEP 4
    # --------------------------------------------------------

    key_shots = find_key_shots(

        video_path=VIDEO_PATH,

        shots=shots,

        shot_transcripts=shot_transcripts,

        number=NUM_KEY_SHOTS

    )

    # --------------------------------------------------------
    # STEP 5
    # --------------------------------------------------------

    gemini_results = (
        analyze_key_shots_with_gemini(

            VIDEO_PATH,

            key_shots

        )
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    print("\n")
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    print(
        f"\nTotal shots: "
        f"{len(shots)}"
    )

    print(
        f"Gemini shots: "
        f"{len(key_shots)}"
    )

    print(
        f"Gemini API reduction: "
        f"{100 * (1 - len(key_shots) / len(shots)):.1f}%"
    )

    print(
        "\nFiles created:"
    )

    print(
        "  all_shot_scores.csv"
    )

    print(
        "  key_shots.csv"
    )

    print(
        "  gemini_results.json"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()