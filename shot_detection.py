from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector
import json
import sys


def detect_shots(video_path):

    video = open_video(video_path)

    scene_manager = SceneManager()

    scene_manager.add_detector(
        ContentDetector(
            threshold=27.0
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

    return shots


if __name__ == "__main__":

    video_path = sys.argv[1]
    output_path = sys.argv[2]

    shots = detect_shots(video_path)

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            {"shots": shots},
            f,
            indent=2
        )

    print(f"Detected {len(shots)} shots")

#python shot_detection.py my_test.mp4 shots.json