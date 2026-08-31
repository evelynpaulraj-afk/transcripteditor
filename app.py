from flask import Flask, jsonify, request, render_template, send_file
import json
import sys
import os

app = Flask(__name__)

PROJECT_PATH = sys.argv[1] if len(sys.argv) > 1 else "project.json"


def load_project():
    with open(PROJECT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_project(project):
    with open(PROJECT_PATH, "w", encoding="utf-8") as f:
        json.dump(project, f, indent=2, ensure_ascii=False)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/project", methods=["GET"])
def get_project():
    project = load_project()
    # Always serve segments sorted by current order, so the UI reflects saved state
    project["segments"].sort(key=lambda s: s["order"])
    return jsonify(project)


@app.route("/api/project", methods=["POST"])
def update_project():
    """
    Expects: {"segments": [{"id": "seg-0003", "order": 0, "included": true}, ...]}
    Only order/included are updated — start/end/text/words are never touched here,
    since those describe the ORIGINAL video and must stay accurate for ffmpeg.
    """
    updates = request.json["segments"]
    project = load_project()
    segment_by_id = {s["id"]: s for s in project["segments"]}

    for u in updates:
        if u["id"] in segment_by_id:
            segment_by_id[u["id"]]["order"] = u["order"]
            segment_by_id[u["id"]]["included"] = u["included"]

    save_project(project)
    return jsonify({"status": "ok"})


@app.route("/video")
def video():
    project = load_project()
    video_path = os.path.abspath(project["video_path"])
    return send_file(video_path, conditional=True)  # conditional=True enables seeking (range requests)


if __name__ == "__main__":
    print(f"Editing: {PROJECT_PATH}")
    app.run(debug=True, port=5000)