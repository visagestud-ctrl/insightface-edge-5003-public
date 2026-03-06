import base64
import os
from typing import List, Dict, Any, Tuple
from threading import Lock

import cv2
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app)


LANDMARK_106_NAMES = {
    51: "nasion", 54: "rhinion", 58: "pronasale", 61: "subnasale",
    66: "alar_r", 67: "alar_l", 72: "mouth_r", 75: "lip_up_center",
    78: "mouth_l", 81: "lip_low_center", 97: "gonion_r", 100: "menton",
    103: "gonion_l", 105: "forehead"
}


def _init_model() -> Any:
    from insightface.app import FaceAnalysis

    det_size = int(os.getenv("DET_SIZE", "384"))
    providers = ["CPUExecutionProvider"]
    fa = FaceAnalysis(providers=providers, allowed_modules=["detection", "landmark_2d_106"])
    fa.prepare(ctx_id=0, det_size=(det_size, det_size))
    return fa


app_face = None
_model_lock = Lock()


def get_model() -> Any:
    global app_face
    if app_face is not None:
        return app_face
    with _model_lock:
        if app_face is None:
            app_face = _init_model()
    return app_face


def find_nearest_edge(img_gray: np.ndarray, x: float, y: float, radius: int = 15) -> Tuple[int, int]:
    h, w = img_gray.shape
    x_i, y_i = int(x), int(y)

    x1 = max(0, x_i - radius)
    y1 = max(0, y_i - radius)
    x2 = min(w, x_i + radius)
    y2 = min(h, y_i + radius)

    region = img_gray[y1:y2, x1:x2]
    if region.size == 0:
        return x_i, y_i

    region_blur = cv2.GaussianBlur(region, (3, 3), 0)
    edges = cv2.Canny(region_blur, 50, 150)
    edge_points = np.where(edges > 0)

    if len(edge_points[0]) == 0:
        return x_i, y_i

    orig_local_x = x_i - x1
    orig_local_y = y_i - y1
    distances = np.sqrt((edge_points[1] - orig_local_x) ** 2 + (edge_points[0] - orig_local_y) ** 2)

    min_idx = int(np.argmin(distances))
    if distances[min_idx] > radius:
        return x_i, y_i

    new_x = x1 + int(edge_points[1][min_idx])
    new_y = y1 + int(edge_points[0][min_idx])
    return new_x, new_y


def refine_all_landmarks(img_gray: np.ndarray, landmarks: List[Dict[str, Any]], strength: str = "medium") -> List[Dict[str, Any]]:
    radius_map = {"light": 5, "medium": 10, "strong": 15, "off": 0}
    radius = radius_map.get(strength, 10)

    contour_indices = set(range(96, 106))
    mouth_indices = set(range(72, 96))
    nose_tip_indices = {58, 59, 60, 61}

    if radius <= 0:
        for p in landmarks:
            p["refined"] = False
        return landmarks

    refined = []
    for lm in landmarks:
        x = float(lm["x"])
        y = float(lm["y"])
        idx = int(lm["index"])

        if idx in contour_indices:
            new_x, new_y = find_nearest_edge(img_gray, x, y, radius=radius + 5)
        elif idx in nose_tip_indices:
            new_x, new_y = find_nearest_edge(img_gray, x, y, radius=radius)
        elif idx in mouth_indices:
            new_x, new_y = find_nearest_edge(img_gray, x, y, radius=max(3, radius - 2))
        else:
            new_x, new_y = find_nearest_edge(img_gray, x, y, radius=max(3, radius - 5))

        moved = (new_x != int(x)) or (new_y != int(y))
        refined.append({
            "index": idx,
            "name": lm.get("name", f"point_{idx}"),
            "x": float(new_x),
            "y": float(new_y),
            "x_norm": lm.get("x_norm", 0),
            "y_norm": lm.get("y_norm", 0),
            "refined": moved,
            "original_x": float(x) if moved else None,
            "original_y": float(y) if moved else None,
        })

    return refined


def decode_image_from_request() -> np.ndarray:
    if "image" in request.files:
        file = request.files["image"]
        img_bytes = file.read()
    else:
        payload = request.get_json(silent=True) or {}
        image_base64 = str(payload.get("imageBase64", "")).strip()
        if not image_base64:
            return None
        img_bytes = base64.b64decode(image_base64)

    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def prepare_inference_image(img: np.ndarray, max_dim: int = 960) -> Tuple[np.ndarray, float]:
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_dim:
        return img, 1.0
    scale = max_dim / float(longest)
    out_w = max(1, int(round(w * scale)))
    out_h = max(1, int(round(h * scale)))
    resized = cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
    return resized, scale


@app.route("/health", methods=["GET"])
def health() -> Any:
    return jsonify({"ok": True, "service": "insightface-edge", "model": "landmark_2d_106"})


@app.route("/detect", methods=["POST"])
def detect() -> Any:
    img = decode_image_from_request()
    if img is None:
        return jsonify({"success": False, "error": "No image provided"}), 400

    try:
        model = get_model()
    except Exception as e:
        return jsonify({"success": False, "error": f"Model init failed: {e}"}), 200

    edge_strength = request.form.get("edge_strength")
    if not edge_strength:
        body = request.get_json(silent=True) or {}
        edge_strength = str(body.get("edge_strength", "strong"))

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    infer_img, scale = prepare_inference_image(img, max_dim=int(os.getenv("MAX_INFER_DIM", "960")))
    infer_h, infer_w = infer_img.shape[:2]

    faces = model.get(infer_img)
    if len(faces) == 0:
        img_flipped = cv2.flip(infer_img, 1)
        faces = model.get(img_flipped)
        if len(faces) > 0:
            for face in faces:
                if hasattr(face, "landmark_2d_106") and face.landmark_2d_106 is not None:
                    face.landmark_2d_106[:, 0] = infer_w - face.landmark_2d_106[:, 0]

    if len(faces) == 0:
        return jsonify({"success": False, "error": "No face detected"}), 200

    face = faces[0]
    if not hasattr(face, "landmark_2d_106") or face.landmark_2d_106 is None:
        return jsonify({"success": False, "error": "No landmarks detected"}), 200

    lm = face.landmark_2d_106
    points = []
    for i in range(len(lm)):
        x = float(lm[i][0]) / scale
        y = float(lm[i][1]) / scale
        x = max(0.0, min(float(w - 1), x))
        y = max(0.0, min(float(h - 1), y))
        points.append({
            "index": i,
            "name": LANDMARK_106_NAMES.get(i, f"point_{i}"),
            "x": x,
            "y": y,
            "x_norm": round(x / w, 4),
            "y_norm": round(y / h, 4),
        })

    points = refine_all_landmarks(gray, points, edge_strength)

    return jsonify({
        "success": True,
        "width": w,
        "height": h,
        "num_faces": len(faces),
        "landmarks": points,
    })


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5003"))
    app.run(host="0.0.0.0", port=port, debug=False)
