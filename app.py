import os
import cv2
import numpy as np

from flask import Flask, request, redirect, jsonify
from flask_cors import CORS
from urllib.parse import urlencode

from ultralytics import YOLO

import firebase_admin
from firebase_admin import credentials, firestore

# ================== CONFIG ==================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PT_MODEL_PATH = os.getenv(
    "PT_MODEL_PATH",
    os.path.join(BASE_DIR, "best.pt"),
)

CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.35"))

FIREBASE_KEY_PATH = os.getenv(
    "FIREBASE_KEY_PATH",
    r"serviceAccountKey.json",
)

# Firestore collection to store verification JSON
FIRESTORE_COLLECTION = os.getenv("FIRESTORE_COLLECTION", "ai_verifications")

CLASS_NAMES = [
    "clip_ok",
    "liner_ok",
    "pad_ok",
    "sleeper_ok",
    "bolt_ok",
    "erc_ok",
    "clip_faulty",
    "liner_faulty",
    "pad_faulty",
    "sleeper_faulty",
    "bolt_faulty",
    "erc_faulty",
    "clip_rust",
    "liner_rust",
    "pad_rust",
    "sleeper_rust",
    "bolt_rust",
    "erc_rust",
    "clip_missing",
    "liner_missing",
    "pad_missing",
    "sleeper_missing",
    "bolt_missing",
    "erc_missing",
    "qr_code",
]

# ================== APP ==================

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

if not os.path.exists(PT_MODEL_PATH):
    raise FileNotFoundError(f"PT model not found at: {PT_MODEL_PATH}")

model = YOLO(PT_MODEL_PATH)

# ================== FIREBASE INIT (Firestore) ==================

db_fs = None

def init_firestore():
    global db_fs
    try:
        if not FIREBASE_KEY_PATH or not os.path.exists(FIREBASE_KEY_PATH):
            print("[Firebase] Firestore skipped: key not found:", FIREBASE_KEY_PATH)
            return

        if not firebase_admin._apps:
            cred = credentials.Certificate(FIREBASE_KEY_PATH)
            firebase_admin.initialize_app(cred)

        db_fs = firestore.client()
        print("[Firebase] Firestore initialized OK")
    except Exception as e:
        print("[Firebase] Firestore init error:", e)

def save_result_to_firestore(result_json: dict, material_id: str = "", source: str = ""):
    """
    Stores the EXACT result JSON + some metadata.
    """
    if db_fs is None:
        return None

    doc = dict(result_json)  # copy exact JSON
    doc["materialId"] = material_id or None
    doc["source"] = source or None
    doc["threshold"] = CONF_THRESHOLD
    doc["model"] = os.path.basename(PT_MODEL_PATH)
    doc["createdAt"] = firestore.SERVER_TIMESTAMP

    ref = db_fs.collection(FIRESTORE_COLLECTION).document()
    doc["id"] = ref.id
    ref.set(doc)
    return ref.id

init_firestore()

# ================== HELPERS ==================

def parse_component(class_name: str):
    name = (class_name or "").lower()
    for comp in ["erc", "liner", "sleeper", "clip", "pad", "bolt"]:
        if comp in name:
            return comp
    return None

def run_pt_inference(image_bgr):
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    results = model.predict(
        source=image_rgb,
        conf=CONF_THRESHOLD,
        imgsz=640,
        device="cpu",
        verbose=False,
    )

    r = results[0]
    if r.boxes is None or len(r.boxes) == 0:
        return None

    best = None
    best_conf = -1.0

    for b in r.boxes:
        conf = float(b.conf.item()) if hasattr(b.conf, "item") else float(b.conf)
        cls_id = int(b.cls.item()) if hasattr(b.cls, "item") else int(b.cls)

        if hasattr(r, "names") and isinstance(r.names, dict) and cls_id in r.names:
            class_name = r.names[cls_id]
        elif cls_id < len(CLASS_NAMES):
            class_name = CLASS_NAMES[cls_id]
        else:
            class_name = str(cls_id)

        comp = parse_component(class_name)
        if comp is None:
            continue

        if conf > best_conf:
            best_conf = conf
            best = {
                "component": comp,
                "confidence": conf,
                "class_name": class_name,
            }

    return best

# ================== ROUTES ==================

@app.route("/")
def home():
    return redirect("/verify")

@app.route("/health")
def health():
    return jsonify({
        "ok": True,
        "ptModelPath": PT_MODEL_PATH,
        "threshold": CONF_THRESHOLD,
        "firestore": bool(db_fs),
        "collection": FIRESTORE_COLLECTION,
    })

@app.route("/verify", methods=["GET"])
def verify_page():
    callback = request.args.get("callback", "")
    mid = request.args.get("materialId", "")

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>AI Verify (PT)</title>
  <style>
    body{{font-family:system-ui;background:#f7e8ff;margin:0;display:flex;min-height:100vh;align-items:center;justify-content:center}}
    .card{{background:#fff;padding:18px;border-radius:18px;max-width:420px;width:92%;box-shadow:0 10px 30px rgba(0,0,0,.08)}}
    button{{padding:10px 14px;border:0;border-radius:999px;background:#A259FF;color:#fff;font-weight:700;cursor:pointer;width:100%}}
    input{{width:100%;margin:10px 0}}
    .muted{{color:#666;font-size:12px}}
  </style>
</head>
<body>
  <div class="card">
    <h3 style="margin:0;color:#4B3A7A">Track Fitting AI Verification (best.pt)</h3>
    <p class="muted">Material: <b>{mid or "-"}</b></p>

    <form method="POST" action="/api/verify_web" enctype="multipart/form-data">
      <input type="hidden" name="callback" value="{callback}"/>
      <input type="hidden" name="materialId" value="{mid}"/>
      <input type="file" name="image" accept="image/*" capture="environment" required/>
      <button type="submit">Run AI Verification</button>
    </form>

    <p class="muted" style="margin-top:10px">No photo is stored. Only result JSON is stored in Firestore.</p>
  </div>
</body>
</html>
"""

@app.route("/api/verify_web", methods=["POST"])
def verify_web():
    file = request.files.get("image")
    callback = request.form.get("callback", "")
    material_id = request.form.get("materialId", "")

    if not file:
        return "No image", 400

    file_bytes = file.read()
    file_array = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return "Failed to decode image", 400

    det = run_pt_inference(img_bgr)

    if det is None:
        result_json = {
            "ok": True,
            "status": "not_detected",
            "component": None,
            "confidence": 0.0,
        }
    else:
        result_json = {
            "ok": True,
            "status": "verified",
            "component": det["component"],
            "confidence": float(det["confidence"]),
        }

    # ✅ store EXACT JSON in Firestore
    doc_id = save_result_to_firestore(result_json, material_id=material_id, source="verify_web")

    # redirect back if callback exists
    if callback:
        q = urlencode({
            "id": material_id,
            "aiStatus": result_json["status"],
            "aiComponent": result_json["component"] or "",
            "aiConfidence": f'{float(result_json["confidence"]):.6f}',
            "aiDocId": doc_id or "",
        })
        return redirect(f"{callback}?{q}")

    # otherwise return json + docId
    result_json["docId"] = doc_id
    return jsonify(result_json)

@app.route("/api/verify", methods=["POST", "OPTIONS"])
def api_verify():
    if request.method == "OPTIONS":
        return ("", 204)

    file = request.files.get("image")
    if not file:
        return jsonify({"ok": False, "error": "image is required"}), 400

    material_id = request.form.get("materialId", "")

    file_bytes = file.read()
    file_array = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(file_array, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({"ok": False, "error": "failed to decode image"}), 400

    det = run_pt_inference(img_bgr)

    if det is None:
        result_json = {
            "ok": True,
            "status": "not_detected",
            "component": None,
            "confidence": 0.0,
        }
    else:
        result_json = {
            "ok": True,
            "status": "verified",
            "component": det["component"],
            "confidence": float(det["confidence"]),
        }

    # ✅ store EXACT JSON in Firestore
    doc_id = save_result_to_firestore(result_json, material_id=material_id, source="api_verify")

    result_json["docId"] = doc_id
    return jsonify(result_json)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
