import os, io, time, json
from flask import Flask, request, jsonify
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import joblib
import numpy as np
from torchvision import transforms
from dotenv import load_dotenv

load_dotenv()

MODEL_PATH = os.getenv("MODEL_PATH", "models/model.joblib")
SCALER_PATH = os.getenv("SCALER_PATH", "models/scaler.joblib")
THRESHOLD = float(os.getenv("THRESHOLD", "0.75"))
MAX_MB = int(os.getenv("MAX_MB", "5"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "me-verifier-v1")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = MAX_MB * 1024 * 1024

mtcnn = MTCNN(image_size=160, margin=14)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

clf = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

@app.route('/healthz')
def health():
    return jsonify({"status":"ok", "model_version": MODEL_VERSION})

def allowed_file(filename):
    return filename.lower().endswith(('.png','.jpg','.jpeg'))

@app.route('/verify', methods=['POST'])
def verify():
    start = time.time()
    if 'image' not in request.files:
        return jsonify({"error":"image missing"}), 400
    f = request.files['image']
    if not allowed_file(f.filename):
        return jsonify({"error":"solo image/jpeg o image/png"}), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert('RGB')
    except:
        return jsonify({"error":"archivo inválido"}), 400

    face = mtcnn(img)
    if face is None:
        timing_ms = (time.time()-start)*1000
        return jsonify({"error":"no face detected", "timing_ms":timing_ms}), 400

    if isinstance(face, torch.Tensor):
        x = face
    else:
        x = preprocess(face).unsqueeze(0)
    if x is None:
        return jsonify({"error": "no se detectó ningún rostro en la imagen"}), 400
    if x.ndim == 3:
        x = x.unsqueeze(0)
    with torch.no_grad():
        emb = resnet(x).cpu().numpy()
    print(emb.shape)
    print("Scaler n_features_in_:", scaler.n_features_in_)

    emb_s = scaler.transform(emb)
    score = float(clf.predict_proba(emb_s)[:,1][0])
    is_me = bool(score >= THRESHOLD)

    timing_ms = (time.time()-start)*1000
    resp = {
        "model_version": MODEL_VERSION,
        "is_me": is_me,
        "score": round(score, 4),
        "threshold": THRESHOLD,
        "timing_ms": round(timing_ms, 2)
    }
    return jsonify(resp), 200

if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
