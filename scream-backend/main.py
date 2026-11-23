import io

import json

import librosa
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from yamnet_numpy import YamnetNumpy

SAMPLE_RATE = 16000
MAX_SECONDS = 10.0        # same as training
MIN_SAMPLES = 16000       # same as training (pad very short clips to 1 second)

app = FastAPI(title="Scream Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Trained classifier, saved as plain numbers (no scikit-learn needed)
with open("scream_classifier_weights.json", encoding="utf-8") as f:
    _cfg = json.load(f)
WEIGHTS = np.array(_cfg["weights"], dtype=np.float64)
BIAS = float(_cfg["bias"])
THRESHOLD = float(_cfg.get("threshold", 0.5))


def scream_probability(embedding):
    logit = float(np.clip(embedding.astype(np.float64) @ WEIGHTS + BIAS, -30, 30))
    return 1.0 / (1.0 + np.exp(-logit))

# YAMNet written in plain numpy: no TensorFlow needed
yamnet = YamnetNumpy("yamnet_weights.npz")


@app.get("/")
def home():
    return {"message": "Scream Detection API is running"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        try:
            audio, _ = librosa.load(io.BytesIO(audio_bytes), sr=SAMPLE_RATE,
                                    mono=True, duration=MAX_SECONDS)
        except Exception:
            raise HTTPException(status_code=400, detail="Could not read this audio file")
        if audio.size == 0:
            raise HTTPException(status_code=400, detail="No audio detected")

        audio = audio.astype(np.float32)
        if audio.size < MIN_SAMPLES:
            audio = np.pad(audio, (0, MIN_SAMPLES - audio.size))

        embedding = yamnet.embedding(audio)
        scream_prob = scream_probability(embedding)

        is_scream = scream_prob >= THRESHOLD
        confidence = scream_prob if is_scream else 1.0 - scream_prob
        return {
            "predicted_class": "scream" if is_scream else "not_scream",
            "confidence": round(confidence * 100, 2),
            "scream_probability": round(scream_prob * 100, 2),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))