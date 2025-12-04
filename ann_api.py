from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import cv2
import joblib
import os
from tensorflow.keras.models import load_model


# === CONFIGURATION ===
IMAGE_TMP_FOLDER = "tmp_uploads"
MODEL_PATH = os.path.join("ANN", "sweetness_model_ann.h5")
SCALER_PATH = os.path.join("ANN", "scaler_ann.pkl")

os.makedirs(IMAGE_TMP_FOLDER, exist_ok=True)


# === LOAD MODEL AND SCALER AT STARTUP ===
model_ann = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)


def extract_color_features(image_bgr: np.ndarray):
    """Return mean R,G,B,H,S,V as in the training code."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    mean_R = np.mean(img_rgb[:, :, 0])
    mean_G = np.mean(img_rgb[:, :, 1])
    mean_B = np.mean(img_rgb[:, :, 2])
    mean_H = np.mean(hsv[:, :, 0])
    mean_S = np.mean(hsv[:, :, 1])
    mean_V = np.mean(hsv[:, :, 2])

    return mean_R, mean_G, mean_B, mean_H, mean_S, mean_V


def detect_defects(image_bgr: np.ndarray) -> float:
    """Robust defect detection, same logic as in ANN notebook."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    v_thresh = np.mean(v) - 0.5 * np.std(v)
    s_thresh = np.mean(s) - 0.5 * np.std(s)

    defect_mask = (v < v_thresh) | (s < s_thresh)
    kernel = np.ones((3, 3), np.uint8)
    defect_mask = cv2.morphologyEx(defect_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    defect_area = np.sum(defect_mask)
    total_area = image_bgr.shape[0] * image_bgr.shape[1]
    return float((defect_area / total_area) * 100.0)


def compute_volume_cm3(length_mm: float, width_mm: float, thickness_mm: float) -> float:
    volume_mm3 = (4.0 / 3.0) * np.pi * (length_mm / 2.0) * (width_mm / 2.0) * (thickness_mm / 2.0)
    return float(volume_mm3 / 1000.0)


def compute_mango_quality(weight_g: float, volume_cm3: float, predicted_sweetness_brix: float, defect_pct: float) -> str:
    if weight_g < 200 or predicted_sweetness_brix < 8 or defect_pct > 15:
        return "Reject"
    if (
        350 <= weight_g <= 800
        and 200 <= volume_cm3 <= 600
        and predicted_sweetness_brix >= 12
        and defect_pct <= 5
    ):
        return "Grade A"
    if (
        200 <= weight_g < 350
        or 150 <= volume_cm3 < 200
        or 10 <= predicted_sweetness_brix < 12
        or 5 < defect_pct <= 10
    ):
        return "Grade B"
    if (
        801 <= weight_g <= 1000
        or 601 <= volume_cm3 <= 700
        or 8 <= predicted_sweetness_brix < 10
        or 10 < defect_pct <= 15
    ):
        return "Grade C"
    return "Unclassified"


class PredictionResponse(BaseModel):
    predicted_sweetness_brix: float
    defect_pct: float
    volume_cm3: float
    quality_grade: str


app = FastAPI(title="Mango Sweetness ANN API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Mango ANN API is running"}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    length_mm: float = Form(...),
    width_mm: float = Form(...),
    thickness_mm: float = Form(...),
    weight_g: float = Form(...),
):
    image_bytes = await image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Invalid image file.")

    mean_R, mean_G, mean_B, mean_H, mean_S, mean_V = extract_color_features(img_bgr)
    defect_pct = detect_defects(img_bgr)
    volume_cm3 = compute_volume_cm3(length_mm, width_mm, thickness_mm)

    features = np.array(
        [
            [
                length_mm,
                width_mm,
                thickness_mm,
                weight_g,
                mean_R,
                mean_G,
                mean_B,
                mean_H,
                mean_S,
                mean_V,
                defect_pct,
            ]
        ],
        dtype=np.float32,
    )

    features_scaled = scaler.transform(features)
    predicted_sweetness = float(model_ann.predict(features_scaled).flatten()[0])

    quality_grade = compute_mango_quality(
        weight_g=weight_g,
        volume_cm3=volume_cm3,
        predicted_sweetness_brix=predicted_sweetness,
        defect_pct=defect_pct,
    )

    return PredictionResponse(
        predicted_sweetness_brix=predicted_sweetness,
        defect_pct=defect_pct,
        volume_cm3=volume_cm3,
        quality_grade=quality_grade,
    )


if __name__ == "__main__":
    uvicorn.run("ann_api:app", host="0.0.0.0", port=8000)



