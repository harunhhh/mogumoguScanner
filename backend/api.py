import os

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io

from ai_logic import FoodAI

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = FastAPI(title="もぐもぐスキャナー API")

# 本番のVercelドメイン（例: https://xxxx.vercel.app）をカンマ区切りで指定
_PROD_ORIGINS = [o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "").split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_PROD_ORIGINS,
    # 開発時のみ: スマホなど同じLAN内の端末からのアクセスも許可する（localhost + 主要なプライベートIP帯 + ポート3000）
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1|192\.168\.\d{1,3}\.\d{1,3}|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[0-1])\.\d{1,3}\.\d{1,3}):3000",
    allow_methods=["*"],
    allow_headers=["*"],
)

ai = FoodAI(
    model_path=os.path.join(_BASE_DIR, "food_model.keras"),
    csv_path=os.path.join(_BASE_DIR, "food_calories.csv"),
)


def _to_jsonable(value):
    if hasattr(value, "item"):
        return value.item()
    return value


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg", "image/webp"):
        raise HTTPException(status_code=400, detail="対応していない画像形式です。")

    raw = await file.read()
    try:
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="画像として読み込めませんでした。")

    result = ai.predict(image)
    return {
        "name": result["name"],
        "confidence": _to_jsonable(result["confidence"]),
        "calories": _to_jsonable(result["calories"]),
        "portion": _to_jsonable(result["portion"]),
        "full_name": result["full_name"],
        "determined": result.get("determined", True),
        "top3": [
            {
                "name": c["name"],
                "confidence": _to_jsonable(c["confidence"]),
                "calories": _to_jsonable(c["calories"]),
                "portion": _to_jsonable(c["portion"]),
            }
            for c in result.get("top3", [])
        ],
    }


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ai.model is not None}
