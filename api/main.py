"""
main.py — FastAPI backend cho hệ gợi ý xe KNN.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

# Cho phép import recommender.py ở root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from recommender import (  # noqa: E402
    ALL_GROUPS,
    CATEGORICAL_COLS,
    CarRecommender,
    DEFAULT_WEIGHTS,
    normalize_weights,
)
from api.llm_explain import explain_recommendations

MODEL_PATH = BASE_DIR / "model" / "car_recommender.joblib"
META_PATH = BASE_DIR / "model" / "model_metadata.json"

if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Không thấy model tại {MODEL_PATH}.\n"
        f"Hãy chạy trước: python model/train_export.py"
    )

# Nạp 1 lần khi khởi động — không đọc lại CSV mỗi request
print(f"[FastAPI] Đang nạp model: {MODEL_PATH}")
bundle = joblib.load(MODEL_PATH)
recommender = CarRecommender.from_bundle(bundle)

with open(META_PATH, encoding="utf-8") as f:
    META = json.load(f)
print(f"[FastAPI] Model version {META['version']} — {META['n_samples']} dòng dữ liệu")

VALID_CATEGORICAL = {
    col: set(META["categories"][col]) for col in CATEGORICAL_COLS
}

app = FastAPI(
    title="Car Recommendation API",
    description="API gợi ý ô tô sử dụng KNN content-based có trọng số",
    version=META["version"],
)

class QueryCriteria(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    price: float | None = Field(default=None, alias="Giá( VNĐ)", gt=0)
    power: float | None = Field(default=None, alias="Công suất(HP)", gt=0)
    seats: int | None = Field(default=None, alias="Chỗ ngồi", ge=1)

    fuel: str | None = Field(default=None, alias="Nhiên liệu")
    purpose: str | None = Field(default=None, alias="Mục đích")
    condition: str | None = Field(default=None, alias="Tình trạng")
    car_type: str | None = Field(default=None, alias="Kiểu xe")
    transmission: str | None = Field(default=None, alias="Hộp số")

class RecommendRequest(BaseModel):
    criteria: QueryCriteria
    weights: dict[str, float] = Field(default_factory=dict)
    k: int = Field(default=5, ge=1, le=20)

class CompareRequest(BaseModel):
    car_names: list[str] = Field(..., min_length=2, max_length=3)

def criteria_to_query(criteria: QueryCriteria) -> dict:
    raw = {
        "Giá( VNĐ)": criteria.price,
        "Công suất(HP)": criteria.power,
        "Chỗ ngồi": criteria.seats,
        "Nhiên liệu": criteria.fuel,
        "Mục đích": criteria.purpose,
        "Tình trạng": criteria.condition,
        "Kiểu xe": criteria.car_type,
        "Hộp số": criteria.transmission,
    }
    return {k: v for k, v in raw.items() if v is not None}

def validate_criteria(query: dict) -> None:
    for col, value in query.items():
        if col in CATEGORICAL_COLS:
            if str(value).strip() not in VALID_CATEGORICAL[col]:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Giá trị '{value}' không hợp lệ cho '{col}'. "
                        f"Các giá trị hợp lệ: {sorted(VALID_CATEGORICAL[col])}"
                    ),
                )

def validate_weights(weights: dict[str, float]) -> dict[str, float]:
    unknown = set(weights) - set(ALL_GROUPS)
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Trọng số có key không hợp lệ: {sorted(unknown)}",
        )
    for name, val in weights.items():
        if not (isinstance(val, (int, float)) and val >= 0 and val != float("inf")):
            raise HTTPException(
                status_code=422,
                detail=f"Trọng số của '{name}' phải là số không âm hữu hạn.",
            )
    merged = {g: weights.get(g, DEFAULT_WEIGHTS[g]) for g in ALL_GROUPS}
    return normalize_weights(merged)

@app.get("/model-info")
def model_info():
    return META

@app.post("/recommend")
def recommend_cars(request: RecommendRequest):
    query = criteria_to_query(request.criteria)
    validate_criteria(query)
    weights = validate_weights(request.weights)

    try:
        result = recommender.recommend(query, weights=weights, k=request.k)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Lỗi khi gợi ý xe: {exc}") from exc

    recommendations = result.to_dict(orient="records")
    explanation = explain_recommendations(
        query=query, weights=weights, recommendations=recommendations,
    )

    return {
        "query": query,
        "weights": weights,
        "count": len(recommendations),
        "recommendations": recommendations,
        "explanation": explanation,
    }

@app.post("/compare")
def compare_cars(request: CompareRequest):
    found = []
    for name in request.car_names:
        match = recommender.df[recommender.df["Tên xe"] == name]
        if match.empty:
            raise HTTPException(
                status_code=404,
                detail=f"Không tìm thấy xe có tên '{name}'",
            )
        car_dict = match.iloc[0].to_dict()
        # Ép kiểu dữ liệu Numpy sang Python gốc để tránh lỗi JSON
        car_dict = {k: (v.item() if hasattr(v, 'item') else v) for k, v in car_dict.items()}
        found.append(car_dict)
    
    return {"cars": found}