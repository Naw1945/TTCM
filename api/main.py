
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from recommender import (
    ALL_GROUPS,
    CATEGORICAL_COLS,
    CarRecommender,
    DEFAULT_WEIGHTS,
    load_and_clean_data,
    normalize_weights,
)
from api.llm_explain import explain_recommendations


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "car0.csv"

app = FastAPI(
    title="Car Recommendation API",
    description="API gợi ý ô tô sử dụng KNN",
    version="1.0.0",
)

df = load_and_clean_data(str(DATA_PATH))
recommender = CarRecommender(df)


class QueryCriteria(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    price: float = Field(alias="Giá( VNĐ)", gt=0)
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


def get_query_dict(criteria: QueryCriteria) -> dict:
    return {
        key: value
        for key, value in {
            "Giá( VNĐ)": criteria.price,
            "Công suất(HP)": criteria.power,
            "Chỗ ngồi": criteria.seats,
            "Nhiên liệu": criteria.fuel,
            "Mục đích": criteria.purpose,
            "Tình trạng": criteria.condition,
            "Kiểu xe": criteria.car_type,
            "Hộp số": criteria.transmission,
        }.items()
        if value is not None
    }


def validate_criteria(query: dict) -> None:
    for column, value in query.items():
        if column not in df.columns:
            raise HTTPException(
                status_code=422,
                detail=f"Dataset không có cột: {column}",
            )

        if column in CATEGORICAL_COLS:
            available_values = set(
                df[column].dropna().astype(str).str.strip()
            )

            if str(value).strip() not in available_values:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"Giá trị '{value}' không hợp lệ "
                        f"cho '{column}'."
                    ),
                )


@app.get("/model-info")
def model_info():
    return {
        "model": "KNN",
        "scaler": "MinMaxScaler",
        "dataset_rows": len(df),
        "criteria": ALL_GROUPS,
        "default_weights": DEFAULT_WEIGHTS,
    }


@app.post("/recommend")
def recommend_cars(request: RecommendRequest):
    query = get_query_dict(request.criteria)
    validate_criteria(query)

    unknown_weights = set(request.weights) - set(ALL_GROUPS)

    if unknown_weights:
        raise HTTPException(
            status_code=422,
            detail=(
                "Tiêu chí trọng số không hợp lệ: "
                f"{sorted(unknown_weights)}"
            ),
        )

    for name, value in request.weights.items():
        if not 0 <= value < float("inf"):
            raise HTTPException(
                status_code=422,
                detail=f"Trọng số của '{name}' phải là số không âm hữu hạn.",
            )

    # Bổ sung trọng số mặc định cho các tiêu chí bị thiếu.
    raw_weights = {
        group: request.weights.get(group, DEFAULT_WEIGHTS[group])
        for group in ALL_GROUPS
    }
    weights = normalize_weights(raw_weights)

    try:
        result = recommender.recommend(
            query,
            weights=weights,
            k=request.k,
        )

        recommendations = result.to_dict(orient="records")

        explanation = explain_recommendations(
            query=query,
            weights=weights,
            recommendations=recommendations,
        )

        return {
            "query": query,
            "weights": weights,
            "count": len(recommendations),
            "recommendations": recommendations,
            "explanation": explanation,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi khi gợi ý xe: {str(exc)}",
        ) from exc
