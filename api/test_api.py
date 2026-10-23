
import time

import pytest
from fastapi.testclient import TestClient

from api.main import app, df


client = TestClient(app)


def test_model_info():
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()
    assert data["model"] == "KNN"
    assert data["scaler"] == "MinMaxScaler"
    assert data["dataset_rows"] > 0


def test_recommend_valid_request():
    payload = {
        "criteria": {
            "Giá( VNĐ)": float(df["Giá( VNĐ)"].max()),
        },
        "weights": {},
        "k": 5,
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200

    data = response.json()
    assert "recommendations" in data
    assert "count" in data
    assert data["count"] == len(data["recommendations"])


def test_recommend_invalid_request():
    payload = {
        "criteria": {
            "Giá( VNĐ)": -100,
        },
        "weights": {},
        "k": 5,
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code == 422


def test_recommend_budget_too_low():
    payload = {
        "criteria": {
            "Giá( VNĐ)": 1,
        },
        "weights": {},
        "k": 5,
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200
    assert response.json()["recommendations"] == []


def test_recommend_response_time():
    payload = {
        "criteria": {
            "Giá( VNĐ)": float(df["Giá( VNĐ)"].max()),
        },
        "weights": {},
        "k": 5,
    }

    start = time.perf_counter()
    response = client.post("/recommend", json=payload)
    elapsed = time.perf_counter() - start

    assert response.status_code == 200
    print(f"\nThời gian phản hồi: {elapsed * 1000:.2f} ms")

    # Chỉ kiểm tra hiệu năng KNN khi chạy test.
    # Không áp ngưỡng cứng vì thời gian máy và Gemini có thể biến động.
