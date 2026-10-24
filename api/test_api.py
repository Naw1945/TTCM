import sys
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import được module api
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_model_info():
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["algorithm"] == "Weighted KNN (content-based)"
    assert data["n_samples"] > 0

def test_recommend_basic():
    # Dùng ngân sách 20 tỷ để chắc chắn không loại chiếc xe nào
    payload = {
        "criteria": {"Giá( VNĐ)": 20000000000, "Chỗ ngồi": 5}, 
        "weights": {},
        "k": 3
    }
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 3
    # Đảm bảo bộ lọc cứng hoạt động
    for car in data["recommendations"]:
        assert car["Giá( VNĐ)"] <= 20000000000
        assert car["Chỗ ngồi"] >= 5

def test_recommend_price_optional():
    # Test bug 4 đã fix: Bỏ trống ngân sách không bị lỗi 422
    payload = {"criteria": {"Nhiên liệu": "Điện"}, "weights": {}, "k": 2}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    assert response.json()["count"] == 2

def test_recommend_invalid_category():
    # Test chặn dữ liệu rác (Nhiên liệu không có trong data)
    payload = {"criteria": {"Nhiên liệu": "Nước"}, "weights": {}, "k": 2}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 422

def test_compare_cars():
    payload = {"car_names": ["VIOS 1.5E-MT", "VIOS 1.5E-CVT"]}
    response = client.post("/compare", json=payload)
    assert response.status_code == 200
    assert len(response.json()["cars"]) == 2