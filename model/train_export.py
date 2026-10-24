"""
train_export.py — Export "model" (bundle) + metadata cho hệ gợi ý xe KNN.

Chạy 1 lần từ thư mục root của project:
    python model/train_export.py

Sinh ra:
    model/car_recommender.joblib   — bundle (scaler, cat_col_map, feature_matrix, df)
    model/model_metadata.json      — mô tả model (version, ranges, categories, ...)

Sau khi chạy, api/main.py sẽ nạp bundle này thay vì đọc CSV mỗi lần khởi động.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import joblib

# Cho phép import recommender.py ở root
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from recommender import (  # noqa: E402
    ALL_GROUPS,
    CATEGORICAL_COLS,
    DEFAULT_WEIGHTS,
    NUMERIC_COLS,
    CarRecommender,
    load_and_clean_data,
)

DATA_PATH = BASE_DIR / "data" / "car0.csv"
MODEL_DIR = BASE_DIR / "model"
MODEL_PATH = MODEL_DIR / "car_recommender.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"


def export() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không thấy dataset: {DATA_PATH}. Đặt car0.csv vào thư mục data/."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[1/3] Đọc dữ liệu từ {DATA_PATH}")
    df = load_and_clean_data(str(DATA_PATH))
    print(f"      → {len(df)} dòng, {len(df.columns)} cột")

    print("[2/3] Fit CarRecommender (MinMaxScaler + One-hot)")
    rec = CarRecommender(df)

    bundle = rec.to_bundle()
    joblib.dump(bundle, MODEL_PATH)
    size_kb = MODEL_PATH.stat().st_size / 1024
    print(f"      → Đã lưu bundle: {MODEL_PATH.relative_to(BASE_DIR)} ({size_kb:.1f} KB)")

    print("[3/3] Ghi metadata")
    metadata = {
        "model_name": "car_knn_recommender",
        "version": "1.0.0",
        "algorithm": "Weighted KNN (content-based)",
        "distance": "Euclidean có trọng số",
        "scaler": "MinMaxScaler",
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "numeric_features": NUMERIC_COLS,
        "categorical_features": CATEGORICAL_COLS,
        "all_groups": ALL_GROUPS,
        "default_weights": DEFAULT_WEIGHTS,
        "categories": {
            c: sorted(df[c].dropna().astype(str).unique().tolist())
            for c in CATEGORICAL_COLS
        },
        "feature_ranges": {
            c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC_COLS
        },
        "n_samples": int(len(df)),
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"      → Đã lưu metadata: {META_PATH.relative_to(BASE_DIR)}")

    print("\n✓ Hoàn tất. Bước tiếp theo:")
    print("    uvicorn api.main:app --reload --port 8000")


if __name__ == "__main__":
    export()