import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"


def load_history() -> list:
    """Đọc toàn bộ lịch sử từ file JSON. Trả về [] nếu file chưa tồn tại hoặc lỗi."""
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        # File hỏng hoặc không đọc được -> coi như chưa có lịch sử, không làm sập app
        return []


def save_search(query: dict, weights: dict, top_car_names: list) -> None:
    """Thêm 1 lượt tìm kiếm mới vào cuối file JSON."""
    history = load_history()
    entry = {
        "thoi_gian": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "tieu_chi": query,
        "trong_so": {k: round(v, 3) for k, v in weights.items()},
        "ket_qua_hang_dau": top_car_names,
    }
    history.append(entry)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)


def clear_history() -> None:
    """Xóa toàn bộ lịch sử (xóa hẳn file JSON)."""
    if os.path.exists(HISTORY_FILE):
        os.remove(HISTORY_FILE)
[
  {
    "thoi_gian": "2026-09-25 16:02:06",
    "tieu_chi": {
      "Giá( VNĐ)": 450000000,
      "Công suất(HP)": 255,
      "Chỗ ngồi": 5,
      "Nhiên liệu": "Dầu",
      "Mục đích": "Dịch vụ",
      "Tình trạng": "Cũ"
    },
    "trong_so": {
      "Giá( VNĐ)": 0.255,
      "Nhiên liệu": 0.255,
      "Công suất(HP)": 0.064,
      "Chỗ ngồi": 0.178,
      "Mục đích": 0.032,
      "Tình trạng": 0.153,
      "Kiểu xe": 0.032,
      "Hộp số": 0.032
    },
    "ket_qua_hang_dau": [
      "Ford Ranger XLS 2.2L 4x2 MT 2018",
      "Mazda BT-50",
      "Ford Everest 2.5L 4x2 MT 2013",
      "VinFast VF5 Plus 2025",
      "Jazz 1.5 VX (2019)"
    ]
  }
]