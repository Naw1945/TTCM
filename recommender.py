"""
recommender.py
Xử lý dữ liệu và bọc KNN (content-based) để gợi ý xe.

THIẾT KẾ (quan trọng — đọc kỹ):
- 3 tiêu chí số (Giá, Công suất, Chỗ ngồi) đóng vai trò BỘ LỌC CỨNG:
    * Giá     : xe.price       <= ngân sách
    * Công suất: xe.power      >= mức tối thiểu
    * Chỗ ngồi : xe.seats      >= mức tối thiểu
  Các xe không đạt bị loại khỏi tập ứng viên TRƯỚC khi tính KNN.
- 5 tiêu chí phân loại (Nhiên liệu, Mục đích, Tình trạng, Kiểu xe, Hộp số)
  tham gia vào khoảng cách Euclid có trọng số (KNN).
- Trọng số người dùng chỉnh trên sidebar được chuẩn hóa tổng = 1.
  Riêng 3 tiêu chí số, trọng số không ảnh hưởng (vì đã là bộ lọc cứng);
  ta vẫn giữ UI để người dùng thấy "trọng số" nhưng thực tế bị gán = 0.
- Công thức khoảng cách: d_w(p,q) = sqrt( sum_i w_i * (p_i - q_i)^2 )
- Điểm phù hợp       : score(p,q) = 1 / (1 + d_w) * 100%

Tách riêng class CarRecommender để:
  - model/train_export.py có thể khởi tạo và export "bundle" (joblib).
  - api/main.py có thể nạp lại bundle đó (from_bundle) mà không cần CSV.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

NUMERIC_COLS = ["Giá( VNĐ)", "Công suất(HP)", "Chỗ ngồi"]
CATEGORICAL_COLS = ["Nhiên liệu", "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số"]
ALL_GROUPS = [
    "Giá( VNĐ)", "Nhiên liệu", "Công suất(HP)", "Chỗ ngồi",
    "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số",
]

# Trọng số mặc định: chia đều.
DEFAULT_WEIGHTS = {g: 1.0 / len(ALL_GROUPS) for g in ALL_GROUPS}


# ---------------------------------------------------------------------------
# Hàm tiện ích
# ---------------------------------------------------------------------------
def normalize_weights(raw_weights: dict) -> dict:
    """Chuẩn hóa trọng số về [0, ∞) và tổng = 1.

    Nếu tổng = 0 (người dùng kéo hết slider về 0), trả về bộ mặc định.
    """
    total = sum(max(v, 0.0) for v in raw_weights.values())
    if total <= 1e-9:
        return dict(DEFAULT_WEIGHTS)
    return {k: max(v, 0.0) / total for k, v in raw_weights.items()}


def load_and_clean_data(path: str) -> pd.DataFrame:
    """Đọc CSV gốc và làm sạch."""
    df = pd.read_csv(path)

    df = df.dropna(subset=["Tên xe"]).reset_index(drop=True)
    df["Tên xe"] = df["Tên xe"].str.strip().str.replace(r"\s{2,}", " ", regex=True)

    # Giá: "458.000.000" -> 458000000
    df["Giá( VNĐ)"] = (
        df["Giá( VNĐ)"].astype(str).str.replace(".", "", regex=False).astype(float)
    )

    # Công suất: có thể có dấu phẩy thập phân kiểu VN (vd "134,1")
    df["Công suất(HP)"] = (
        df["Công suất(HP)"].astype(str).str.replace(",", ".", regex=False).astype(float)
    )

    # Chuẩn hóa khoảng trắng ở các cột phân loại
    for col in CATEGORICAL_COLS:
        if col != "Nhiên liệu":
            df[col] = df[col].astype(str).str.strip()

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lớp CarRecommender
# ---------------------------------------------------------------------------
class CarRecommender:
    """Bọc pipeline: MinMaxScaler + One-hot thủ công + KNN có trọng số.

    Có 2 cách khởi tạo:
      - CarRecommender(df)                : dùng khi train (model/train_export.py)
      - CarRecommender.from_bundle(bundle): dùng khi inference (api/main.py)
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

        # (1) Chuẩn hóa Min-Max các cột số về [0, 1]
        self.scaler = MinMaxScaler()
        self.numeric_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.df[NUMERIC_COLS]),
            columns=NUMERIC_COLS,
        )

        # (2) One-hot cho các cột phân loại
        self.encoded_cat = pd.get_dummies(
            self.df[CATEGORICAL_COLS], prefix=CATEGORICAL_COLS,
        )
        self.cat_col_map = {
            col: [c for c in self.encoded_cat.columns if c.startswith(f"{col}_")]
            for col in CATEGORICAL_COLS
        }

        # (3) Ma trận đặc trưng đầy đủ (chưa nhân trọng số)
        self.feature_matrix = pd.concat(
            [self.numeric_scaled[NUMERIC_COLS], self.encoded_cat], axis=1,
        ).astype(float)

    # ------------------------------------------------------------------
    @classmethod
    def from_bundle(cls, bundle: dict) -> "CarRecommender":
        """Nạp lại recommender từ bundle đã export bằng joblib.

        bundle có dạng:
            {
              "scaler":          MinMaxScaler đã fit,
              "cat_col_map":     dict[str, list[str]],
              "feature_matrix":  pd.DataFrame,
              "df":              pd.DataFrame gốc (đã clean),
            }
        """
        obj = cls.__new__(cls)
        obj.df = bundle["df"]
        obj.scaler = bundle["scaler"]
        obj.cat_col_map = bundle["cat_col_map"]
        obj.feature_matrix = bundle["feature_matrix"]
        # numeric_scaled và encoded_cat không cần thiết khi chỉ inference
        return obj

    def to_bundle(self) -> dict:
        """Xuất bundle để joblib.dump()."""
        return {
            "scaler": self.scaler,
            "cat_col_map": self.cat_col_map,
            "feature_matrix": self.feature_matrix,
            "df": self.df,
        }

    # ------------------------------------------------------------------
    def unique_values(self, col: str):
        return sorted(self.df[col].dropna().astype(str).unique().tolist())

    # ------------------------------------------------------------------
    def _hard_filter_mask(self, query: dict) -> np.ndarray:
        """Lọc cứng theo 3 tiêu chí số (Giá, Công suất, Chỗ ngồi).

        Trả về numpy bool array, True = xe còn lại trong tập ứng viên.
        """
        mask = np.ones(len(self.df), dtype=bool)
        if "Giá( VNĐ)" in query:
            mask &= (self.df["Giá( VNĐ)"].values <= float(query["Giá( VNĐ)"]))
        if "Công suất(HP)" in query:
            mask &= (self.df["Công suất(HP)"].values >= float(query["Công suất(HP)"]))
        if "Chỗ ngồi" in query:
            mask &= (self.df["Chỗ ngồi"].values >= float(query["Chỗ ngồi"]))
        return mask

    def _per_dimension_weights(self, group_weights: dict, query: dict) -> np.ndarray:
        """Vector trọng số theo đúng thứ tự cột của feature_matrix.

        - Cột số: LUÔN có weight = 0 (đã là hard filter, không tham gia KNN).
        - Cột phân loại: nếu xuất hiện trong query mới có weight; chia đều
          cho số cột one-hot sinh ra từ tiêu chí đó (công thức 2.4):
              w_g,j = W_g / m_g
        """
        weights = []
        for _ in NUMERIC_COLS:
            weights.append(0.0)

        for col in CATEGORICAL_COLS:
            sub_cols = self.cat_col_map[col]
            m_g = max(len(sub_cols), 1)
            if col in query:
                w_gj = group_weights.get(col, 0.0) / m_g
            else:
                w_gj = 0.0
            weights.extend([w_gj] * len(sub_cols))
        return np.array(weights, dtype=float)

    def _encode_query(self, query: dict) -> np.ndarray:
        """Chuyển tiêu chí người dùng thành vector cùng không gian đặc trưng.

        - Cột số: đặt 0.5 (trung tính) — vì weight = 0 nên không ảnh hưởng.
        - Cột phân loại: 1.0 nếu khớp category người dùng chọn, ngược lại 0.0.
        """
        row = {col: 0.5 for col in NUMERIC_COLS}

        for col in CATEGORICAL_COLS:
            chosen = query.get(col)
            for sc in self.cat_col_map[col]:
                row[sc] = 1.0 if (chosen is not None and sc == f"{col}_{chosen}") else 0.0

        return pd.DataFrame([row])[self.feature_matrix.columns].values[0].astype(float)

    # ------------------------------------------------------------------
    def _criteria_breakdown(self, query: dict, row: pd.Series):
        """Trả về (chi_tiet, giai_thich) cho 1 xe cụ thể.

        - 3 tiêu chí số: hiển thị pass/fail (100% nếu qua filter, 0% nếu không
          — nhưng trên thực tế mọi xe đã qua filter đều = 100%).
        - 5 tiêu chí phân loại: 100% nếu trùng, 0% nếu khác.
        """
        chi_tiet: dict[str, float] = {}
        giai_thich: list[str] = []

        for col in NUMERIC_COLS:
            if col not in query:
                continue
            target = float(query[col])
            car_val = float(row[col])

            if col == "Giá( VNĐ)":
                passed = car_val <= target
                car_text = f"{car_val:,.0f}".replace(",", ".") + " VNĐ"
                target_text = f"{target:,.0f}".replace(",", ".") + " VNĐ"
                if passed:
                    giai_thich.append(f"✓ Giá: {car_text} (≤ {target_text}) — đạt yêu cầu.")
                else:
                    giai_thich.append(f"✗ Giá: {car_text} (> {target_text}) — vượt ngân sách.")
            elif col == "Công suất(HP)":
                passed = car_val >= target
                if passed:
                    giai_thich.append(f"✓ Công suất: {car_val:g} HP (≥ {target:g} HP) — đạt yêu cầu.")
                else:
                    giai_thich.append(f"✗ Công suất: {car_val:g} HP (< {target:g} HP) — chưa đạt mức tối thiểu.")
            else:  # Chỗ ngồi
                passed = car_val >= target
                if passed:
                    giai_thich.append(f"✓ Số chỗ: {int(car_val)} (≥ {int(target)}) — đạt yêu cầu.")
                else:
                    giai_thich.append(f"✗ Số chỗ: {int(car_val)} (< {int(target)}) — chưa đạt mức tối thiểu.")

            chi_tiet[col] = 100.0 if passed else 0.0

        for col in CATEGORICAL_COLS:
            if col not in query:
                continue
            target = query[col]
            car_val = row[col]
            pct = 100.0 if str(car_val) == str(target) else 0.0
            chi_tiet[col] = pct
            if pct == 100.0:
                giai_thich.append(f"✓ {col}: đúng yêu cầu ({target}).")
            else:
                giai_thich.append(f"✗ {col}: xe là '{car_val}', khác yêu cầu '{target}'.")

        return chi_tiet, giai_thich

    # ------------------------------------------------------------------
    def recommend(self, query: dict, weights: dict | None = None, k: int = 5) -> pd.DataFrame:
        """Quy trình:
            1) Lọc cứng (budget / power min / seats min).
            2) Vector trọng số (chỉ categorical có weight).
            3) Encode query.
            4) Khoảng cách Euclid có trọng số trên tập ứng viên.
            5) Sắp xếp, lấy top-k, gắn điểm phù hợp + giải thích.
        """
        # (1) Lọc cứng
        mask = self._hard_filter_mask(query)
        valid_df = self.df[mask].reset_index(drop=True)
        valid_features = self.feature_matrix[mask].reset_index(drop=True)

        if len(valid_df) == 0:
            return valid_df.assign(**{"Khoảng cách": [], "Độ phù hợp (%)": []})

        # (2) Trọng số
        group_weights = weights if weights is not None else DEFAULT_WEIGHTS
        w = self._per_dimension_weights(group_weights, query)

        # Nếu không có trọng số nào > 0 (user bỏ hết categorical), thì tất cả
        # xe đều có distance = 0 → top-k là k xe đầu tiên (theo thứ tự CSV).
        # Đây là edge case, chấp nhận được.

        # (3) Encode query
        q_vector = self._encode_query(query)

        # (4) Khoảng cách
        diff_sq = (valid_features.values - q_vector) ** 2
        d_w = np.sqrt((diff_sq * w).sum(axis=1))

        # (5) Top-k
        k = min(k, len(valid_df))
        order = np.argsort(d_w)[:k]

        result = valid_df.iloc[order].copy()
        result["Khoảng cách"] = d_w[order]
        result["Độ phù hợp (%)"] = (1.0 / (1.0 + result["Khoảng cách"]) * 100).round(1)
        result = result.reset_index(drop=True)

        # (6) Chi tiết + giải thích
        chi_tiet_list, giai_thich_list = [], []
        for _, row in result.iterrows():
            chi_tiet, giai_thich = self._criteria_breakdown(query, row)
            chi_tiet_list.append(chi_tiet)
            giai_thich_list.append(giai_thich)
        result["Chi tiết tiêu chí"] = chi_tiet_list
        result["Giải thích"] = giai_thich_list

        return result