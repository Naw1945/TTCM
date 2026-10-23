import requests
import streamlit as st
import pandas as pd

from recommender import load_and_clean_data, DEFAULT_WEIGHTS, normalize_weights
from history import load_history, save_search, clear_history

API_URL = "http://127.0.0.1:8000"
DATA_PATH = "car0.csv"

st.set_page_config(
    page_title="Hệ thống gợi ý ô tô theo nhu cầu (KNN)",
    page_icon="🚗",
    layout="wide",
)


@st.cache_data
def get_data():
    return load_and_clean_data(DATA_PATH)


def format_vnd(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " VNĐ"


def show_score_bar(label: str, score: float):
    """Hiển thị thanh tiến trình đáp ứng tiêu chí"""
    if score >= 80:
        color = "#4CAF50"  # xanh lá
        text_style = f"color: {color}; font-weight: bold;"
    elif score >= 50:
        color = "#FFC400"  # vàng
        text_style = f"color: {color}; font-weight: normal;"
    else:
        color = "#F44336"  # đỏ
        text_style = f"color: {color}; font-weight: normal;"

    html_str = f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
            <span style="{text_style}">{label}</span>
            <span style="{text_style}">{score:.1f}%</span>
        </div>
        <div style="width: 100%; background-color: #e0e0e0; border-radius: 4px; height: 8px;">
            <div style="width: {score}%; background-color: {color}; height: 100%; border-radius: 4px;"></div>
        </div>
    </div>
    """
    st.markdown(html_str, unsafe_allow_html=True)


WEIGHT_LABELS = {
    "Giá( VNĐ)": "Giá",
    "Nhiên liệu": "Nhiên liệu",
    "Công suất(HP)": "Công suất",
    "Chỗ ngồi": "Chỗ ngồi",
    "Mục đích": "Mục đích",
    "Tình trạng": "Tình trạng",
    "Kiểu xe": "Kiểu xe",
    "Hộp số": "Hộp số",
}

try:
    df = get_data()
except Exception as exc:
    st.error(f"Không thể đọc dữ liệu xe: {exc}")
    st.stop()


def get_options(column):
    if column not in df.columns:
        return []
    return sorted(
        df[column].dropna().astype(str).str.strip().unique().tolist()
    )


def get_model_info():
    try:
        response = requests.get(f"{API_URL}/model-info", timeout=3)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None


model_info = get_model_info()

st.title("🚗 Hệ thống hỗ trợ lựa chọn xe ô tô theo nhu cầu sử dụng")
st.caption("Nhóm 8 - Đồ án môn Hệ trợ giúp quyết định — Gợi ý xe bằng thuật toán KNN và tích hợp Gemini API")

if model_info is None:
    st.warning(
        "⚠️ Chưa kết nối được FastAPI Backend. Hãy chạy server (`uvicorn main:app --reload`) trước khi tìm kiếm xe.")
else:
    st.success(
        f"Đã kết nối API | Mô hình: {model_info.get('model', 'KNN')} | Số dòng dữ liệu: {model_info.get('dataset_rows', 'N/A')}")

# --- Sidebar: Trọng số ---
with st.sidebar:
    st.header("⚖️ Trọng số thuộc tính")
    st.caption("Tổng trọng số sẽ tự động chuẩn hóa bằng 1.00.")

    raw_weights = {}
    for group in DEFAULT_WEIGHTS:
        raw_weights[group] = st.slider(
            WEIGHT_LABELS.get(group, group), 0.0, 5.0, 1.0, 0.5, key=f"weight_{group}"
        )

    active_weights = normalize_weights(raw_weights)

    st.divider()
    st.caption("Trọng số thực tế sau chuẩn hóa:")
    weight_df = pd.DataFrame(
        {
            "Thuộc tính": [WEIGHT_LABELS.get(g, g) for g in active_weights],
            "Trọng số": [f"{v:.3f}" for v in active_weights.values()],
        }
    )
    st.dataframe(weight_df, hide_index=True, use_container_width=True)

# --- Form Nhập Tiêu Chí ---
st.subheader("1. Nhập tiêu chí mong muốn")

with st.form("car_search_form"):
    max_price = float(df["Giá( VNĐ)"].max())
    min_price = float(df["Giá( VNĐ)"].min())

    use_price = st.checkbox("Có giới hạn ngân sách tối đa", value=True)
    price_trieu = st.slider(
        "Ngân sách tối đa (triệu VNĐ)",
        min_value=int(min_price / 1_000_000),
        max_value=int(max_price / 1_000_000),
        value=int(max_price / 1_000_000),
        step=10,
        disabled=not use_price,
    )
    price = price_trieu * 1_000_000
    if use_price:
        st.caption(f"≈ {format_vnd(price)} (Lọc cứng: Loại bỏ các xe vượt ngân sách này)")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fuel = st.selectbox("Nhiên liệu", ["Không yêu cầu"] + get_options("Nhiên liệu"))
        purpose = st.selectbox("Mục đích sử dụng", ["Không yêu cầu"] + get_options("Mục đích"))

    with col2:
        car_type = st.selectbox("Kiểu xe", ["Không yêu cầu"] + get_options("Kiểu xe"))
        transmission = st.selectbox("Hộp số", ["Không yêu cầu"] + get_options("Hộp số"))

    with col3:
        condition = st.selectbox("Tình trạng", ["Không yêu cầu"] + get_options("Tình trạng"))

        power_options = sorted(pd.to_numeric(df["Công suất(HP)"], errors="coerce").dropna().unique().tolist())
        min_power = float(min(power_options)) if power_options else 0.0
        max_power = float(max(power_options)) if power_options else 100.0

        use_power = st.checkbox("Lọc theo công suất tối thiểu", value=False)
        min_required_power = st.slider(
            "Công suất tối thiểu (HP)",
            min_value=int(min_power),
            max_value=int(max_power),
            value=int(min_power),
            step=5,
            disabled=not use_power
        )

    with col4:
        seat_options = sorted(pd.to_numeric(df["Chỗ ngồi"], errors="coerce").dropna().unique().astype(int).tolist())
        min_seats = st.selectbox(
            "Số chỗ ngồi tối thiểu",
            options=[0] + seat_options,
            format_func=lambda value: "Không yêu cầu" if value == 0 else str(value),
        )
        k = st.number_input("Số lượng xe gợi ý (K)", min_value=1, max_value=20, value=5)

    submitted = st.form_submit_button("Tìm xe phù hợp", type="primary", use_container_width=True)

# --- Xử lý sự kiện Gửi Form và Gọi FastAPI ---
if "last_result" not in st.session_state:
    st.session_state["last_result"] = None

if submitted:
    query = {}
    if use_price:
        query["Giá( VNĐ)"] = float(price)
    if use_power and min_required_power > 0:
        query["Công suất(HP)"] = float(min_required_power)
    if min_seats > 0:
        query["Chỗ ngồi"] = int(min_seats)

    optional_criteria = {
        "Nhiên liệu": fuel,
        "Mục đích": purpose,
        "Tình trạng": condition,
        "Kiểu xe": car_type,
        "Hộp số": transmission,
    }

    for column, value in optional_criteria.items():
        if value != "Không yêu cầu":
            query[column] = value

    payload = {
        "criteria": query,
        "weights": active_weights,
        "k": int(k),
    }

    try:
        with st.spinner("Đang kết nối API để tính toán KNN và phân tích Gemini..."):
            response = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)

        if response.status_code == 422:
            detail = response.json().get("detail", "Dữ liệu đầu vào không hợp lệ.")
            st.error(f"Lỗi dữ liệu đầu vào (422): {detail}")
        else:
            response.raise_for_status()
            data = response.json()

            recommendations = data.get("recommendations", [])
            st.session_state["last_result"] = recommendations
            st.session_state["last_explanation"] = data.get("explanation")

            if recommendations:
                top_car_names = [car.get("Tên xe", "Không rõ tên") for car in recommendations]
                save_search(query, active_weights, top_car_names)
                st.toast("Đã tìm kiếm thành công và lưu vào lịch sử!", icon="💾")
            else:
                st.info("Không tìm thấy xe nào thỏa mãn điều kiện lọc cứng ngân sách và tiêu chí.")

    except requests.ConnectionError:
        st.error("Không thể kết nối tới FastAPI Server tại `http://127.0.0.1:8000`. Hãy đảm bảo server đang chạy.")
    except requests.Timeout:
        st.error("Yêu cầu xử lý quá thời gian chờ (Timeout). Hãy thử lại.")
    except requests.RequestException as exc:
        st.error(f"Lỗi khi gọi API: {exc}")

# --- HIỂN THỊ KẾT QUẢ GỢI Ý ---
recommendations = st.session_state.get("last_result", [])

if recommendations is not None:
    st.divider()
    st.subheader("2. Kết quả gợi ý")

    # Hiển thị giải thích tổng quan từ Gemini API nếu có
    gemini_explanation = st.session_state.get("last_explanation")
    if gemini_explanation:
        with st.expander("🤖 Đánh giá và Phân tích tổng quan từ Gemini AI", expanded=True):
            st.write(gemini_explanation)

    if len(recommendations) == 0:
        st.warning("Chưa có danh sách xe nào để hiển thị.")
    else:
        result_df = pd.DataFrame(recommendations)

        display_columns = [
            col for col in [
                "Tên xe", "Giá( VNĐ)", "Nhiên liệu", "Công suất(HP)",
                "Chỗ ngồi", "Mục đích", "Tình trạng", "Kiểu xe", "Hộp số", "Độ phù hợp (%)"
            ] if col in result_df.columns
        ]

        # Định dạng lại giá tiền hiển thị ở bảng tổng quan
        temp_df = result_df.copy()
        if "Giá( VNĐ)" in temp_df.columns:
            temp_df["Giá( VNĐ)"] = temp_df["Giá( VNĐ)"].apply(lambda x: format_vnd(float(x)) if pd.notnull(x) else x)

        st.dataframe(temp_df[display_columns], use_container_width=True, hide_index=True)

        # --- Chi tiết từng xe & Thanh tiến trình tiêu chí ---
        st.subheader("3. Chi tiết mức độ đáp ứng từng xe")
        for index, car in enumerate(recommendations, start=1):
            car_name = car.get("Tên xe", f"Xe {index}")
            score = car.get("Độ phù hợp (%)", 0.0)
            distance = car.get("Khoảng cách", 0.0)

            with st.expander(f"{index}. {car_name} — Độ phù hợp: {float(score):.1f}%"):
                col1, col2 = st.columns([3, 4])

                with col1:
                    st.write(f"**Giá:** {format_vnd(float(car.get('Giá( VNĐ)', 0)))}")
                    st.write(f"**Nhiên liệu:** {car.get('Nhiên liệu', 'N/A')}")
                    st.write(f"**Công suất:** {car.get('Công suất(HP)', 'N/A')} HP")
                    st.write(f"**Số chỗ:** {car.get('Chỗ ngồi', 'N/A')} chỗ")
                    st.write(f"**Mục đích:** {car.get('Mục đích', 'N/A')}")

                with col2:
                    st.metric("Độ phù hợp", f"{float(score):.1f}%")
                    st.caption(f"Khoảng cách hình học KNN: {float(distance):.4f}")
                    st.write(f"**Kiểu xe:** {car.get('Kiểu xe', 'N/A')}")
                    st.write(f"**Tình trạng:** {car.get('Tình trạng', 'N/A')}")
                    st.write(f"**Hộp số:** {car.get('Hộp số', 'N/A')}")

                breakdown = car.get("Chi tiết tiêu chí")
                if breakdown and isinstance(breakdown, dict):
                    st.markdown("**Mức độ đáp ứng theo từng tiêu chí:**")
                    for col_name, sub_score in breakdown.items():
                        show_score_bar(WEIGHT_LABELS.get(col_name, col_name), float(sub_score))

                reasons = car.get("Giải thích")
                if reasons and isinstance(reasons, list):
                    st.markdown("**Lý do chi tiết từ hệ thống:**")
                    for r in reasons:
                        if "✓" in r:
                            st.markdown(f"<span style='color: #2e7d32; font-weight: bold;'>{r}</span>",
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #d32f2f;'>{r}</span>", unsafe_allow_html=True)

# --- 4. Lịch sử tìm kiếm ---
st.divider()
st.subheader("4. Lịch sử tìm kiếm phiên làm việc")

history = load_history()
if not history:
    st.caption("Chưa có lượt tìm kiếm nào được lưu trong `history.json`.")
else:
    st.caption(f"Đã ghi nhận {len(history)} lượt tìm kiếm gần đây.")

    for idx, entry in enumerate(reversed(history[-10:]), start=1):
        with st.expander(f"🕒 Lượt tìm kiếm lúc: {entry.get('thoi_gian', 'N/A')}"):
            col_h1, col_h2 = st.columns(2)
            with col_h1:
                st.markdown("**Tiêu chí đã thiết lập:**")
                tieu_chi_dict = entry.get("tieu_chi", {})
                if not tieu_chi_dict:
                    st.write("- Không có tiêu chí chi tiết")
                else:
                    for k, v in tieu_chi_dict.items():
                        label_name = WEIGHT_LABELS.get(k, k)
                        val_str = format_vnd(v) if k == "Giá( VNĐ)" else str(v)
                        st.write(f"- **{label_name}:** {val_str}")

            with col_h2:
                st.markdown("**Top xe đề xuất trong phiên:**")
                top_cars = entry.get("ket_qua_hang_dau", [])
                for car_num, car_name in enumerate(top_cars, 1):
                    st.write(f"{car_num}. {car_name}")

    if st.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
        clear_history()
        st.session_state["last_result"] = None
        st.success("Đã xóa sạch lịch sử tìm kiếm.")
        st.rerun()