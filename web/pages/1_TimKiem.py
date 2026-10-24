import sys
from pathlib import Path
import requests
import streamlit as st
import pandas as pd
import altair as alt

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from history import save_search

st.set_page_config(page_title="Tìm kiếm & So sánh", page_icon="🔍", layout="wide")

API_URL = "http://localhost:8000"

def format_vnd(value: float) -> str:
    return f"{value:,.0f}".replace(",", ".") + " VNĐ"

st.title("🔍 Tìm kiếm & So sánh Xe")

# --- SIDEBAR: Trọng số ---
with st.sidebar:
    st.header("⚖️ Trọng số tiêu chí")
    st.caption("Hệ thống tự động chuẩn hóa tổng trọng số = 1.0")
    
    WEIGHT_LABELS = {
        "Nhiên liệu": "Nhiên liệu", "Mục đích": "Mục đích", 
        "Tình trạng": "Tình trạng", "Kiểu xe": "Kiểu xe", "Hộp số": "Hộp số",
    }
    raw_weights = {}
    for group in WEIGHT_LABELS:
        raw_weights[group] = st.slider(WEIGHT_LABELS[group], 0.0, 5.0, 1.0, 0.5, key=f"w_{group}")

# --- MAIN: Form nhập liệu ---
with st.form("search_form"):
    st.subheader("1. Nhập tiêu chí tìm kiếm")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("**Bộ lọc cứng**")
        use_price = st.checkbox("Giới hạn ngân sách tối đa", value=True)
        price_mil = st.number_input("Ngân sách (Triệu VNĐ)", min_value=100, max_value=20000, value=1500, step=50)
        
        use_power = st.checkbox("Công suất tối thiểu")
        min_power = st.number_input("Công suất (HP)", min_value=0, max_value=1000, value=0, step=10)
        
        min_seats = st.selectbox("Số chỗ ngồi tối thiểu", [0, 4, 5, 7, 8, 16], format_func=lambda x: "Không yêu cầu" if x==0 else f"≥ {x} chỗ")

    with col2:
        st.markdown("**Tiêu chí ưu tiên (KNN)**")
        fuel = st.selectbox("Nhiên liệu", ["Không yêu cầu", "Xăng", "Dầu", "Hybrid", "Điện"])
        purpose = st.selectbox("Mục đích", ["Không yêu cầu", "Cá nhân & Gia đình", "Dịch vụ"])
        condition = st.selectbox("Tình trạng", ["Không yêu cầu", "Mới", "Cũ"])

    with col3:
        st.markdown("**Tiêu chí ưu tiên (Tiếp)**")
        car_type = st.selectbox("Kiểu xe", ["Không yêu cầu", "Sedan", "SUV/CUV", "Hatchback", "MPV", "Bán tải", "Coupe", "Thương mại/Van"])
        transmission = st.selectbox("Hộp số", ["Không yêu cầu", "AT", "MT"])
        k = st.number_input("Số lượng xe gợi ý", min_value=1, max_value=10, value=5)

    submitted = st.form_submit_button("🚀 Tìm xe ngay", width="stretch", type="primary")

# --- Xử lý khi submit ---
if submitted:
    query = {}
    if use_price: query["Giá( VNĐ)"] = price_mil * 1_000_000
    if use_power and min_power > 0: query["Công suất(HP)"] = min_power
    if min_seats > 0: query["Chỗ ngồi"] = min_seats
    if fuel != "Không yêu cầu": query["Nhiên liệu"] = fuel
    if purpose != "Không yêu cầu": query["Mục đích"] = purpose
    if condition != "Không yêu cầu": query["Tình trạng"] = condition
    if car_type != "Không yêu cầu": query["Kiểu xe"] = car_type
    if transmission != "Không yêu cầu": query["Hộp số"] = transmission

    payload = {"criteria": query, "weights": raw_weights, "k": k}

    with st.spinner("Đang tính toán KNN và phân tích..."):
        try:
            res = requests.post(f"{API_URL}/recommend", json=payload, timeout=60)
            if res.status_code == 200:
                data = res.json()
                st.session_state["search_results"] = data["recommendations"]
                st.session_state["gemini_text"] = data.get("explanation")
                top_names = [c["Tên xe"] for c in data["recommendations"]]
                save_search(query, raw_weights, top_names)
            else:
                st.error(f"Lỗi API: {res.json().get('detail')}")
        except Exception as e:
            st.error(f"Lỗi kết nối: {e}")

# --- Hiển thị kết quả ---
if "search_results" in st.session_state and st.session_state["search_results"]:
    results = st.session_state["search_results"]
    st.divider()
    st.subheader("2. Kết quả gợi ý")
    
    if st.session_state.get("gemini_text"):
        st.info(f"🤖 **Trợ lý Gemini phân tích:**\n\n{st.session_state['gemini_text']}")
    
    df_res = pd.DataFrame(results)
    df_display = df_res[["Tên xe", "Giá( VNĐ)", "Nhiên liệu", "Công suất(HP)", "Chỗ ngồi", "Độ phù hợp (%)"]].copy()
    df_display["Giá( VNĐ)"] = df_display["Giá( VNĐ)"].apply(lambda x: format_vnd(x))
    st.dataframe(df_display, width="stretch", hide_index=True)

    # --- BIỂU ĐỒ TRỰC QUAN (TABS) ---
    st.markdown("**Biểu đồ trực quan:**")
    tab1, tab2, tab3 = st.tabs(["🎯 Điểm phù hợp (%)", "💰 Giá bán (VNĐ)", "⚡ Công suất (HP)"])
    
    def draw_horiz_bar(df, col_name, color_scheme):
        chart = alt.Chart(df).mark_bar().encode(
            y=alt.Y('Tên xe:N', sort='-x', title="", axis=alt.Axis(labelLimit=250)),
            x=alt.X(f'{col_name}:Q', title=col_name),
            color=alt.Color(f'{col_name}:Q', scale=alt.Scale(scheme=color_scheme)),
            tooltip=['Tên xe', col_name]
        ).properties(height=250)
        st.altair_chart(chart, width="stretch")

    with tab1:
        draw_horiz_bar(df_res, 'Độ phù hợp (%)', 'blues')
    with tab2:
        draw_horiz_bar(df_res, 'Giá( VNĐ)', 'oranges')
    with tab3:
        draw_horiz_bar(df_res, 'Công suất(HP)', 'greens')

    st.subheader("3. Chi tiết đánh giá")
    for idx, car in enumerate(results, 1):
        with st.expander(f"{idx}. {car['Tên xe']} - Độ phù hợp: {car['Độ phù hợp (%)']}%"):
            st.write(f"**Giá:** {format_vnd(car['Giá( VNĐ)'])} | **Nhiên liệu:** {car['Nhiên liệu']} | **Công suất:** {car['Công suất(HP)']} HP")
            for reason in car["Giải thích"]:
                if "✓" in reason:
                    st.markdown(f"<span style='color:green'>{reason}</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:red'>{reason}</span>", unsafe_allow_html=True)

    # --- PHẦN SO SÁNH TÍCH HỢP ---
    st.divider()
    st.subheader("4. So sánh các xe")
    st.caption("Chỉ so sánh các xe nằm trong danh sách gợi ý ở trên.")
    
    result_car_names = df_res["Tên xe"].tolist()
    default_cars = result_car_names[:2] if len(result_car_names) >= 2 else result_car_names
    
    selected_compare = st.multiselect(
        "Chọn 2 đến 3 xe để so sánh chi tiết:",
        options=result_car_names,
        default=default_cars
    )

    if len(selected_compare) < 2:
        st.warning("Vui lòng chọn ít nhất 2 xe để so sánh.")
    elif len(selected_compare) > 3:
        st.warning("Chỉ so sánh tối đa 3 xe.")
    else:
        if st.button("Tiến hành so sánh", type="secondary"):
            try:
                res_cmp = requests.post(f"{API_URL}/compare", json={"car_names": selected_compare}, timeout=10)
                if res_cmp.status_code == 200:
                    cmp_data = res_cmp.json()["cars"]
                    df_cmp = pd.DataFrame(cmp_data)
                    
                    st.markdown("**Bảng thông số chi tiết:**")
                    st.dataframe(df_cmp.set_index("Tên xe").T.astype(str), width="stretch")

                    # --- TỰ ĐỘNG SINH NHẬN XÉT BẰNG CODE (RULE-BASED) ---
                    st.markdown("### Nhận xét cho người dùng")
                    
                    df_sorted = df_cmp.sort_values(by="Giá( VNĐ)")
                    cheapest = df_sorted.iloc[0]
                    most_expensive = df_sorted.iloc[-1]
                    
                    seat_text = f"về số chỗ ngồi, xe {most_expensive['Tên xe']} rộng rãi hơn với {int(most_expensive['Chỗ ngồi'])} chỗ." if most_expensive['Chỗ ngồi'] > cheapest['Chỗ ngồi'] else ""
                    
                    hp_text = ""
                    if most_expensive['Công suất(HP)'] > cheapest['Công suất(HP)']:
                        hp_text = f"Tuy nhiên, {most_expensive['Tên xe']} có công suất mạnh mẽ hơn ({most_expensive['Công suất(HP)']} HP so với {cheapest['Công suất(HP)']} HP)."
                    elif cheapest['Công suất(HP)'] > most_expensive['Công suất(HP)']:
                        hp_text = f"Bất ngờ là {cheapest['Tên xe']} lại có công suất nhỉnh hơn ({cheapest['Công suất(HP)']} HP so với {most_expensive['Công suất(HP)']} HP)."

                    nhan_xet = (
                        f"Dựa trên phân tích dữ liệu: **{cheapest['Tên xe']}** là lựa chọn tiết kiệm hơn với mức giá "
                        f"{cheapest['Giá( VNĐ)']:,.0f} VNĐ (thấp hơn {most_expensive['Giá( VNĐ)'] - cheapest['Giá( VNĐ)']:,.0f} VNĐ so với {most_expensive['Tên xe']}). "
                        f"{seat_text} {hp_text} "
                        f"Nếu ưu tiên ngân sách và di chuyển cơ bản, hãy chọn {cheapest['Tên xe']}. "
                        f"Nếu cần không gian rộng hơn và trải nghiệm mạnh mẽ, {most_expensive['Tên xe']} là lựa chọn hợp lý."
                    ).replace(",", ".")

                    st.success(nhan_xet)

                    # BIỂU ĐỒ
                    st.markdown("**Biểu đồ so sánh Giá (Minh họa):**")
                    chart_cmp = alt.Chart(df_cmp).mark_bar().encode(
                        y=alt.Y('Tên xe:N', title="", axis=alt.Axis(labelLimit=250)),
                        x=alt.X('Giá( VNĐ):Q', title="Giá (VNĐ)"),
                        color='Tên xe:N',
                        tooltip=['Tên xe', 'Giá( VNĐ)']
                    ).properties(height=250)
                    st.altair_chart(chart_cmp, width="stretch")
                else:
                    st.error("Lỗi khi lấy dữ liệu so sánh.")
            except Exception as e:
                st.error(f"Lỗi kết nối API: {e}")