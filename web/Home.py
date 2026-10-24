import sys
from pathlib import Path
import requests
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

st.set_page_config(page_title="Hệ thống gợi ý xe", page_icon="🚗", layout="wide")

API_URL = "http://localhost:8000"

st.title("🚗 Hệ thống Hỗ trợ Lựa chọn Ô tô theo Nhu cầu")
st.caption("Đồ án môn Thực tập chuyên môn — KNN & Gemini API")

st.markdown("""
### Chào mừng bạn!
Hệ thống giúp bạn tìm kiếm chiếc xe phù hợp nhất dựa trên nhu cầu thực tế và so sánh trực quan các xe với nhau.

👉 **Hãy bắt đầu bằng cách chọn mục "Tìm kiếm & So sánh" ở thanh menu bên trái!**
""")

st.divider()

# Hiển thị trạng thái API
st.subheader("Trạng thái hệ thống")
try:
    res = requests.get(f"{API_URL}/model-info", timeout=3)
    if res.status_code == 200:
        meta = res.json()
        st.success("🟢 Hệ thống Backend đang hoạt động bình thường.")
        col1, col2, col3 = st.columns(3)
        col1.metric("Phiên bản Model", meta.get("version", "1.0.0"))
        col2.metric("Thuật toán", meta.get("algorithm", "KNN"))
        col3.metric("Dữ liệu", f"{meta.get('n_samples', 0)} mẫu xe")
except requests.ConnectionError:
    st.error("🔴 Không thể kết nối tới Backend API. Hãy đảm bảo server đang chạy.")