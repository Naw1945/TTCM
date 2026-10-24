import sys
from pathlib import Path
import streamlit as st
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from history import load_history, clear_history

st.set_page_config(page_title="Lịch sử", page_icon="🕒")

st.title("🕒 Lịch sử tìm kiếm")

history = load_history()

if not history:
    st.info("Chưa có lượt tìm kiếm nào được lưu.")
else:
    st.write(f"Tổng cộng: **{len(history)}** lượt tìm kiếm.")
    
    if st.button("🗑️ Xóa toàn bộ lịch sử", type="secondary"):
        clear_history()
        st.success("Đã xóa lịch sử!")
        st.rerun()

    st.divider()
    for idx, entry in enumerate(reversed(history[-10:]), 1):
        with st.expander(f"Lượt {len(history) - idx + 1} - {entry.get('thoi_gian')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Tiêu chí đã nhập:**")
                for k, v in entry.get("tieu_chi", {}).items():
                    val_str = f"{v:,.0f} VNĐ".replace(",", ".") if k == "Giá( VNĐ)" else str(v)
                    st.write(f"- {k}: {val_str}")
            with col2:
                st.markdown("**Top xe gợi ý:**")
                for i, car in enumerate(entry.get("ket_qua_hang_dau", []), 1):
                    st.write(f"{i}. {car}")