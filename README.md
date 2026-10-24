# Hệ thống Hỗ trợ Lựa chọn Ô tô (Car Recommender System)

Đồ án môn Thực tập chuyên môn. Hệ thống gợi ý xe ô tô dựa trên nhu cầu người dùng sử dụng thuật toán KNN (K-Nearest Neighbors) kết hợp Trí tuệ nhân tạo (Gemini API).

## 🚀 Công nghệ sử dụng
- **Backend:** Python, FastAPI, Scikit-learn, Pandas
- **Frontend:** Streamlit
- **AI:** Google Gemini API
- **Data:** 365 mẫu xe (car0.csv)

## 📁 Cấu trúc thư mục
- `data/`: Chứa file dữ liệu `car0.csv`
- `model/`: Chứa script huấn luyện/export và file model `joblib`
- `api/`: Chứa Backend FastAPI, logic gọi Gemini, và file test
- `web/`: Giao diện Streamlit đa trang (Home, Tìm kiếm, So sánh, Lịch sử)

## ⚙️ Hướng dẫn cài đặt & Chạy

**Bước 1: Cài đặt thư viện**
```bash
pip install -r requirements.txt