# 🚗 Hệ thống Hỗ trợ Lựa chọn Ô tô theo Nhu cầu (Car Recommender System)

Đồ án môn Thực tập chuyên môn. Hệ thống giúp người dùng tìm kiếm và so sánh các mẫu xe ô tô phù hợp nhất dựa trên nhu cầu thực tế. Dự án sử dụng thuật toán **KNN (K-Nearest Neighbors)** kết hợp với Trí tuệ nhân tạo **(Google Gemini API)** để đưa ra lời khuyên thông minh.

## 🚀 Công nghệ sử dụng
- **Backend:** Python 3.10, FastAPI, Scikit-learn, Pandas, Numpy.
- **Frontend:** Streamlit, Altair (vẽ biểu đồ).
- **Trí tuệ nhân tạo (LLM):** Google Gemini API (gemini-2.5-flash).
- **Dữ liệu:** 365 mẫu xe (lưu trữ tại `data/car0.csv`).

## 📁 Cấu trúc thư mục
```text
TTCM/
├── data/                  # Chứa file dữ liệu gốc (car0.csv)
├── model/                 # Chứa script huấn luyện/export và file model
│   ├── train_export.py    # Script tạo file joblib và metadata
│   ├── car_recommender.joblib
│   └── model_metadata.json
├── api/                   # Backend FastAPI
│   ├── main.py            # Chứa các endpoint (/recommend, /compare, /model-info)
│   ├── llm_explain.py     # Logic gọi Gemini API
│   └── test_api.py        # Script kiểm thử tự động
├── web/                   # Giao diện Streamlit
│   ├── Home.py            # Trang chủ
│   └── pages/
│       ├── 1_TimKiem.py   # Trang Tìm kiếm & So sánh
│       └── 2_LichSu.py    # Trang Lịch sử
├── recommender.py         # Lớp xử lý thuật toán KNN (dùng chung)
├── history.py             # Logic lưu trữ lịch sử (JSON)
├── requirements.txt
├── README.md
└── ARCHITECTURE.md
⚙️ Hướng dẫn cài đặt & Chạy (Local)
Bước 1: Cài đặt thư viện
Yêu cầu máy tính đã cài đặt Python 3.10+.

bash
pip install -r requirements.txt
Bước 2: Export Model (Chỉ chạy 1 lần đầu)
Lệnh này sẽ đọc file CSV, huấn luyện các bộ mã hóa và lưu thành file .joblib.

bash
python model/train_export.py
Bước 3: Khởi chạy Backend (Cửa sổ CMD 1)
Lưu ý: Tính năng gợi ý của AI (Gemini) là tùy chọn. Nếu không có API Key, hệ thống vẫn chạy bình thường nhưng sẽ không có phần nhận xét tự động.

Trên Windows (Command Prompt):

cmd
set GEMINI_API_KEY=your_api_key_here
uvicorn api.main:app --reload --port 8000
Trên macOS / Linux:

bash
export GEMINI_API_KEY="your_api_key_here"
uvicorn api.main:app --reload --port 8000
Bước 4: Khởi chạy Giao diện (Cửa sổ CMD 2)
Mở một cửa sổ Terminal mới tại thư mục gốc và chạy:

bash
streamlit run web/Home.py
Trình duyệt sẽ tự động mở địa chỉ http://localhost:8501.

🧪 Kiểm thử (Testing)
Dự án có sẵn bộ test API tự động để đảm bảo tính ổn định của hệ thống (bao gồm test các trường hợp lỗi, bộ lọc cứng, và dữ liệu rác).
Chạy lệnh sau tại thư mục gốc:

bash
pytest api/test_api.py -v
