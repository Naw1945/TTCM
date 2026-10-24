import os
import json
from google import genai

MODEL_NAME = "gemini-2.5-flash"

def explain_recommendations(query: dict, weights: dict, recommendations: list[dict]) -> str | None:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not recommendations:
        return None

    prompt = f"""
Bạn là trợ lý giải thích kết quả của hệ thống hỗ trợ lựa chọn ô tô.
Hãy viết 3–4 câu tiếng Việt giải thích kết quả gợi ý.
Chỉ sử dụng dữ liệu được cung cấp bên dưới. Không tự thêm thông số.
Tiêu chí người dùng: {json.dumps(query, ensure_ascii=False, default=str)}
Trọng số: {json.dumps(weights, ensure_ascii=False, default=str)}
Kết quả KNN: {json.dumps(recommendations, ensure_ascii=False, default=str)}
"""
    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return (response.text or "").strip() or None
    except Exception as e:
        print(f"🔴 LỖI GỌI GEMINI (Recommend): {e}") 
        return None