
import os
import json

from google import genai


MODEL_NAME = "gemini-2.5-flash"


def explain_recommendations(
    query: dict,
    weights: dict,
    recommendations: list[dict],
) -> str | None:
    """Sinh giải thích tiếng Việt dựa trên kết quả KNN."""

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key or not recommendations:
        return None

    prompt = f"""
Bạn là trợ lý giải thích kết quả của hệ thống hỗ trợ lựa chọn ô tô.

Hãy viết 3–4 câu tiếng Việt giải thích kết quả gợi ý.
Chỉ sử dụng dữ liệu được cung cấp bên dưới.
Ưu tiên tiêu chí có trọng số cao và mức độ đáp ứng từng tiêu chí.
Nếu xe không đáp ứng một tiêu chí, hãy nói rõ.
Không tự thêm thông số, không tự xếp hạng lại xe.
Điểm phù hợp là kết quả tính toán của KNN, không phải đánh giá chất lượng xe.

Tiêu chí người dùng:
{json.dumps(query, ensure_ascii=False, default=str)}

Trọng số:
{json.dumps(weights, ensure_ascii=False, default=str)}

Kết quả KNN:
{json.dumps(recommendations, ensure_ascii=False, default=str)}
"""

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
        )

        return (response.text or "").strip() or None

    except Exception:
        # Gemini lỗi thì không làm gián đoạn chức năng KNN.
        return None
