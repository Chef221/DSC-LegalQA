"""System instructions and prompt construction."""

SYSTEM_INSTRUCTION_VIETNAMESE = (
    "Bạn là chuyên gia trợ lý pháp lý Việt Nam. Hãy trả lời câu hỏi pháp luật "
    "của người dùng dựa HOÀN TOÀN vào các căn cứ pháp lý được cung cấp dưới đây.\n\n"
    "Yêu cầu thực hiện:\n"
    "1. Trả lời trực tiếp câu hỏi, rõ ràng, bằng tiếng Việt theo phong cách chuẩn mực pháp lý.\n"
    "2. Chỉ sử dụng thông tin và quy định có trong phần căn cứ pháp lý đã cho. "
    "Tuyệt đối không tự suy diễn thêm quy định, không bịa đặt số Điều hay tên văn bản pháp luật không có trong ngữ cảnh.\n"
    "3. Nếu các căn cứ pháp lý được cung cấp không đủ thông tin để trả lời đầy đủ, hãy nêu rõ là chưa đủ căn cứ pháp lý.\n"
    "4. Trích dẫn chính xác tên điều, số điều và văn bản từ ngữ cảnh khi viện dẫn căn cứ.\n"
    "5. Chỉ đưa ra nội dung câu trả lời cuối cùng, không giải thích quá trình suy nghĩ."
)

ANSWER_CONTROL_HEADER = "[ANSWER_CONTROL]"
ANSWER_CONTROL_FOOTER = "[/ANSWER_CONTROL]"


def build_generation_prompt(question: str, formatted_evidence: str) -> str:
    """Build authoritative prompt for Qwen3.5-2B generator."""
    prompt = (
        f"{ANSWER_CONTROL_HEADER}\n"
        f"{SYSTEM_INSTRUCTION_VIETNAMESE}\n"
        f"{ANSWER_CONTROL_FOOTER}\n\n"
        f"Căn cứ pháp lý:\n{formatted_evidence}\n\n"
        f"Câu hỏi:\n{question}\n\n"
        f"Câu trả lời:"
    )
    return prompt
