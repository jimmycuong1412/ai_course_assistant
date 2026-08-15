# 📋 Workshop2 - Danh sách phân công công việc

## Thành viên và Công việc

| Thành viên | Công việc |
|---|---|
| **GiapHN** | Xây dựng ChromaDB: tạo embeddings offline, lưu trữ persistent, chunking PDF và triển khai truy vấn tìm kiếm theo ngữ nghĩa |
| **CuongTQ7** | Tích hợp Azure OpenAI SDK (`api_client.py`): `make_api_call` tạo client mới mỗi request kèm anti-caching header + `X-Request-ID`; retry bằng tenacity (exponential backoff, 5 lần) cho lỗi RateLimit/Connection/API; hỗ trợ streaming và `tool_choice=auto`. Thiết kế `SYSTEM_PROMPT` Chain-of-Thought + few-shot: ép model gọi tool khi gặp từ khóa khóa học và luôn trả lời đúng ngôn ngữ người hỏi |
| **DungPQ6** | Tích hợp STT (faster-whisper) và TTS (gTTS); chuyển giọng nói thành câu hỏi và câu trả lời thành audio; tự nhận diện ngôn ngữ Việt/Anh |
| **AnNPH5** | Xây dựng giao diện Streamlit; sidebar cấu hình la mã; quản lý phiên hội thoại nhiều lượt; ép câu trả lời đúng ngôn ngữ người hỏi |
| **LocNTH** | Định nghĩa function calling schema (`search_course_knowledge`); triển khai tool handler và xử lý lỗi/edge cases khi search thất bại |
| **ThangTP3** | Thiết kế conversation flow; tinh chỉnh sidebar và hiển thị trạng thái knowledge base; chuẩn bị bộ câu hỏi demo |
| **TrucNLT** | Viết unit tests (pytest + mock) và kiểm thử end-to-end; chuẩn bị conversation logs minh họa multi-turn |
| **VietNQ32** | Làm slide kiến trúc hệ thống và insight dự án; thuyết trình chính trên sân khấu; chuẩn bị mock data user queries/FAQs |
| **ThanhVD7** | Tích hợp cuối pipeline STT → RAG → LLM → TTS trôi chảy; rà soát lỗi và đảm bảo demo chạy mượt |
 