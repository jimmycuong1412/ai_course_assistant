# 📋 Hackathon - Danh sách phân công công việc

## Thành viên và Công việc

| Thành viên | Công việc |
|---|---|
| **GiapHN** | Xây dựng và duy trì **Pinecone Serverless Vector Store** (`src/rag/vector_store.py`); hoàn thiện **Document/OCR pipeline** (`src/rag/document_processor.py`, `src/engines/vision_engine.py`) bằng PyMuPDF và `gpt-4o-mini`; xử lý metadata sau khi chunking |
| **CuongTQ7** | Xây dựng và duy trì **LangGraph ReAct Agent** (`src/agent/agent_runner.py`): tool routing, Thought → Action → Observation, sliding-window memory 10 lượt, trích xuất nguồn và hiển thị tóm tắt luồng xử lý của agent |
| **DungPQ6** | Duy trì **TTS Engine** (`src/engines/tts_engine.py`) và **Vision Engine** (`src/engines/vision_engine.py`); phát triển **Speech-to-Text** hỗ trợ tiếng Việt và tiếng Anh, có xử lý lỗi khi thiếu microphone hoặc API |
| **AnNPH5** | Hoàn thiện giao diện Streamlit (`src/app.py`): upload ảnh, sidebar, quản lý hội thoại nhiều lượt, hiển thị citations, gợi ý câu hỏi nhanh khi mở chat mới và gợi ý câu hỏi follow-up sau mỗi câu trả lời |
| **LocNTH** | Duy trì các tool cho agent (`src/agent/tools.py`): `search_course_knowledge` với self-querying (`query`, `category`, `doc_code`), `TavilySearch`, xử lý search rỗng, filter sai, lỗi mạng và thiếu API key |
| **ThangTP3** | Thiết kế và duy trì prompt (`src/agent/prompts.py`) với `ChatPromptTemplate`, `FewShotChatMessagePromptTemplate`; chuẩn bị câu hỏi demo cho RAG, web search và vision; định nghĩa nội dung tóm tắt luồng reasoning an toàn để hiển thị trên UI |
| **TrucNLT** | Viết unit test pytest và mock cho agent/tools/vision; kiểm thử end-to-end các luồng RAG, follow-up, multimodal, TTS/STT và truy vấn ngoài domain; chuẩn bị conversation logs minh họa cho demo |
| **VietNQ32** | Cập nhật slide kiến trúc hệ thống gồm Pinecone, ReAct Agent, Vision, Tavily và TTS/STT; chuẩn bị user stories, MVP feature list, mock queries/FAQs, ảnh demo, screenshots, test results và nội dung thuyết trình |
| **ThanhVD7** | Tích hợp end-to-end pipeline: Vision → Agent → Tools → TTS/STT; rà soát lỗi giữa các module; hoàn thiện hướng dẫn chạy, deployment và deploy ứng dụng; đảm bảo demo chạy ổn định trên Windows |
 