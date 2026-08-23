# 📋 Workshop4 - Danh sách phân công công việc

## Thành viên và Công việc

| Thành viên | Công việc |
|---|---|
| **GiapHN** | Xây dựng **Pinecone Serverless Vector Store** (`src/rag/vector_store.py`) thay ChromaDB; **Document/OCR pipeline** (`src/rag/document_processor.py`, `src/engines/vision_engine.py`): parse PDF/slide bằng PyMuPDF, OCR screenshot qua `gpt-4o-mini`, gắn metadata header sau khi chunking |
| **CuongTQ7** | Chuyển từ Azure OpenAI SDK (`api_client.py`) sang **LangGraph ReAct Agent** (`src/agent/agent_runner.py`): `create_react_agent`, vòng lặp Thought→Action→Observation, sliding-window memory (10 lượt gần nhất), gắn tài liệu tham khảo vào response |
| **DungPQ6** | Giữ **TTS Engine** (`src/engines/tts_engine.py`: gTTS + langdetect tự nhận diện ngôn ngữ); STT đã bỏ ở workshop4 nên nhận thêm phần **Vision Engine** đầu vào ảnh (`src/engines/vision_engine.py`) — debug screenshot lỗi, đọc diagram bằng Structured Output |
| **AnNPH5** | Xây dựng lại giao diện Streamlit (`src/app.py`): upload ảnh cho multimodal, sidebar cấu hình, quản lý phiên hội thoại nhiều lượt, hiển thị nguồn tài liệu trích dẫn |
| **LocNTH** | Định nghĩa lại tool cho agent (`src/agent/tools.py`): `search_course_knowledge` với self-querying (`query`, `category`, `doc_code`) + tool `TavilySearch`; xử lý lỗi/edge case khi search thất bại |
| **ThangTP3** | Thiết kế lại prompt (`src/agent/prompts.py`): `ChatPromptTemplate` + `FewShotChatMessagePromptTemplate` + Chain-of-Thought; chuẩn bị bộ câu hỏi demo cho kiến trúc mới (RAG nội bộ + web search + vision) |
| **TrucNLT** | Viết lại unit test (pytest + mock) cho agent/tools/vision trong `tests/`; kiểm thử end-to-end pipeline mới; chuẩn bị conversation logs minh họa multi-turn + multimodal |
| **VietNQ32** | Cập nhật slide kiến trúc hệ thống theo sơ đồ mới (Pinecone, ReAct Agent, Vision, Tavily) và insight dự án; thuyết trình chính; chuẩn bị mock data user queries/FAQs + ảnh demo cho vision |
| **ThanhVD7** | Tích hợp end-to-end pipeline mới: Vision → Agent (ReAct) → Tools (Pinecone/Tavily) → TTS; rà soát lỗi giữa các module đã restructure và đảm bảo demo chạy mượt |
 