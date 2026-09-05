# AI Course Assistant

AI Course Assistant is a domain-specific, multimodal assistant for the **AI Application Engineer** course. It answers questions about assignments, workshops, and guidelines using the course materials as its knowledge base. It also supports screenshot analysis, voice input/output, citations, conversation history, and contextual follow-up questions.

The application is implemented as a Streamlit UI backed by a LangGraph ReAct agent, Pinecone two-stage retrieval, OpenAI-compatible models, and optional Tavily web search.

## What It Does

- Searches course documents using natural-language questions.
- Filters retrieval by document category and normalized assignment/workshop code.
- Uses Pinecone dense retrieval followed by neural reranking.
- Shows source filenames and page numbers for retrieved course content.
- Maintains a bounded conversation window of the most recent 10 turns.
- Generates starter questions for a new chat and up to three contextual follow-up questions after an answer.
- Redirects unrelated questions back to course topics.
- Accepts uploaded PNG/JPG screenshots or diagrams for multimodal analysis.
- Supports Vietnamese and English speech input through PhoWhisper and Whisper.
- Generates Vietnamese or English audio on demand using gTTS.
- Shows observable execution steps and tool calls in the UI. It does not expose hidden chain-of-thought.
- Persists chat sessions locally in `.cache/chat_sessions.json`.

## Architecture

### Query flow

```text
User text / voice / image
          |
          v
    Streamlit UI (src/app.py)
          |
          +--> VisionEngine (when an image is uploaded)
          |
          v
    LangGraph ReAct Agent
          |
          +--> search_course_knowledge
          |       |
          |       +--> Pinecone similarity search (fetch_k=15)
          |       +--> Pinecone Inference reranking (top_k=5)
          |
          +--> Tavily Search (optional, for live external information)
          |
          v
    Answer + source metadata + execution steps
          |
          +--> Follow-up questions
          +--> Text-to-Speech (on demand)
```

### Document ingestion flow

```text
PDF files in resources/
          |
          v
PyMuPDF page extraction
          |
          +--> Vision analysis for embedded screenshots/diagrams
          |
          v
RecursiveCharacterTextSplitter
          |
          v
Metadata enrichment:
category, doc_code, source_file, page_number, chunk_id
          |
          v
OpenAI-compatible embeddings -> Pinecone Serverless index
```

Retrieval first obtains a broad candidate set with dense similarity search (`fetch_k=15`). When enough candidates are available, Pinecone Inference reranks them with `bge-reranker-v2-m3` and returns the most relevant five chunks. If reranking fails, the implementation falls back to the first `top_k` candidates.

## Repository Contents

```text
hackathon/
├── resources/                    # Course PDFs: assignments, workshops, guidelines
├── scripts/
│   ├── start.bat                 # Windows Command Prompt launcher
│   ├── start.ps1                 # Windows PowerShell launcher
│   └── start.sh                  # macOS/Linux launcher
├── src/
│   ├── app.py                    # Streamlit application and chat workflow
│   ├── agent/
│   │   ├── agent_runner.py       # LangGraph ReAct orchestration and source extraction
│   │   ├── followup_generator.py # Follow-up and off-topic redirect suggestions
│   │   ├── prompts.py             # System prompt and few-shot examples
│   │   └── tools.py               # Course search and optional Tavily tools
│   ├── engines/
│   │   ├── stt_engine.py         # Vietnamese/English speech-to-text
│   │   ├── tts_engine.py         # Language-aware text-to-speech
│   │   └── vision_engine.py      # Structured multimodal image analysis
│   └── rag/
│       ├── document_processor.py # PDF extraction, image analysis and chunking
│       └── vector_store.py        # Pinecone index and two-stage retrieval
├── tests/                        # Unit and Streamlit AppTest tests
├── .env.example                  # Environment variable template
├── requirements.txt              # Python dependencies
└── TASKS_LIST.md                 # Team task allocation
```

The repository currently contains 26 PDF resources: 14 assignments, 9 workshop files, and 3 guideline files. The test suite contains 11 `test_*.py` files.

## Requirements

- Python 3.10 or newer.
- An OpenAI-compatible chat, embedding, and vision endpoint.
- Pinecone API access for the Serverless index and inference reranking.
- Tavily API access is optional and enables live web search.
- A microphone is required for voice input.
- The first speech-to-text run may download the configured Hugging Face model.

## Configuration

Copy `.env.example` to `.env` and fill in the credentials for your environment:

```env
OPENAI_ENDPOINT=https://your-compatible-endpoint.example/v1
OPENAI_API_KEY=your_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_VISION_MODEL=gpt-4o-mini

PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=course-knowledge-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_RERANK_MODEL=bge-reranker-v2-m3

# Optional: enables Tavily live web search
TAVILY_API_KEY=your_tavily_api_key

# Optional model overrides
PHOWHISPER_MODEL=vinai/PhoWhisper-base
WHISPER_EN_MODEL=openai/whisper-base.en
```

Never commit `.env` or real API keys. The application can start with Tavily disabled, but the chat, embeddings, vision, and Pinecone configuration must match the capabilities of the configured endpoint and credentials.

## Run on Windows

From the repository root:

```powershell
.\scripts\start.ps1
```

The script creates or reuses a virtual environment, installs `requirements.txt`, sets `PYTHONPATH`, and starts Streamlit at `http://localhost:8501`.

To skip dependency installation after the first run:

```powershell
.\scripts\start.ps1 -NoInstall
```

Command Prompt:

```bat
scripts\start.bat
```

macOS/Linux:

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

Manual launch:

```bash
python -m pip install -r requirements.txt
PYTHONPATH=. streamlit run src/app.py
```

On Windows PowerShell, use `$env:PYTHONPATH = "."` before the manual Streamlit command if imports are not resolved.

## Using the Application

1. Open a new chat and choose a starter question, or type a question about an assignment, workshop, guideline, or technical topic.
2. For a screenshot or diagram question, upload a PNG/JPG file in the sidebar.
3. Choose Vietnamese or English before recording a voice message.
4. Review the answer and expand the execution steps when you need to see the invoked tool and a short result preview.
5. Open the source control to inspect filenames and page numbers.
6. Use the follow-up buttons to continue the conversation without retyping context.
7. Select **Listen** on an answer to generate audio.

## Testing

Run the full test suite from the repository root:

```bash
python -m pytest
```

The tests use mocks for external services and cover agent orchestration, prompts, tools, document processing, vector retrieval, speech/vision engines, source aggregation, sliding-window memory, follow-up generation, and Streamlit interaction flows. The Streamlit tests specifically cover starter questions, follow-up buttons, off-topic redirect mode, and protection against repeated processing of an uploaded image.

Tests validate application behavior without requiring live API calls. They do not prove the quality, latency, or availability of a production deployment.

## Current Scope and Limitations

- Pinecone, model endpoints, and required credentials are external dependencies.
- Tavily is optional; without `TAVILY_API_KEY`, the live web-search tool is not registered.
- Retrieval quality depends on the course PDFs, embedding model, metadata, and reranker availability.
- Speech models may require a large initial download and local compute resources.
- Chat sessions are stored locally for the current machine; there is no multi-user authentication or shared session database.
- No public deployment URL or benchmark result is claimed by this README. Add those details only after they have been verified.
- `to-do.md` tracks remaining product work such as deployment and any follow-up validation.

## Team and Project References

- Team task allocation: [TASKS_LIST.md](TASKS_LIST.md)
- Hackathon requirements and evaluation criteria: `../hackathon.md`
- Presentation materials: [slides/](slides/)
- Course knowledge base: [resources/](resources/)

Update the presentation with the verified team name, member list, GitHub URL, deployment URL, screenshots, and measured test/deployment results before submission.
