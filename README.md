# 🎓 Workshop 4: AI Course Assistant - Advanced Multimodal Two-Stage RAG & ReAct Agent System

An enterprise-grade, multi-turn **Retrieval-Augmented Generation (RAG)** chatbot and autonomous **ReAct AI Agent** system for the **AI Application Engineer** course. Built using **LangChain**, **LangGraph**, **Pinecone Serverless Vector Store** with **Native Inference Re-ranking (`bge-reranker-v2-m3`)**, **Tavily Web Search**, and **Multimodal Vision Analysis**.

---

## 🌟 Key Improvements Over Workshop 3

| Feature / Component | Workshop 3 (Baseline) | Workshop 4 (Upgraded System) |
| :--- | :--- | :--- |
| **Vector Database** | Local ChromaDB with basic vector ingestion. | **Pinecone Serverless Vector Store** with cosine similarity and enriched metadata filtering (`category`, `doc_code`, `page_number`, `source_file`). |
| **Retrieval Architecture** | Single-stage dense retrieval (`top_k = 5`) susceptible to context loss and noise. | **Two-Stage Retrieval (Retrieve & Re-rank)**: Pinecone Bi-Encoder broad retrieval (`fetch_k = 15`) combined with Pinecone Inference Cross-Encoder (`bge-reranker-v2-m3`) to rank Top 5 most relevant chunks. |
| **Document & Slide Parsing** | PyMuPDF (`fitz`) basic manual page text slicing. | **PyMuPDF Engine (`fitz`) + Multimodal Vision OCR**: High-performance layout reading for PPT/PDFs with automated screenshot transcription via `gpt-4o-mini` before splitting. |
| **Retrieval Precision** | Raw dense vector similarity without query classification. | **Self-Querying Parameterization**: ReAct Agent dynamically formulates English semantic queries (`query`) while extracting explicit target identifiers (`doc_code`, `category`) to eliminate cross-document retrieval confusion. |
| **Chunk Enrichment** | Basic page-level chunking. | **Post-Split Context Attachment**: Injects compact standardized headers (`Document`, `Category`, `Code`) onto chunks *after* splitting, preventing empty header fragmentation. |
| **Agent Orchestration** | Manual tool-calling loop via direct OpenAI chat completions. | **LangGraph ReAct Agent (`create_react_agent`)** with autonomous multi-step reasoning, tool execution, and dynamic query translation. |
| **Source Attribution** | Unstructured citation inside text (prone to hallucination and TTS artifacts). | **Direct Object-Level Metadata Extraction**: Extracts source files and page numbers directly from retrieved `Document` objects and displays them cleanly in an expandable UI component. |
| **Prompt Engineering** | Raw string system prompt. | **LangChain `ChatPromptTemplate`**, `FewShotChatMessagePromptTemplate`, and `MessagesPlaceholder` with Chain-of-Thought reasoning guidelines. |
| **External Real-time Data** | None (Limited strictly to internal course documents). | **Tavily Web Search Tool** for live web intelligence, library breaking changes, and external API documentation. |
| **Multimodal Capabilities** | Text-only query support. | **Multimodal Vision Engine** via `gpt-4o-mini` with Pydantic Structured Outputs to debug user-submitted screenshot errors and inspect architecture diagrams. |
| **Memory Management** | Full unconstrained message history appending. | **Bounded Sliding Window Conversation Memory** (preserving the most recent 10 turns to conserve token budget). |

---

## 🛠️ Requirements & Setup

### 1. Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Fast Python package manager) or standard Python `pip`
- OpenAI / Custom Gateway credentials with Chat, Embedding, and Vision support (`gpt-4o-mini`, `text-embedding-3-small`)
- [Pinecone API key](https://app.pinecone.io/) (Serverless Index & Inference support)
- [Tavily Search API key](https://app.tavily.com/) (Optional, for web search capabilities)

### 2. Environment Configuration

Copy the example environment file and fill in your credentials:

```bash
cp .env.example .env
```

Ensure your `.env` contains:

```env
# OpenAI / Custom Gateway Configuration
OPENAI_ENDPOINT=https://your-custom-endpoint.com/v1
OPENAI_API_KEY=your_openai_api_key
OPENAI_CHAT_MODEL=gpt-4o-mini
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
OPENAI_VISION_MODEL=gpt-4o-mini

# Pinecone Vector Store & Inference Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=course-knowledge-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_RERANK_MODEL=bge-reranker-v2-m3

# External Tools Configuration
TAVILY_API_KEY=your_tavily_api_key
```

---

## 🚀 Execution & Quick Start

Use the included startup scripts inside the `scripts/` directory to automatically set up the virtual environment, install dependencies, and launch the Streamlit interface:

### Option 1: macOS / Linux (Bash)

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

### Option 2: Windows (PowerShell)

```powershell
.\scripts\start.ps1
```

Tip: Pass `-NoInstall` to skip re-installing dependencies on subsequent runs.

### Option 3: Windows (Command Prompt)

```cmd
scripts\start.bat
```

---

## 🏗️ System Architecture

```text
                                 +-------------------------+
                                 |   User (Text / Image)   |
                                 +------------+------------+
                                              |
                                              v
                                 +-------------------------+
                                 | Streamlit UI (src/app.py)|
                                 +------------+------------+
                                              |
                     +------------------------+------------------------+
                     | (User Screenshot Uploaded)                      | (Text / Augmented Prompt)
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |      VisionEngine       |                       |   CourseAgentRunner     |
        | (src/engines/vision...) |                       | (src/agent/agent_runner)|
        | (Structured Extraction) |                       | (LangGraph ReAct Loop)  |
        +------------+------------+                       +------------+------------+
                     |                                                 |
                     +----------------> [Augmented Context] ----------->
                                                                       | (Dynamic Tool Selection)
                                            +--------------------------+--------------------------+
                                            |                                                     |
                                            v                                                     v
                            +-------------------------------+                     +-------------------------------+
                            |   search_course_knowledge     |                     |         TavilySearch          |
                            |     (src/agent/tools.py)      |                     |     (src/agent/tools.py)      |
                            +---------------+---------------+                     +---------------+---------------+
                                            |                                                     |
                                            v                                                     v
                            +-------------------------------+                     +-------------------------------+
                            |   Stage 1: Broad Retrieval    |                     |      Live Web Intelligence    |
                            |   (Pinecone Vector Search)    |                     |       (External Web API)      |
                            |       (fetch_k = 15)          |                     +-------------------------------+
                            +---------------+---------------+
                                            |
                                            v
                            +-------------------------------+
                            |   Stage 2: Neural Re-Ranking  |
                            |  (pc.inference.rerank API)    |
                            |   (bge-reranker-v2-m3, top=5) |
                            +---------------+---------------+
                                            |
                                            v
                            +-------------------------------+
                            | PyMuPDF + Document Ingestion  |
                            | (src/rag/document_processor)  |
                            +---------------+---------------+
                                            |
                                            v
                            +-------------------------------+
                            |    Course PDFs (resources/)   |
                            +-------------------------------+
```

---

## 📂 Repository Structure

```text
workshop4/
├── resources/                  # Internal course PDF documents (Assignments, Workshops, Guidelines)
├── scripts/                    # Startup automation scripts
│   ├── start.bat               # Windows CMD launcher
│   ├── start.ps1               # Windows PowerShell launcher
│   └── start.sh                # macOS / Linux Bash launcher
├── src/                        # Core system source code
│   ├── agent/                  # Agent reasoning & tool handling
│   │   ├── __init__.py
│   │   ├── agent_runner.py     # LangGraph ReAct Agent orchestration & source metadata extraction
│   │   ├── prompts.py          # LangChain ChatPromptTemplate with CoT & Few-Shot Examples
│   │   └── tools.py            # LangChain Tools (Two-Stage RAG retrieval & Tavily Search)
│   ├── engines/                # Specialized service engines
│   │   ├── __init__.py
│   │   ├── tts_engine.py       # Text-to-Speech Engine (gTTS + Language Auto-Detection)
│   │   └── vision_engine.py    # Multimodal Vision Analyzer with Structured Output
│   ├── rag/                    # Data processing & Vector Database
│   │   ├── __init__.py
│   │   ├── document_processor.py # PDF Loader (PyMuPDF) & Semantic Text Splitter
│   │   └── vector_store.py     # Pinecone Vector Store with Native Inference Re-ranking
│   ├── __init__.py
│   └── app.py                  # Main Streamlit Web Application
├── tests/                      # Unit test suites
├── .env.example                # Sample environment variables template
├── .gitignore                  # Git ignore rules
├── README.md                   # Project documentation
├── requirements.txt            # Project dependencies
└── TASKS_LIST.md               # Task tracking checklist

```

---

## 💡 Methodology & Approach Explanation

1. **High-Fidelity Document & Visual Ingestion:**
    - Employs **PyMuPDF (`fitz`)** for native bounding-box layout parsing across converted PowerPoint slides (Workshops) and PDF guidelines.
    - Automates image inspection by extracting embedded screenshots and transcribing UI elements, code snippets, and button actions using `VisionEngine` (`gpt-4o-mini`) directly during document ingestion.

2. **Semantic Chunking & Post-Split Context Enrichment:**
    - Applies `RecursiveCharacterTextSplitter` on clean body text to preserve natural paragraph and sentence boundaries.
    - Injects standardized metadata headers (`Document: ... | Category: ... | Code: ...`) directly onto chunks *after* splitting, ensuring every chunk retains document identity without generating empty header fragments.

3. **Two-Stage Retrieval (Retrieve & Re-Rank):**
    - **Stage 1 (Bi-Encoder Retrieval):** Fetches an expanded candidate pool (`fetch_k = 15`) from Pinecone using dense vector embeddings to maximize recall and prevent context loss.
    - **Stage 2 (Cross-Encoder Re-Ranking):** Applies Pinecone's native inference reranking (`pc.inference.rerank` with `bge-reranker-v2-m3`) to re-score candidate chunks based on full query-passage cross-attention, returning the Top 5 most relevant chunks.

4. **Self-Querying Parameterized Retrieval:**
    - Provides structured parameter hooks (`query`, `category`, `doc_code`) in `search_course_knowledge`.
    - The ReAct Agent isolates document identifiers (e.g., `"Assignment 10"` $\rightarrow$ `doc_code="assignment_10"`) and converts semantic intent into enriched English search terms, ensuring zero cross-document vector drift.

5. **Autonomous Multi-Tool Orchestration & Clean Attribution:**
    - Powered by `create_react_agent` from `langgraph.prebuilt` to dynamically decide between internal knowledge search and external research (`Tavily`).
    - Source documents and page numbers are captured directly from Python `Document` metadata objects and displayed cleanly in UI expanders, ensuring pristine TTS output without reading raw filenames aloud.