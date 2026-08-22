# 🎓 Workshop 4: AI Course Assistant - Advanced Multimodal RAG & ReAct Agent System

An enterprise-grade, multi-turn **Retrieval-Augmented Generation (RAG)** chatbot and autonomous **ReAct AI Agent** system for the **AI Application Engineer** course. Built using **LangChain**, **LangGraph**, **Pinecone Serverless Vector Store**, **Tavily Web Search**, and **Multimodal Vision Analysis**.

---

## 🌟 Key Improvements Over Workshop 3

| Feature / Component | Workshop 3 (Baseline) | Workshop 4 (Upgraded System) |
| :--- | :--- | :--- |
| **Vector Database** | Local ChromaDB with basic vector ingestion. | **Pinecone Serverless Vector Store** with cosine similarity and metadata filtering (`category`, `page`, `source_file`). |
| **Document Processing** | PyMuPDF (`fitz`) manual page reading and slicing. | **`PyPDFLoader`** + **`RecursiveCharacterTextSplitter`** with chunk overlaps and enriched chunk-level metadata. |
| **Agent Orchestration** | Manual tool-calling loop via direct OpenAI chat completions. | **LangGraph ReAct Agent (`create_react_agent`)** with autonomous reasoning, tool selection, and execution. |
| **Prompt Engineering** | Raw string system prompt. | **LangChain `ChatPromptTemplate`**, `FewShotChatMessagePromptTemplate`, and `MessagesPlaceholder`. |
| **External Real-time Data** | None (Limited strictly to internal course documents). | **Tavily Web Search Tool** for live web intelligence, library breaking changes, and external API documentation. |
| **Multimodal Capabilities** | Text-only query support. | **Multimodal Vision Engine** via `gpt-4o-mini` with Pydantic Structured Outputs to debug screenshot errors and inspect architecture diagrams. |
| **Memory Management** | Full unconstrained message history appending. | **Bounded Sliding Window Conversation Memory** (preserving the most recent 10 turns to conserve token budget). |

---

## 🛠️ Requirements & Setup

### 1. Prerequisites

- Python 3.10+
- [uv](https://github.com/astral-sh/uv) (Fast Python package manager) or standard Python `pip`
- OpenAI / Custom Gateway credentials with Chat, Embedding, and Vision support (`gpt-4o-mini`, `text-embedding-3-small`)
- [Pinecone API key](https://app.pinecone.io/) (Serverless Index support)
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

# Pinecone Vector Store Configuration
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=course-knowledge-index
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1

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
                                 |  Streamlit UI (app.py)  |
                                 +------------+------------+
                                              |
                     +------------------------+------------------------+
                     | (Image Uploaded)                                | (Text / Augmented Prompt)
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |      VisionEngine       |                       |   CourseAgentRunner     |
        |   (vision_engine.py)    |                       |   (agent_runner.py)     |
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
                            |          (tools.py)           |                     |          (tools.py)           |
                            +---------------+---------------+                     +---------------+---------------+
                                            |                                                     |
                                            v                                                     v
                            +-------------------------------+                     +-------------------------------+
                            |    Pinecone Vector Store      |                     |      Live Web Intelligence    |
                            |       (vector_store.py)       |                     |       (External Web API)      |
                            +---------------+---------------+                     +-------------------------------+
                                            |
                                            v
                            +-------------------------------+
                            |     PyPDFLoader + Splitter    |
                            |    (document_processor.py)    |
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
│   │   ├── agent_runner.py     # LangGraph ReAct Agent orchestration & history window
│   │   ├── prompts.py          # LangChain ChatPromptTemplate with CoT & Few-Shot Examples
│   │   └── tools.py            # LangChain Tools (Pinecone RAG retrieval & Tavily Search)
│   ├── engines/                # Specialized service engines
│   │   ├── __init__.py
│   │   ├── tts_engine.py       # Text-to-Speech Engine (gTTS + Language Auto-Detection)
│   │   └── vision_engine.py    # Multimodal Vision Analyzer with Structured Output
│   ├── rag/                    # Data processing & Vector Database
│   │   ├── __init__.py
│   │   ├── document_processor.py # PDF Loader (PyPDFLoader) & Semantic Text Splitter
│   │   └── vector_store.py     # Pinecone Serverless Vector Store & Similarity Search
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

1. **Semantic Chunking & Metadata Enrichment:**
    - Utilizes `PyPDFLoader` to extract page-level context and `RecursiveCharacterTextSplitter` with natural boundary separators (`\n\n`, `\n`, `. `) to prevent mid-sentence truncation.
    - Attaches structural metadata (`category`, `source_file`, `page_number`, `chunk_id`) to every chunk to enable filtered vector retrieval.

2. **Autonomous Tool Routing (ReAct Agent Pattern):**
    - Employs `create_react_agent` from `langgraph.prebuilt` coupled with `ChatOpenAI`.
    - The model dynamically cycles through Thought $\rightarrow$ Action $\rightarrow$ Observation phases, deciding whether to query Pinecone, query Tavily, or answer directly.

3. **Multimodal Vision Integration:**
    - Converts uploaded images to Base64 data streams and analyzes them using `ChatOpenAI` with Pydantic Structured Outputs (`VisionAnalysisResponse`), extracting actionable error details before agent processing.