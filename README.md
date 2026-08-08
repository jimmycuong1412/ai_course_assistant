# 🎓 AI Course Assistant - Real-World Chatbot System

An intelligent multi-turn AI Assistant for the **AI Application Engineer** course. Built with Azure OpenAI API (`gpt-4o-mini`), Streamlit, `rank_bm25` search engine with `spaCy` lemmatization, and robust retry mechanics.

This repository serves as the complete implementation for **Workshop 2: Building Real-World Chatbot Systems Using Azure OpenAI API**.

---

## 🌟 Key Features

* **Multi-Turn Conversation & History Management**: Preserves dialogue context across interactions with automatic request history sanitization.
* **Function Calling / Tool Integration**: Dynamic knowledge retrieval from course PDFs (Assignments, Workshops, Guidelines) using a dedicated `search_course_knowledge` tool.
* **Advanced Search Engine (`rank_bm25` + `spaCy`)**:
  * Lemmatization via `spaCy` (`en_core_web_sm`) to handle word variations (e.g., *summarize*, *summarizing*, *summary* -> *summary*).
  * High-precision keyword scoring using BM25Okapi algorithm.
* **Resilient API Architecture**:
  * **Tenacity Retry Wrapper**: Automatic exponential backoff retries for transient errors (`RateLimitError`, `APIConnectionError`, `APIError`).
  * **Anti-Caching Header Invalidation**: Generates clean TCP connections per call to bypass gateway proxy caching issues.
* **Batch Evaluation Suite**: CLI utility (`batch_eval.py`) to run batch test cases and export metrics to JSON.
* **Modular Clean Architecture**: Full separation between UI logic, API client wrappers, search engine, and tool handlers.
* **Unit Test Coverage**: Comprehensive suite using `pytest` and `pytest-mock`.

---

## 🏗️ System Architecture

```text
                   +-------------------------+
                   |      Streamlit UI       |
                   |        (app.py)         |
                   +------------+------------+
                                |
                                v
                   +-------------------------+
                   |     API Client Layer    |
                   |     (api_client.py)     |
                   | (Tenacity Retry / HTTP) |
                   +------------+------------+
                                |
        +-----------------------+-----------------------+
        |                                               |
        v (Turn 1: Tool Check)                          v (Turn 2: Response Stream)
+-----------------------+                       +-----------------------+
|  Azure OpenAI Model   |                       |  Azure OpenAI Model   |
|   (gpt-4o-mini)       |                       |   (gpt-4o-mini)       |
+-----------+-----------+                       +-----------------------+
        |
        | (Function Calling)
        v
+-----------------------+
|     Tool Handler      |
|      (tools.py)       |
+-----------+-----------+
        |
        v
+-----------------------+
| Course Search Engine  |
|  (search_engine.py)   |
|  (BM25 + spaCy NLP)   |
+-----------+-----------+
        |
        v
+-----------------------+
| Course PDFs / Chunks  |
|     (resources/)      |
+-----------------------+
```

---

## 📁 Repository Structure

```text
.
├── resources/              # PDF documents (Assignments, Workshops, Guidelines)
├── api_client.py           # Unified OpenAI client wrapper with retry logic
├── app.py                  # Streamlit Web User Interface
├── batch_eval.py           # Batch processing script for automated test suites
├── prompts.py              # System prompt definitions (CoT reasoning & Few-shot)
├── search_engine.py        # PDF Ingestion, spaCy Lemmatizer & BM25 Search Engine
├── tools.py                # OpenAI function schema & tool execution handler
├── requirements.txt        # Project dependencies
├── README.md               # Project documentation
└── tests/                  # Pytest unit tests
    ├── test_api_client.py
    ├── test_batch_eval.py
    ├── test_search_engine.py
    └── test_tools.py
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites

- Python: 3.10 or higher

- Environment Manager: `uv` or standard `venv`

### 2. Environment Setup

Clone the repository and create a virtual environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the lightweight `spaCy` English NLP model:

```bash
python -m spacy download en_core_web_sm
```

### 3. Configuration (.env)

Create a `.env` file in the root directory:

```text
AZURE_OPENAI_ENDPOINT=[https://your-azure-openai-endpoint.com/](https://your-azure-openai-endpoint.com/)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_MODEL=gpt-4o-mini
```

---

## 💻 Usage

### Run the Web Chatbot Application

Launch the Streamlit interactive chat interface:

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`. Enter your Azure settings in the sidebar (or let it pick defaults from .env) and start asking questions about course assignments or workshops.

### Run Batch Evaluation (`batch_eval.py`)

To run the automated test scenarios (TC_01 to TC_04) and export results to `batch_eval_results.json`:

```bash
python batch_eval.py
```

### Run Unit Tests

Run all unit tests using `pytest`:

```bash
pytest
```

Run tests with verbose output:

```bash
pytest -v
```

All API calls in the test suite are mocked using `pytest-mock`, ensuring zero API costs and fast execution.