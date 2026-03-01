# PythonProject101: RAG Pipeline with LangChain & Ollama

This project implements a Retrieval-Augmented Generation (RAG) pipeline to query and interact with PDF documents. It uses LangChain for orchestration, Ollama for local LLM and embedding models, and `uv` for dependency management.

## Overview

The pipeline performs the following steps:
1.  **Load**: Loads a PDF document (e.g., `nke-10k-2023.pdf`).
2.  **Split**: Splits the document into manageable chunks.
3.  **Embed & Store**: Generates embeddings using Ollama and stores them in an in-memory vector store.
4.  **Retrieve**: Performs similarity searches to find relevant chunks for a given query.
5.  **Augment & Generate**: (Optional) Uses an agent-like setup to answer questions based on the retrieved context.

## Requirements

- **Python**: 3.13 or higher (managed via `uv`)
- **Ollama**: Installed and running locally.
- **Models**:
    - LLM: `llama3.1:8b` (default, configurable)
    - Embeddings: `nomic-embed-text` (default, configurable)
- **Data**: A PDF file (defaults to `./nke-10k-2023.pdf`).

## Project Structure

```text
.
├── Configuration.py            # Settings and environment configuration
├── Pipeline_steps.py           # Core RAG steps: load, split, build vector store
├── Retriever_tool_agent_setup.py # Retriever chains, tools, and agent demo logic
├── main.py                     # Main entry point for the pipeline
├── nke-10k-2023.pdf            # Sample data (Nike 10-K report)
├── pyproject.toml              # Project dependencies and metadata
└── uv.lock                     # Lock file for deterministic dependencies
```

## Setup

1.  **Install `uv`**:
    Follow the [official installation guide](https://github.com/astral-sh/uv).

2.  **Install Dependencies**:
    ```bash
    uv sync
    ```

3.  **Prepare Ollama**:
    Ensure Ollama is running and pull the required models:
    ```bash
    ollama pull llama3.1:8b
    ollama pull nomic-embed-text
    ```

## Usage

To run the main RAG pipeline and agent demo:

```bash
uv run main.py
```

This will:
- Load the PDF specified in `Configuration.py`.
- Split it into chunks and build a vector store.
- Run a few basic test queries.
- Demonstrate a retriever chain batch execution.
- Launch an agent demo answering "When was Nike incorporated?".

## Configuration

Settings are managed in `Configuration.py` via the `Settings` dataclass. You can modify:

- `pdf_path`: Path to your PDF file.
- `llm_model`: The Ollama LLM model name.
- `embedding_model`: The Ollama embedding model name.
- `chunk_size` / `chunk_overlap`: Text splitting parameters.

### Environment Variables
- `no_proxy` / `NO_PROXY`: Set to `localhost,127.0.0.1` by default in `Configuration.configure_environment()` to ensure local communication with Ollama.

## Scripts

The following entry points are available:

- `main.py`: The primary script to execute the full pipeline.

## Tests

- [ ] **TODO**: Add automated tests for pipeline steps and retriever logic.

## License

- [ ] **TODO**: Specify the license for this project.
