# 👟 Nike 10-K — Local RAG Agent

![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1.x-1C3C3C)
![Ollama](https://img.shields.io/badge/LLM-Ollama%20(local)-000000?logo=ollama&logoColor=white)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Deploy-Docker%20Compose-2496ED?logo=docker&logoColor=white)
![Stars](https://img.shields.io/github/stars/xaqlayn/RAG_Agent?style=social)

A **fully local** RAG application that answers questions about Nike's 2023 10-K
filing. No cloud LLM calls, no API keys, no data leaving your machine —
everything runs through **Ollama** (Llama 3.1 for generation, Nomic for
embeddings), orchestrated with **LangChain** and served through a
**Streamlit** chat UI.

## 🔐 Privacy-first by design

- **100% local inference** — powered by Ollama, nothing ever leaves your machine
- **No third-party APIs** — no dependency on OpenAI, Anthropic, or any cloud LLM
- **Ideal for sensitive documents** — built for cases where cloud uploads aren't an option

## 🗺️ Architecture diagram

<p align="center">
  <img src="docs/workflow-diagram.svg" alt="RAG pipeline and Docker architecture diagram" width="900">
</p>

```mermaid
flowchart LR
    subgraph APP["🐳 rag-app container · Streamlit :8501"]
        direction TB
        subgraph ingest["Ingestion (cached on startup)"]
            direction LR
            PDF(["📄 Nike 10-K PDF"]) --> SPLIT["LOAD & SPLIT<br/>PyPDFLoader + splitter"]
            SPLIT --> EMBED["EMBED<br/>nomic-embed-text"]
            EMBED --> VS[("VECTOR STORE<br/>InMemoryVectorStore")]
        end
        subgraph query["Per query"]
            direction LR
            Q(["💬 User Question"]) --> RET["RETRIEVE<br/>top-k similarity search"]
            RET --> CTX["BUILD CONTEXT<br/>system + human message"]
            CTX --> GEN["GENERATE<br/>Llama 3.1, streamed"]
        end
        VS -.-> RET
        GEN --> ANS(["🏁 Answer + sources"])
    end

    subgraph OLLAMA["🐳 ollama container · :11434"]
        MODELS["llama3.1:8b<br/>nomic-embed-text"]
    end

    EMBED -. docker network .-> MODELS
    GEN -. docker network .-> MODELS

    classDef retrieve fill:#3b82f6,color:#fff,stroke:#1d4ed8,stroke-width:2px
    classDef embed fill:#f59e0b,color:#111,stroke:#b45309,stroke-width:2px
    classDef generate fill:#22c55e,color:#111,stroke:#15803d,stroke-width:2px
    classDef storage fill:#0891b2,color:#fff,stroke:#0e7490,stroke-width:2px
    classDef context fill:#6b7280,color:#fff,stroke:#374151,stroke-width:2px
    classDef terminal fill:#111827,color:#fff,stroke:#000,stroke-width:2px
    classDef container fill:#0f172a,color:#fff,stroke:#000,stroke-width:2px

    class SPLIT,RET retrieve
    class EMBED embed
    class GEN generate
    class VS storage
    class CTX context
    class PDF,Q,ANS terminal
    class MODELS container
```

1. **Ingestion** (runs once, cached via `st.cache_resource`) — load the PDF,
   split it into chunks, embed them with **Nomic** via Ollama, and hold them
   in an in-memory vector store.
2. **Retrieve** — on each question, pull the top-k most similar chunks.
3. **Build context** — assemble a system + human message pair grounded in
   those chunks.
4. **Generate** — **Llama 3.1** streams the answer token-by-token into the
   chat UI, with a "View Sources" expander showing exactly which chunks were used.

## 🧠 Two retrieval paths in this repo

- **`app.py` (the deployed Streamlit app)** — a straightforward retrieve →
  generate chain: `Retriever_tool_agent_setup.make_retriever_chain` does a
  similarity search, and the result is fed directly into `ChatOllama`.
- **`Retriever_tool_agent_setup.run_agent_demo`** (used from `main.py`) — an
  experimental **agentic** variant using LangChain's `create_agent` with a
  `dynamic_prompt` middleware that injects retrieved context per-turn.

## 🛠️ Tech stack

| Layer | Choice |
|---|---|
| LLM | Llama 3.1:8b (local, via Ollama) |
| Embeddings | Nomic-Embed-Text (local, via Ollama) |
| Orchestration | LangChain (`ChatOllama`, `OllamaEmbeddings`, agents) |
| Vector store | `InMemoryVectorStore` |
| UI | Streamlit |
| Deployment | Docker, Docker Compose, Docker Hub |

## 📦 Project structure

| File | Responsibility |
|------|-----------------|
| `Configuration.py` | `Settings` dataclass — model names, chunk size, paths |
| `Pipeline_steps.py` | Load PDF → split → build the vector store |
| `Retriever_tool_agent_setup.py` | Retriever chain, retrieval tool, and the agent demo |
| `main.py` | CLI entry point — runs the full pipeline + agent demo |
| `app.py` | Streamlit chat UI (the deployed app) |
| `nke-10k-2023.pdf` | Sample document — Nike's 2023 10-K filing |
| `Dockerfile` / `docker-compose.yml` | Container build + two-service orchestration (app + Ollama) |
| `setup.sh` | One-command Docker setup: pulls images and models |

## 🚀 Setup

### 🐳 Option A: Docker (recommended)

```bash
git clone https://github.com/xaqlayn/RAG_Agent.git
cd RAG_Agent
chmod +x setup.sh
./setup.sh
```

`setup.sh` starts the containers, waits for Ollama, and pulls `llama3.1:8b`
and `nomic-embed-text`. Then open [http://localhost:8501](http://localhost:8501).

<details>
<summary>Manual Docker steps</summary>

```bash
docker compose up -d
docker exec -it ollama ollama pull llama3.1:8b
docker exec -it ollama ollama pull nomic-embed-text
```

</details>

### 🐍 Option B: Local development

```bash
pip install -r requirements.txt

# make sure Ollama is running locally and pull the models
ollama pull llama3.1:8b
ollama pull nomic-embed-text

streamlit run app.py
```

## 🔧 Notable implementation details

- **Resource-constrained friendly** — runs on 8GB RAM using `python:3.11-slim`.
- **Service isolation** — Ollama and the Streamlit app run as separate
  containers, talking only over the internal Docker bridge network.
- **Grounded answers** — the system prompt explicitly instructs the model to
  say "I don't know" rather than hallucinate outside the retrieved context.

## 👨‍💻 Author

**Saqlain Majeed** — [GitHub](https://github.com/xaqlayn) · [Docker Hub](https://hub.docker.com/u/xack1122)
