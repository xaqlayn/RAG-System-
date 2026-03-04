# 🦙 Local RAG System

A fully local, privacy-first **Retrieval-Augmented Generation (RAG)** system. This project allows you to interact with your private documents using **Llama 2** and **Nomic Embeddings**, ensuring that no data ever leaves your machine.

---

## 🌟 Key Features
* **Total Privacy:** Runs 100% locally via **Ollama**. No cloud APIs, no data leaks.
* **Advanced Embeddings:** Utilizes `nomic-embed-text` for high-performance semantic search.
* **Smart Retrieval:** Powered by **ChromaDB** for efficient vector storage and document indexing.
* **Context-Aware Chat:** Uses **Llama 2** to generate answers grounded strictly in your provided documents.

---

## 🏗️ Architecture
1.  **Ingest:** Load documents (PDF, TXT, MD).
2.  **Split:** Break text into optimized chunks using `RecursiveCharacterTextSplitter`.
3.  **Embed:** Convert text chunks into vectors using the **Nomic** model via Ollama.
4.  **Store:** Save vectors into a local **ChromaDB** instance.
5.  **Query:** Retrieve relevant context and generate a response using **Llama 2**.

---

## 🛠️ Tech Stack
* **LLM:** [Llama 2](https://ollama.com/library/llama2)
* **Embeddings:** [Nomic-Embed-Text](https://ollama.com/library/nomic-embed-text)
* **Orchestration:** [LangChain](https://python.langchain.com/)
* **Vector Database:** [ChromaDB](https://www.trychroma.com/)
* **Environment:** [Ollama](https://ollama.com/)

---

## 🚀 Getting Started

### 1. Install Ollama
Download and install Ollama from [ollama.com](https://ollama.com/). Once installed, pull the required models:
```bash
ollama pull llama2
ollama pull nomic-embed-text
