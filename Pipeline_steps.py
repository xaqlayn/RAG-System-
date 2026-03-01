
# -----------------------------
# Pipeline steps
# -----------------------------

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import Iterable, List, Tuple
import os


def load_pdf(path: str) -> List[Document]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"PDF not found at path: {path}\n"
            f"Place the file there or update Settings.pdf_path."
        )
    loader = PyPDFLoader(path)
    return loader.load()


def split_documents(
    docs: Iterable[Document],
    *,
    chunk_size: int,
    chunk_overlap: int,
    add_start_index: bool,
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=add_start_index,
    )
    return splitter.split_documents(list(docs))


def build_vector_store(chunks: List[Document], embeddings: OllamaEmbeddings) -> InMemoryVectorStore:
    vector_store = InMemoryVectorStore(embeddings)
    vector_store.add_documents(documents=chunks)
    return vector_store


def print_top_doc(results: List[Document], *, label: str) -> None:
    if not results:
        print(f"{label}: (no results)")
        return
    print(f"{label}:\n{results[0]}\n")


def run_basic_queries(vector_store: InMemoryVectorStore, embeddings: OllamaEmbeddings) -> None:
    results = vector_store.similarity_search("When was Nike incorporated?", k=1)
    print_top_doc(results, label="Q1 top result")

    scored: List[Tuple[Document, float]] = vector_store.similarity_search_with_score(
        "What was Nike's revenue in 2023?",
        k=1,
    )
    if scored:
        doc, score = scored[0]
        print(f"Q2 top score: {score}\n")
        print(doc, "\n")

    query_vec = embeddings.embed_query("How were Nike's margins impacted in 2023?")
    by_vec = vector_store.similarity_search_by_vector(query_vec, k=1)
    print_top_doc(by_vec, label="Q3 by-vector top result")


