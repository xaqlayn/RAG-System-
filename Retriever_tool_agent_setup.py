# -----------------------------
# Retriever + tool + agent setup
# -----------------------------

from langchain_core.documents import Document
from langchain_core.runnables import chain
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import ChatOllama
from langchain.tools import tool
from typing import Iterable, List, Tuple


def make_retriever_chain(vector_store: InMemoryVectorStore):
    @chain
    def retriever(query: str) -> List[Document]:
        return vector_store.similarity_search(query, k=1)

    return retriever


def make_retrieve_context_tool(vector_store: InMemoryVectorStore, *, k: int):
    @tool(response_format="content_and_artifact")
    def retrieve_context(query: str):
        """Retrieve relevant chunks to help answer a query."""
        retrieved_docs = vector_store.similarity_search(query, k=k)
        serialized = "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}"
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    return retrieve_context


def run_agent_demo(llm: ChatOllama, vector_store: InMemoryVectorStore) -> None:
    # Imports kept local so the "basic pipeline" remains usable even if agent APIs change
    from langchain.agents import create_agent
    from langchain.agents.middleware import ModelRequest, dynamic_prompt

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        last_query = request.state["messages"][-1].text
        retrieved_docs = vector_store.similarity_search(last_query, k=4)
        docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

        return (
            "You are a helpful assistant. Use the following context in your response:"
            f"\n\n{docs_content}"
        )

    agent = create_agent(llm, tools=[], middleware=[prompt_with_context])

    query = "When was Nike incorporated?"
    for step in agent.stream(
        {"messages": [{"role": "user", "content": query}]},
        stream_mode="values",
    ):
        step["messages"][-1].pretty_print()


