# -----------------------------
# Main
# -----------------------------


from langchain_ollama import ChatOllama, OllamaEmbeddings
import Retriever_tool_agent_setup
import Pipeline_steps
import Configuration




def main() -> None:
    Configuration.configure_environment()
    settings = Configuration.Settings()

    docs = Pipeline_steps.load_pdf(settings.pdf_path)
    print(f"Loaded pages: {len(docs)}")

    chunks = Pipeline_steps.split_documents(
        docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        add_start_index=settings.add_start_index,
    )
    print(f"Chunks: {len(chunks)}")

    llm = ChatOllama(model=settings.llm_model, temperature=settings.temperature)
    embeddings = OllamaEmbeddings(model=settings.embedding_model)

    vector_store = Pipeline_steps.build_vector_store(chunks, embeddings)

    Pipeline_steps.run_basic_queries(vector_store, embeddings)

    retriever_chain = Retriever_tool_agent_setup.make_retriever_chain(vector_store)
    retriever_chain.batch(
        [
            "How many distribution centers does Nike have in the US?",
            "When was Nike incorporated?",
        ],
    )

    retrieve_context = Retriever_tool_agent_setup.make_retrieve_context_tool(vector_store, k=settings.top_k_default)
    _ = retrieve_context  # keep referenced to avoid "unused" confusion in some linters

    # Optional: agent demo (comment out if you only want retrieval)
    Retriever_tool_agent_setup.run_agent_demo(llm, vector_store)


if __name__ == "__main__":
    main()