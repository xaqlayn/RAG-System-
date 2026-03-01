# -----------------------------
# Configuration
# -----------------------------

from dataclasses import dataclass
import os

@dataclass(frozen=True)
class Settings:
    pdf_path: str = "./nke-10k-2023.pdf"
    llm_model: str = "llama3.1:8b"
    embedding_model: str = "nomic-embed-text"
    temperature: float = 0.1
    chunk_size: int = 1000
    chunk_overlap: int = 200
    add_start_index: bool = True
    top_k_default: int = 2


def configure_environment() -> None:
    """
    Keep networking behavior explicit & reproducible.
    Adjust if your environment differs.
    """
    os.environ.setdefault("no_proxy", "localhost,127.0.0.1")
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")