from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
THIS_DIR = Path(__file__).resolve().parent
CHROMA_DB_PATH = str(THIS_DIR.parent.parent / "knowledge" / "chroma_db")

# Global retriever instance (lazy loaded)
_vectorstore = None


def get_vectorstore() -> Chroma:
    """
    Lazy-load and return the ChromaDB vectorstore.
    Reuses the same instance across multiple calls.
    """
    global _vectorstore
    if _vectorstore is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            persist_directory=CHROMA_DB_PATH, embedding_function=embeddings
        )
    return _vectorstore


def retrieve_context(user_query: str, k: int = 3) -> str:
    """
    Perform semantic search on the knowledge base and return relevant context.

    Args:
        user_query: The user's question or message
        k: Number of relevant documents to retrieve (default: 3)

    Returns:
        Formatted string containing the retrieved context
    """
    vectorstore = get_vectorstore()
    results: list[Document] = vectorstore.similarity_search(user_query, k=k)

    if not results:
        return "Keine relevanten Informationen gefunden."

    # Format the retrieved documents into a context string
    context_parts = []
    for i, doc in enumerate(results, 1):
        context_parts.append(f"[Dokument {i}]\n{doc.page_content}")

    return "\n\n".join(context_parts)
