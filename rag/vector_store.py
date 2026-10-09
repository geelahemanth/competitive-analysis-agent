from langchain_chroma import Chroma

from config.settings import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIRECTORY
from services.embeddings import get_embeddings


def get_vector_store() -> Chroma:
    """
    Return the Chroma vector store used by the application.

    The same configuration is reused for both:
    - document ingestion
    - query retrieval
    """

    embeddings = get_embeddings()

    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY,
    )
