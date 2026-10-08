from langchain_openai import OpenAIEmbeddings

from config.settings import OPENAI_EMBEDDING_MODEL


def get_embeddings() -> OpenAIEmbeddings:
    """
    Returns the OpenAI embedding model used for
    document indexing and query retrieval.
    """
    return OpenAIEmbeddings(
        model=OPENAI_EMBEDDING_MODEL
    )
