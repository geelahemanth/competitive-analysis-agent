from pathlib import Path
from threading import Lock

from langchain_chroma import Chroma

from config.settings import (
    CHROMA_COLLECTION_NAME,
    CHROMA_PERSIST_DIRECTORY,
)
from services.embeddings import get_embeddings


_vector_store = None
_vector_store_lock = Lock()


def get_vector_store() -> Chroma:
    """
    Return one shared Chroma vector-store instance.

    This prevents multiple agent tool calls from trying
    to initialize the same PersistentClient concurrently.
    """

    global _vector_store

    if _vector_store is not None:
        return _vector_store

    with _vector_store_lock:

        # Another thread may have created it
        # while this thread was waiting.
        if _vector_store is not None:
            return _vector_store

        persist_path = Path(
            CHROMA_PERSIST_DIRECTORY
        ).resolve()

        persist_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        _vector_store = Chroma(
            collection_name=CHROMA_COLLECTION_NAME,
            embedding_function=get_embeddings(),
            persist_directory=str(persist_path),
        )

    return _vector_store