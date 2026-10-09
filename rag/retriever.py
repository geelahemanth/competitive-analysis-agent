from typing import Dict, List, Optional

from langchain_core.documents import Document

from config.settings import TOP_K
from rag.vector_store import get_vector_store


def semantic_search(
    query: str,
    top_k: int = TOP_K,
) -> List[Document]:
    """
    Perform semantic similarity search across all competitor documents.

    Args:
        query: Natural-language search query.
        top_k: Maximum number of documents to return.

    Returns:
        List of relevant LangChain Documents.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query=query,
        k=top_k,
    )

    return documents


def search_competitor(
    competitor_name: str,
) -> List[Document]:
    """
    Retrieve documents for one exact competitor using metadata filtering.

    Args:
        competitor_name: Exact competitor name stored in metadata.

    Returns:
        Documents belonging to that competitor.
    """

    if not competitor_name or not competitor_name.strip():
        raise ValueError(
            "Competitor name cannot be empty."
        )

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query=competitor_name,
        k=TOP_K,
        filter={
            "competitor": competitor_name.strip()
        },
    )

    return documents


def search_with_filter(
    query: str,
    metadata_filter: Dict[str, str],
    top_k: int = TOP_K,
) -> List[Document]:
    """
    Perform semantic search while restricting results
    using Chroma metadata filters.

    Example:

        search_with_filter(
            query="student focused products",
            metadata_filter={
                "industry": "Consumer Electronics"
            }
        )

    Args:
        query:
            Natural-language semantic query.

        metadata_filter:
            Chroma metadata filter.

        top_k:
            Maximum number of documents to return.

    Returns:
        Matching Documents.
    """

    if not query or not query.strip():
        raise ValueError(
            "Search query cannot be empty."
        )

    if not metadata_filter:
        raise ValueError(
            "Metadata filter cannot be empty."
        )

    vector_store = get_vector_store()

    documents = vector_store.similarity_search(
        query=query,
        k=top_k,
        filter=metadata_filter,
    )

    return documents




if __name__ == "__main__":

    print("\n--- Semantic Search ---")

    results = semantic_search(
        "premium laptops for creators",
        top_k=3,
    )

    for index, document in enumerate(
        results,
        start=1,
    ):
        print(
            f"\nResult {index}: "
            f"{document.metadata.get('competitor')}"
        )

        print(document.page_content[:300])

    print("\n--- Competitor Search ---")

    results = search_competitor(
        "NovaTech"
    )

    for document in results:
        print(
            document.metadata.get("competitor")
        )