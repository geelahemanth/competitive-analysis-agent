import re
from typing import List

from langchain_core.documents import Document

from ingestion.load_data import load_competitor_data
from ingestion.preprocess import preprocess_competitor_data
from rag.vector_store import get_vector_store


def create_document_id(competitor_name: str) -> str:
    """
    Create a stable document ID from the competitor name.

    Example:
        "Apex Digital" -> "apex-digital"
        "NovaTech"     -> "novatech"
    """

    document_id = competitor_name.strip().lower()

    document_id = re.sub(
        r"[^a-z0-9]+",
        "-",
        document_id,
    )

    return document_id.strip("-")


def generate_document_ids(
    documents: List[Document],
) -> List[str]:
    """
    Generate stable IDs for all competitor documents.
    """

    document_ids = [
        create_document_id(
            document.metadata["competitor"]
        )
        for document in documents
    ]

    if len(document_ids) != len(set(document_ids)):
        raise ValueError(
            "Duplicate competitor names found in dataset."
        )

    return document_ids


def build_index() -> None:
    """
    Build or update the ChromaDB competitor index.
    """

    print("Loading competitor dataset...")

    dataframe = load_competitor_data()

    print(
        f"Loaded {len(dataframe)} competitor records."
    )

    print("Preprocessing competitor data...")

    documents = preprocess_competitor_data(
        dataframe
    )

    print(
        f"Created {len(documents)} documents."
    )

    document_ids = generate_document_ids(
        documents
    )

    print("Connecting to ChromaDB...")

    vector_store = get_vector_store()

    existing_documents = vector_store.get_by_ids(
        document_ids
    )

    existing_ids = {
        document.id
        for document in existing_documents
        if document.id
    }

    documents_to_add = []
    ids_to_add = []

    documents_to_update = []
    ids_to_update = []

    for document, document_id in zip(
        documents,
        document_ids,
    ):

        if document_id in existing_ids:
            documents_to_update.append(document)
            ids_to_update.append(document_id)

        else:
            documents_to_add.append(document)
            ids_to_add.append(document_id)

    if documents_to_add:

        print(
            f"Adding {len(documents_to_add)} "
            "new documents..."
        )

        vector_store.add_documents(
            documents=documents_to_add,
            ids=ids_to_add,
        )

    if documents_to_update:

        print(
            f"Updating {len(documents_to_update)} "
            "existing documents..."
        )

        vector_store.update_documents(
            ids=ids_to_update,
            documents=documents_to_update,
        )

    print()
    print("Index build completed successfully.")

    print(
        f"Added: {len(documents_to_add)}"
    )

    print(
        f"Updated: {len(documents_to_update)}"
    )


if __name__ == "__main__":
    build_index()