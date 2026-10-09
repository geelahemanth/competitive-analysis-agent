from typing import List

from langchain_core.documents import Document
from langchain_core.tools import tool
from pydantic import BaseModel, Field

from rag.retriever import (
    semantic_search,
    search_competitor,
)


class SearchCompetitorsInput(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description=(
            "Natural-language description of the competitor "
            "information to search for."
        ),
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=5,
        description=(
            "Maximum number of relevant competitor documents "
            "to retrieve."
        ),
    )


class CompetitorProfileInput(BaseModel):
    competitor_name: str = Field(
        ...,
        min_length=1,
        description=(
            "Exact competitor name whose profile should be retrieved."
        ),
    )


def format_documents(
    documents: List[Document],
) -> str:
    """
    Convert retrieved LangChain Documents into readable
    evidence for the LLM agent.
    """

    if not documents:
        return (
            "No relevant competitor information was found."
        )

    formatted_documents = []

    for index, document in enumerate(
        documents,
        start=1,
    ):
        competitor = document.metadata.get(
            "competitor",
            "Unknown competitor",
        )

        source = document.metadata.get(
            "source",
            "Unknown source",
        )

        formatted_documents.append(
            f"""
Evidence {index}
Competitor: {competitor}
Source: {source}

{document.page_content}
""".strip()
        )

    return "\n\n---\n\n".join(
        formatted_documents
    )


@tool(args_schema=SearchCompetitorsInput)
def search_competitors(
    query: str,
    top_k: int = 3,
) -> str:
    """
    Search across competitor knowledge using semantic similarity.

    Use this tool when you need to discover competitors or find
    companies matching a business concept, target market, product
    type, marketing strategy, pricing strategy, strength, weakness,
    financial characteristic, or other competitive attribute.

    Examples:
    - competitors targeting students
    - companies focused on premium laptops
    - competitors using influencer marketing
    - companies with strong AI positioning
    """

    documents = semantic_search(
        query=query,
        top_k=top_k,
    )

    return format_documents(
        documents
    )


@tool(args_schema=CompetitorProfileInput)
def get_competitor_profile(
    competitor_name: str,
) -> str:
    """
    Retrieve detailed information about one specific competitor.

    Use this tool when the competitor name is already known and
    you need its product description, target market, pricing
    strategy, marketing strategy, strengths, weaknesses,
    financial summary, recent initiatives, or region.
    """

    documents = search_competitor(
        competitor_name=competitor_name,
    )

    return format_documents(
        documents
    )


COMPETITOR_TOOLS = [
    search_competitors,
    get_competitor_profile,
]




if __name__ == "__main__":

    print("\n--- Search Competitors Tool ---\n")

    result = search_competitors.invoke(
        {
            "query": (
                "competitors targeting students "
                "with affordable laptops"
            ),
            "top_k": 3,
        }
    )

    print(result)

    print("\n--- Competitor Profile Tool ---\n")

    result = get_competitor_profile.invoke(
        {
            "competitor_name": "NovaTech"
        }
    )

    print(result)