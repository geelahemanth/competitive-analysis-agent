import re
from typing import List

import pandas as pd
from langchain_core.documents import Document


REQUIRED_COLUMNS = [
    "Competitor Name",
    "Industry",
    "Product Description",
    "Target Market",
    "Pricing Strategy",
    "Marketing Strategy",
    "Key Strengths",
    "Key Weaknesses",
    "Financial Summary",
    "Recent Initiative",
    "Primary Region",
]


def clean_text(value) -> str:
    """
    Clean individual text values.

    - Converts missing values to empty strings
    - Removes excessive whitespace
    - Removes line breaks and tabs
    - Keeps meaningful punctuation and symbols
    """

    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Replace tabs/newlines/multiple spaces with one space
    text = re.sub(r"\s+", " ", text)

    return text


def validate_columns(dataframe: pd.DataFrame) -> None:
    """
    Ensure that all required columns exist in the dataset.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )


def preprocess_competitor_data(
    dataframe: pd.DataFrame,
) -> List[Document]:
    """
    Convert competitor DataFrame rows into LangChain Documents.

    Each competitor becomes one Document containing the main
    business information and useful metadata.
    """

    validate_columns(dataframe)

    documents = []

    for _, row in dataframe.iterrows():

        competitor_name = clean_text(row["Competitor Name"])
        industry = clean_text(row["Industry"])
        product_description = clean_text(row["Product Description"])
        target_market = clean_text(row["Target Market"])
        pricing_strategy = clean_text(row["Pricing Strategy"])
        marketing_strategy = clean_text(row["Marketing Strategy"])
        strengths = clean_text(row["Key Strengths"])
        weaknesses = clean_text(row["Key Weaknesses"])
        financial_summary = clean_text(row["Financial Summary"])
        recent_initiative = clean_text(row["Recent Initiative"])
        primary_region = clean_text(row["Primary Region"])

        page_content = f"""
Competitor: {competitor_name}

Industry:
{industry}

Product Description:
{product_description}

Target Market:
{target_market}

Pricing Strategy:
{pricing_strategy}

Marketing Strategy:
{marketing_strategy}

Key Strengths:
{strengths}

Key Weaknesses:
{weaknesses}

Financial Summary:
{financial_summary}

Recent Initiative:
{recent_initiative}

Primary Region:
{primary_region}
""".strip()

        metadata = {
            "competitor": competitor_name,
            "industry": industry,
            "region": primary_region,
            "source": "competitor_analysis_dataset.csv",
        }

        document = Document(
            page_content=page_content,
            metadata=metadata,
        )

        documents.append(document)

    return documents