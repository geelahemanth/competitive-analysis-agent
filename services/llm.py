from langchain_openai import ChatOpenAI

from config.settings import (
    OPENAI_CHAT_MODEL,
    OPENAI_REASONING_EFFORT,
)


def get_llm() -> ChatOpenAI:
    """
    OpenAI reasoning model used by the autonomous agent.
    """

    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,

        reasoning={
            "effort": OPENAI_REASONING_EFFORT,
            "summary": "auto",
        },

        use_responses_api=True,
        output_version="responses/v1",

        timeout=60,
        max_retries=2,
    )