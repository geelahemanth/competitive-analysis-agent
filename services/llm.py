from langchain_openai import ChatOpenAI

from config.settings import (
    OPENAI_CHAT_MODEL,
    OPENAI_REASONING_EFFORT,
)


def get_llm() -> ChatOpenAI:
    """
    Create the OpenAI reasoning model used by the autonomous agent.
    """

    return ChatOpenAI(
        model=OPENAI_CHAT_MODEL,
        reasoning={
            "effort": OPENAI_REASONING_EFFORT,
        },
        use_responses_api=True,
        timeout=60,
        max_retries=2,
    )