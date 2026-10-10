from langchain_core.messages import SystemMessage

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from services.llm import get_llm
from tools.competitor_tools import COMPETITOR_TOOLS


# Base OpenAI reasoning model
llm = get_llm()


# Give the model access to our competitor tools
llm_with_tools = llm.bind_tools(
    COMPETITOR_TOOLS
)


def agent_node(
    state: AgentState,
) -> dict:
    """
    Main reasoning node for the autonomous competitive
    intelligence agent.

    The model receives:
    - system instructions
    - user messages
    - previous AI messages
    - previous tool results

    It then decides whether to:
    - call another tool
    - or return a final answer
    """

    messages = [
        SystemMessage(
            content=SYSTEM_PROMPT
        ),
        *state["messages"],
    ]

    response = llm_with_tools.invoke(
        messages
    )

    return {
        "messages": [response]
    }