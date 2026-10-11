from langchain_core.messages import SystemMessage

from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState
from services.llm import get_llm
from tools.competitor_tools import COMPETITOR_TOOLS


llm = get_llm()

llm_with_tools = (
    llm
    .bind_tools(COMPETITOR_TOOLS)
    .bind(parallel_tool_calls=False)
)


def agent_node(
    state: AgentState,
) -> dict:
    """
    Main reasoning node.
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