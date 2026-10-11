from langchain_core.messages import (
    AIMessage,
    ToolMessage,
)
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.agent import agent_node
from agent.state import AgentState
from tools.competitor_tools import COMPETITOR_TOOLS


def extract_reasoning_summaries(message: AIMessage) -> list[str]:
    """
    Extract OpenAI reasoning summaries from an AIMessage.

    With output_version='responses/v1', reasoning summaries appear
    inside message.content.
    """
    summaries = []

    if not isinstance(message.content, list):
        return summaries

    for block in message.content:
        if not isinstance(block, dict):
            continue

        if block.get("type") != "reasoning":
            continue

        reasoning_summaries = block.get("summary", [])
        if not isinstance(reasoning_summaries, list):
            continue

        for summary in reasoning_summaries:
            if not isinstance(summary, dict):
                continue

            text = summary.get("text")
            if isinstance(text, str) and text.strip():
                summaries.append(text)

    return summaries


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(COMPETITOR_TOOLS))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

competitive_agent_graph = graph_builder.compile()


def run_with_trace(
    query: str,
) -> None:
    """
    Run the autonomous agent and display:
    - reasoning summaries
    - agent tool decisions
    - tool arguments
    - tool observations
    - final answer
    """
    inputs = {
        "messages": [
            {
                "role": "user",
                "content": query,
            }
        ]
    }
    final_response = None

    print("\n===================================")
    print("AUTONOMOUS AGENT EXECUTION")
    print("===================================")

    print("\n[USER]")
    print(query)

    for update in competitive_agent_graph.stream(
        inputs,
        config={
            "recursion_limit": 20
        },
        stream_mode="updates",
    ):

        # =================================
        # AGENT NODE
        # =================================
        if "agent" in update:
            messages = update["agent"].get(
                "messages",
                [],
            )

            if not messages:
                continue

            message = messages[-1]

            if not isinstance(
                message,
                AIMessage,
            ):
                continue

            # -----------------------------
            # REASONING SUMMARIES
            # -----------------------------
            reasoning_summaries = extract_reasoning_summaries(message)
            if reasoning_summaries:
                print("\n[REASONING SUMMARY]")
                for summary in reasoning_summaries:
                    print(summary)

            # -----------------------------
            # TOOL DECISIONS OR FINAL ANSWER
            # -----------------------------
            if message.tool_calls:
                for tool_call in message.tool_calls:
                    tool_name = tool_call.get("name", "unknown_tool")
                    tool_args = tool_call.get("args", {})

                    print("\n[AGENT ACTION]")
                    print(f"Tool: {tool_name}")
                    print(f"Arguments: {tool_args}")

            elif message.text:
                print("\n[AGENT STATUS]")
                print("No further tool call requested.")
                print("Preparing final answer...")
                final_response = message.text

        # =================================
        # TOOL NODE
        # =================================
        if "tools" in update:
            messages = update["tools"].get(
                "messages",
                [],
            )

            for message in messages:
                if not isinstance(message, ToolMessage):
                    continue

                tool_name = getattr(message, "name", None)
                print("\n[TOOL RESULT]")
                if tool_name:
                    print(f"Tool: {tool_name}")

                content = str(message.content)
                # Keep terminal output readable
                max_length = 1000
                if len(content) > max_length:
                    content = content[:max_length] + "\n...[truncated]"
                print(content)

    print("\n===================================")
    print("FINAL RESPONSE")
    print("===================================\n")

    print(final_response)


if __name__ == "__main__":
    run_with_trace(
        """Analyze NovaTech's competitive position.

From the available competitors, determine which
company poses the greatest strategic threat to
NovaTech over the next three years.

Consider:
- customer overlap
- pricing overlap
- geographic overlap
- growth
- brand differentiation
- ability to enter NovaTech's student segment

Do not simply choose the largest company.

Gather whatever competitor information you need,
compare the strongest candidates, explain conflicting
evidence, and identify what additional data would be
needed for a high-confidence decision."""
    )
