from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent.agent import agent_node
from agent.state import AgentState
from tools.competitor_tools import COMPETITOR_TOOLS


graph_builder = StateGraph(AgentState)
graph_builder.add_node("agent", agent_node)
graph_builder.add_node("tools", ToolNode(COMPETITOR_TOOLS))
graph_builder.add_edge(START, "agent")
graph_builder.add_conditional_edges("agent", tools_condition)
graph_builder.add_edge("tools", "agent")

competitive_agent_graph = graph_builder.compile()


def run_with_trace(query: str) -> None:
    """
    Run the autonomous agent while printing its observable
    execution trajectory.

    Shows:
    - tool selected by the agent
    - tool arguments
    - tool results
    - when the agent stops using tools
    - final response
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

    print(f"\n[USER]\n{query}")

    for update in competitive_agent_graph.stream(
        inputs,
        config={
            "recursion_limit": 20
        },
        stream_mode="updates",
    ):

        # -------------------------
        # AGENT NODE
        # -------------------------
        if "agent" in update:

            messages = update["agent"].get(
                "messages",
                [],
            )

            if not messages:
                continue

            message = messages[-1]

            if isinstance(message, AIMessage):

                # Agent decided to call one or more tools
                if message.tool_calls:

                    for tool_call in message.tool_calls:

                        tool_name = tool_call.get(
                            "name",
                            "unknown_tool",
                        )

                        tool_args = tool_call.get(
                            "args",
                            {},
                        )

                        print(
                            "\n[AGENT ACTION]"
                        )

                        print(
                            f"Tool: {tool_name}"
                        )

                        print(
                            f"Arguments: {tool_args}"
                        )

                # No tool call = agent has decided to answer
                elif message.content:

                    print(
                        "\n[AGENT STATUS]"
                    )

                    print(
                        "No further tool call requested."
                    )

                    print(
                        "Preparing final answer..."
                    )

                    final_response = message.text

        # -------------------------
        # TOOL NODE
        # -------------------------
        if "tools" in update:

            messages = update["tools"].get(
                "messages",
                [],
            )

            for message in messages:

                if not isinstance(
                    message,
                    ToolMessage,
                ):
                    continue

                tool_name = getattr(
                    message,
                    "name",
                    None,
                )

                print(
                    "\n[TOOL RESULT]"
                )

                if tool_name:
                    print(
                        f"Tool: {tool_name}"
                    )

                content = str(
                    message.content
                )

                # Keep terminal output readable
                max_length = 1000

                if len(content) > max_length:
                    content = (
                        content[:max_length]
                        + "\n...[truncated]"
                    )

                print(content)

    print("\n===================================")
    print("FINAL RESPONSE")
    print("===================================\n")

    print(final_response)


if __name__ == "__main__":

    run_with_trace(
        (
            "Compare NovaTech and Apex Digital. "
            "Which company has the stronger "
            "positioning and why?"
        )
    )
