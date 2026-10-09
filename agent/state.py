from langgraph.graph import MessagesState


class AgentState(MessagesState):
    """
    Shared state for the competitive analysis agent.

    MessagesState maintains the conversation and tool execution
    history across LangGraph nodes.
    """

    pass