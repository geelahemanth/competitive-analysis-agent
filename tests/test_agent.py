from importlib import import_module
from unittest.mock import Mock, patch

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, ToolMessage


@pytest.fixture
def traced_agent(monkeypatch):
    # Build the real graph without creating an OpenAI client.
    with patch("services.llm.get_llm"):
        graph = import_module("agent.graph")

    agent = import_module("agent.agent")
    model = Mock()
    monkeypatch.setattr(agent, "llm_with_tools", model)
    return graph, model


def test_trace_prints_text_response_and_stops(traced_agent, capsys):
    graph, model = traced_agent
    model.invoke.return_value = AIMessage(
        content=[{"type": "text", "text": "Final comparison."}]
    )

    graph.run_with_trace("Compare the competitors.")

    output = capsys.readouterr().out
    assert output.rstrip().endswith("Final comparison.")
    assert "'type': 'text'" not in output
    assert model.invoke.call_count == 1


def test_trace_executes_tool_and_returns_result_to_agent(
    traced_agent, monkeypatch, capsys
):
    graph, model = traced_agent
    tools = import_module("tools.competitor_tools")
    lookup = Mock(return_value=[Document(
        page_content="NovaTech offers affordable devices.",
        metadata={"competitor": "NovaTech", "source": "test dataset"},
    )])
    monkeypatch.setattr(tools, "search_competitor", lookup)
    model.invoke.side_effect = [
        AIMessage(content="", tool_calls=[{
            "name": "get_competitor_profile",
            "args": {"competitor_name": "NovaTech"},
            "id": "profile_call",
            "type": "tool_call",
        }]),
        AIMessage(content="NovaTech competes on affordability."),
    ]

    graph.run_with_trace("Describe NovaTech.")

    lookup.assert_called_once_with(competitor_name="NovaTech")
    assert model.invoke.call_count == 2
    messages = model.invoke.call_args.args[0]
    tool_result = next(m for m in messages if isinstance(m, ToolMessage))
    assert tool_result.tool_call_id == "profile_call"
    assert "NovaTech offers affordable devices." in tool_result.content
    output = capsys.readouterr().out
    assert "[AGENT ACTION]" in output
    assert "[TOOL RESULT]" in output
    assert output.rstrip().endswith("NovaTech competes on affordability.")
