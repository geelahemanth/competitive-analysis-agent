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
    final_output = output.split("FINAL RESPONSE", 1)[1]
    assert "'type': 'text'" not in final_output
    assert "[REASONING SUMMARY]" not in output
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
        AIMessage(content=[{
            "type": "reasoning",
            "summary": [{
                "type": "summary_text",
                "text": "Retrieve NovaTech's profile before comparing.",
            }],
        }], tool_calls=[{
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
    assert (
        "[REASONING SUMMARY]\nRetrieve NovaTech's profile before comparing."
        in output
    )
    assert output.index("[REASONING SUMMARY]") < output.index("[AGENT ACTION]")
    assert output.rstrip().endswith("NovaTech competes on affordability.")


@pytest.mark.parametrize("content, expected", [
    ("Plain text response.", []),
    ([{"type": "text", "text": "Final answer."}], []),
    ([
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "First summary."},
            {"type": "summary_text", "text": "Second summary."},
        ]},
        {"type": "text", "text": "Final answer."},
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "Third summary."},
        ]},
    ], ["First summary.", "Second summary.", "Third summary."]),
    ([
        "A string content block.",
        {},
        {"type": "reasoning"},
        {"type": "reasoning", "summary": None},
        {"type": "reasoning", "summary": "Not a list."},
        {"type": "text", "summary": [{"text": "Not a reasoning summary."}]},
        {"type": "reasoning", "summary": [
            None, "Not a summary object.", {},
            {"text": None}, {"text": 123}, {"text": ""}, {"text": "   "},
            {"type": "summary_text", "text": "Valid summary."},
        ]},
    ], ["Valid summary."]),
])
def test_extract_reasoning_summaries(traced_agent, content, expected):
    graph, _ = traced_agent
    assert graph.extract_reasoning_summaries(AIMessage(content=content)) == expected


def test_trace_prints_summary_and_final_text(traced_agent, capsys):
    graph, model = traced_agent
    model.invoke.return_value = AIMessage(content=[
        {"type": "reasoning", "summary": [
            {"type": "summary_text", "text": "Compared the target markets."},
        ]},
        {"type": "text", "text": "Company B has stronger premium positioning."},
    ])

    graph.run_with_trace("Compare the competitors.")

    output = capsys.readouterr().out
    assert "[REASONING SUMMARY]\nCompared the target markets." in output
    assert output.rstrip().endswith("Company B has stronger premium positioning.")
    assert model.invoke.call_count == 1
