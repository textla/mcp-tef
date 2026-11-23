"""Integration tests for the test-report CLI command."""

import json
from xml.etree import ElementTree as ET  # nosec B405

import httpx
import pytest
import respx
from click.testing import CliRunner

from mcp_tef_cli.commands.test_report import test_report as report_cmd
from mcp_tef_cli.constants import (
    EXIT_INVALID_ARGUMENTS,
    EXIT_RESOURCE_NOT_FOUND,
    EXIT_SUCCESS,
)

pytestmark = [pytest.mark.integration]

TEF = "http://localhost:8000"
MCP = "http://localhost:3000/mcp"


def mock_case(case_id: str, name: str, tool: str | None) -> dict:
    """Return a TestCaseResponse JSON expecting tool, or no tool."""
    expected = [{"mcp_server_url": MCP, "tool_name": tool}] if tool else None
    return {
        "id": case_id,
        "name": name,
        "query": f"query for {name}",
        "expected_tool_calls": expected,
        "available_mcp_servers": [MCP],
        "created_at": "2025-01-01T00:00:00Z",
        "updated_at": "2025-01-01T00:00:00Z",
    }


def mock_run(case_id: str, classification: str | None, **overrides) -> dict:
    """Return a completed TestRunResponse JSON."""
    run = {
        "id": f"run-{case_id}",
        "test_case_id": case_id,
        "status": "completed",
        "classification": classification,
        "execution_time_ms": 1200,
        "tool_call_matches": [],
        "created_at": "2025-01-01T00:00:01Z",
    }
    run.update(overrides)
    return run


def mock_runs(*runs: dict) -> httpx.Response:
    """Return a PaginatedTestRunResponse."""
    return httpx.Response(200, json={"items": list(runs), "total": len(runs)})


def read_cases(path) -> dict[str, ET.Element]:
    """Parse the report and return its testcases by name."""
    root = ET.parse(path).getroot()  # nosec B314
    return {tc.attrib["name"]: tc for tc in root.iter("testcase")}


class TestTestReport:
    """Test the test-report command."""

    @respx.mock
    def test_reports_each_case_by_its_latest_run(self, tmp_path):
        """TP passes, FN fails, a failed run errors, no run is skipped."""
        cases = {
            "c1": mock_case("c1", "pass", "get_weather"),
            "c2": mock_case("c2", "miss", "get_weather"),
            "c3": mock_case("c3", "broke", None),
            "c4": mock_case("c4", "unrun", None),
        }
        for case_id, body in cases.items():
            respx.get(f"{TEF}/test-cases/{case_id}").mock(
                return_value=httpx.Response(200, json=body)
            )
        fn_match = {
            "match_type": "FN",
            "expected_tool_call": {"mcp_server_url": MCP, "tool_name": "get_weather"},
        }
        runs = {
            "c1": mock_runs(
                mock_run("c1", "FN", id="old", created_at="2024-01-01T00:00:00Z"),
                mock_run("c1", "TP"),
            ),
            "c2": mock_runs(mock_run("c2", "FN", tool_call_matches=[fn_match])),
            "c3": mock_runs(mock_run("c3", None, status="failed", error_message="timeout")),
            "c4": mock_runs(),
        }
        respx.get(f"{TEF}/test-runs").mock(
            side_effect=lambda request: runs[request.url.params["test_case_id"]]
        )
        output = tmp_path / "report.xml"

        args = ["--url", TEF, "--output", str(output)]
        for case_id in cases:
            args += ["--test-case-id", case_id]
        result = CliRunner().invoke(report_cmd, args)

        assert result.exit_code == EXIT_SUCCESS, result.output
        got = read_cases(output)
        assert got["pass"].find("failure") is None
        assert got["pass"].attrib["time"] == "1.200"
        failure = got["miss"].find("failure")
        assert failure is not None
        assert failure.attrib["message"] == "FN: expected get_weather, called no tool"
        assert "FN: expected get_weather, called -" in (failure.text or "")
        error = got["broke"].find("error")
        assert error is not None
        assert error.attrib["message"] == "timeout"
        skipped = got["unrun"].find("skipped")
        assert skipped is not None
        assert skipped.attrib["message"] == "no test run"

    @respx.mock
    def test_without_ids_reports_every_case_and_skips_unsent_file_cases(
        self, tmp_path, monkeypatch
    ):
        """No --test-case-id reports the server's cases; --test-file adds the rest."""
        monkeypatch.setenv("TEF_SERVER_URL", TEF)
        monkeypatch.setenv("MCP_SERVER_URL", MCP)
        respx.get(f"{TEF}/test-cases").mock(
            return_value=httpx.Response(
                200, json={"items": [mock_case("c1", "sent", None)], "total": 1}
            )
        )
        respx.get(f"{TEF}/test-runs").mock(return_value=mock_runs(mock_run("c1", "TN")))
        test_file = tmp_path / "tests.json"
        test_file.write_text(
            json.dumps(
                [
                    {"name": name, "query": "q", "available_mcp_servers": ["${MCP_SERVER_URL}"]}
                    for name in ("sent", "not sent")
                ]
            )
        )
        output = tmp_path / "report.xml"

        result = CliRunner().invoke(
            report_cmd, ["--test-file", str(test_file), "--output", str(output)]
        )

        assert result.exit_code == EXIT_SUCCESS, result.output
        got = read_cases(output)
        assert got["sent"].find("skipped") is None
        skipped = got["not sent"].find("skipped")
        assert skipped is not None
        assert skipped.attrib["message"] == "not run"

    @respx.mock
    def test_unknown_case_is_not_found(self, tmp_path):
        """A missing test case exits with the not-found code and writes nothing."""
        respx.get(f"{TEF}/test-cases/nope").mock(return_value=httpx.Response(404))
        output = tmp_path / "report.xml"

        result = CliRunner().invoke(
            report_cmd, ["--url", TEF, "--test-case-id", "nope", "--output", str(output)]
        )

        assert result.exit_code == EXIT_RESOURCE_NOT_FOUND
        assert not output.exists()

    def test_bad_test_file_is_invalid_arguments(self, tmp_path):
        """An unreadable --test-file fails before any request."""
        result = CliRunner().invoke(
            report_cmd,
            ["--url", TEF, "--test-file", str(tmp_path / "missing.json"), "--output", "x.xml"],
        )

        assert result.exit_code == EXIT_INVALID_ARGUMENTS


def test_failure_names_every_tool_called():
    """An FN with one of two expected calls matched still names the matched call."""
    from mcp_tef_models.schemas import TestCaseResponse, TestRunResponse

    from mcp_tef_cli.commands.test_report import junit_case

    tc = TestCaseResponse.model_validate(mock_case("c1", "two calls", "get_weather"))
    weather = {"mcp_server_url": MCP, "tool_name": "get_weather"}
    run = TestRunResponse.model_validate(
        mock_run(
            "c1",
            "FN",
            tool_call_matches=[
                {
                    "match_type": "TP",
                    "expected_tool_call": weather,
                    "actual_tool_call": {"name": "get_weather", "mcp_server_url": MCP},
                },
                {"match_type": "FN", "expected_tool_call": weather},
            ],
        )
    )

    case = junit_case(tc, run, "mcp-tef")

    assert case.failure == "FN: expected get_weather, called get_weather"
