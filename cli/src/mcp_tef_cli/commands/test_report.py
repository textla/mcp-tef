"""Test report command: JUnit XML of test cases' latest runs."""

import asyncio

import click
from mcp_tef_models.schemas import TestCaseResponse, TestRunResponse

from mcp_tef_cli.client import ClientConfig, TefClient
from mcp_tef_cli.commands.test_case import load_test_cases_from_file
from mcp_tef_cli.constants import DEFAULT_CONTAINER_NAME, EXIT_INVALID_ARGUMENTS, EXIT_SUCCESS
from mcp_tef_cli.junit import JUnitCase, write_junit
from mcp_tef_cli.output import print_error, print_success
from mcp_tef_cli.utils import handle_api_errors, resolve_tef_url

DEFAULT_TIMEOUT = 120
PAGE_SIZE = 100
MAX_CONCURRENT_REQUESTS = 10


async def fetch_cases_and_runs(
    client: TefClient, test_case_ids: list[str]
) -> list[tuple[TestCaseResponse, TestRunResponse | None]]:
    """Fetch each test case and its latest test run.

    Args:
        client: mcp-tef client
        test_case_ids: Test cases to fetch; empty for every test case on the server

    Returns:
        (test case, latest run or None) pairs, in test_case_ids order or the server's
    """
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async def get_case(test_case_id: str) -> TestCaseResponse:
        async with semaphore:
            return await client.get_test_case(test_case_id)

    async def latest_run(test_case_id: str) -> TestRunResponse | None:
        async with semaphore:
            runs: list[TestRunResponse] = []
            while True:
                page = await client.list_test_runs(
                    test_case_id=test_case_id, offset=len(runs), limit=PAGE_SIZE
                )
                runs.extend(page.items)
                if not page.items or len(runs) >= page.total:
                    break
        return max(runs, key=lambda r: r.created_at, default=None)

    if test_case_ids:
        cases = list(await asyncio.gather(*(get_case(i) for i in test_case_ids)))
    else:
        cases = []
        while True:
            page = await client.list_test_cases(offset=len(cases), limit=PAGE_SIZE)
            cases.extend(page.items)
            if not page.items or len(cases) >= page.total:
                break
    runs = await asyncio.gather(*(latest_run(c.id) for c in cases))
    return list(zip(cases, runs, strict=True))


def describe_expected(tc: TestCaseResponse) -> str:
    """Name the tools a test case expects, or say it expects none."""
    if not tc.expected_tool_calls:
        return "no tool call"
    return ", ".join(call.tool_name for call in tc.expected_tool_calls)


def junit_case(tc: TestCaseResponse, run: TestRunResponse | None, classname: str) -> JUnitCase:
    """Turn a test case and its latest run into a <testcase>.

    A completed TP or TN run passes; FP or FN fails; a failed run is an error;
    a missing or unfinished run is skipped.
    """
    case = JUnitCase(name=tc.name, classname=classname)
    lines = [f"Query: {tc.query}", f"Expected: {describe_expected(tc)}"]
    case.properties["test_case_id"] = tc.id
    if run is None:
        case.skipped = "no test run"
        case.details = "\n".join(lines)
        return case

    case.properties["test_run_id"] = run.id
    if run.model_settings:
        case.properties["model"] = f"{run.model_settings.provider}/{run.model_settings.model}"
    if run.execution_time_ms is not None:
        case.time_seconds = run.execution_time_ms / 1000
    if run.classification:
        case.properties["classification"] = run.classification
    if run.avg_parameter_correctness is not None:
        case.properties["avg_parameter_correctness"] = f"{run.avg_parameter_correctness:.2f}"
    if run.confidence_score:
        case.properties["confidence_score"] = run.confidence_score

    for match in run.tool_call_matches:
        expected = match.expected_tool_call.tool_name if match.expected_tool_call else "-"
        actual = match.actual_tool_call.name if match.actual_tool_call else "-"
        line = f"{match.match_type}: expected {expected}, called {actual}"
        if match.parameter_correctness is not None:
            line += f" (parameters {match.parameter_correctness:.1f}/10)"
        lines.append(line)
    case.details = "\n".join(lines)

    if run.status == "failed":
        case.error = run.error_message or "test run failed"
    elif run.status != "completed":
        case.skipped = f"test run is {run.status}"
    elif run.classification in ("FP", "FN"):
        called = list(
            dict.fromkeys(
                m.actual_tool_call.name for m in run.tool_call_matches if m.actual_tool_call
            )
        )
        case.failure = (
            f"{run.classification}: expected {describe_expected(tc)}, "
            f"called {', '.join(called) or 'no tool'}"
        )
    return case


@click.command(name="test-report")
@click.option(
    "--test-case-id",
    "test_case_ids",
    multiple=True,
    help="Test case to report; repeat for more (default: every test case on the server)",
)
@click.option(
    "--test-file",
    type=click.Path(dir_okay=False),
    default=None,
    help=(
        "JSON file the test cases were created from (as 'test-case create --from-file' "
        "takes); a case in it with no reported test case is skipped as not run"
    ),
)
@click.option(
    "--output",
    required=True,
    type=click.Path(dir_okay=False),
    help="JUnit XML file to write",
)
@click.option(
    "--suite-name",
    default="mcp-tef",
    help="JUnit testsuite name and testcase classname (default: mcp-tef)",
)
@click.option(
    "--url",
    "tef_url",
    default=None,
    help="mcp-tef server URL (or set TEF_SERVER_URL; default: Docker container discovery)",
)
@click.option(
    "--container-name",
    default=None,
    help=f"Docker container name for URL discovery (default: {DEFAULT_CONTAINER_NAME})",
)
@click.option(
    "--timeout",
    type=int,
    default=DEFAULT_TIMEOUT,
    help=f"HTTP request timeout in seconds (default: {DEFAULT_TIMEOUT})",
)
@click.option(
    "--insecure",
    is_flag=True,
    help="Skip SSL certificate verification",
)
def test_report(
    test_case_ids: tuple[str, ...],
    test_file: str | None,
    output: str,
    suite_name: str,
    tef_url: str | None,
    container_name: str | None,
    timeout: int,
    insecure: bool,
) -> None:
    """Write a JUnit XML report of test cases and their latest test runs.

    Each test case is one <testcase>. A completed TP or TN run passes, FP or FN
    is a failure, a failed run is an error, and a case with no finished run is
    skipped. The command exits 0 once the report is written, whatever the
    results; read the report to gate on them.

    Examples:

      \b
      # Report two test cases
      mtef test-report --test-case-id a1b2c3d4-... --test-case-id b2c3d4e5-... \\
        --output test-report.xml

      \b
      # Report every test case, and skip the cases in the file that weren't created
      mtef test-report --test-file test-cases.json --output test-report.xml
    """
    file_names: list[str] = []
    if test_file:
        try:
            file_names = [tc.name for tc in load_test_cases_from_file(test_file)]
        except click.BadParameter as e:
            print_error("Test file error", str(e.message))
            raise SystemExit(EXIT_INVALID_ARGUMENTS) from e

    url = resolve_tef_url(tef_url, container_name, "table")
    config = ClientConfig(base_url=url, timeout=float(timeout), verify_ssl=not insecure)

    async def _fetch() -> list[tuple[TestCaseResponse, TestRunResponse | None]]:
        client = TefClient(config)
        try:
            return await fetch_cases_and_runs(client, list(test_case_ids))
        finally:
            await client.close()

    with handle_api_errors(url, "Test case", ", ".join(test_case_ids) or None):
        pairs = asyncio.run(_fetch())

    cases = [junit_case(tc, run, suite_name) for tc, run in pairs]
    reported = {tc.name for tc, _ in pairs}
    for name in file_names:
        if name not in reported:
            cases.append(JUnitCase(name=name, classname=suite_name, skipped="not run"))
            reported.add(name)

    write_junit(output, suite_name, cases)
    failed = sum(c.failure is not None for c in cases)
    errors = sum(c.error is not None for c in cases)
    skipped = sum(c.skipped is not None for c in cases)
    print_success(
        f"Wrote {output}: {len(cases)} test cases, {failed} failed, "
        f"{errors} errors, {skipped} skipped"
    )
    raise SystemExit(EXIT_SUCCESS)
