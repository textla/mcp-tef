"""Unit tests for JUnit XML report writing."""

from xml.etree import ElementTree as ET  # nosec B405

from mcp_tef_cli.junit import JUnitCase, write_junit


def test_write_junit_counts_and_outcomes(tmp_path):
    """Each outcome gets its element and the suite counts them."""
    path = tmp_path / "nested" / "report.xml"
    write_junit(
        path,
        "suite",
        [
            JUnitCase(name="ok", classname="c", time_seconds=1.5, details="out"),
            JUnitCase(name="bad", classname="c", failure="FN", details="why"),
            JUnitCase(name="broke", classname="c", error="boom"),
            JUnitCase(name="later", classname="c", skipped="not run", details="query"),
            JUnitCase(name="props", classname="c", properties={"clarity": "8"}),
        ],
    )

    root = ET.parse(path).getroot()  # nosec B314
    suite = root.find("testsuite")
    assert suite is not None
    assert suite.attrib["name"] == "suite"
    assert (suite.attrib["tests"], suite.attrib["failures"]) == ("5", "1")
    assert (suite.attrib["errors"], suite.attrib["skipped"]) == ("1", "1")
    cases = {tc.attrib["name"]: tc for tc in suite.findall("testcase")}
    assert cases["ok"].attrib["time"] == "1.500"
    assert cases["ok"].findtext("system-out") == "out"
    failure = cases["bad"].find("failure")
    assert failure is not None
    assert (failure.attrib["message"], failure.text) == ("FN", "why")
    assert cases["broke"].find("error") is not None
    assert cases["later"].find("skipped") is not None
    assert cases["later"].findtext("system-out") == "query"
    prop = cases["props"].find("properties/property")
    assert prop is not None
    assert prop.attrib == {"name": "clarity", "value": "8"}


def test_write_junit_replaces_characters_xml_cannot_hold(tmp_path):
    """Control characters in LLM output don't make the report unreadable."""
    path = tmp_path / "report.xml"
    write_junit(path, "s", [JUnitCase(name="a\x00b", classname="c", failure="x\x1by")])

    tc = ET.parse(path).getroot().find("testsuite/testcase")  # nosec B314
    assert tc is not None
    assert tc.attrib["name"] == "a\ufffdb"
