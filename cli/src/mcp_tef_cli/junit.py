"""JUnit XML reports for mcp-tef CLI commands."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree as ET  # nosec B405

__all__ = ["JUnitCase", "write_junit"]

# Characters XML 1.0 can't hold; LLM output sometimes carries them.
_INVALID_XML_CHARS = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\ufffe\uffff]")


@dataclass
class JUnitCase:
    """One <testcase>.

    At most one of failure, error and skipped is set; each is the element's
    message. details is that element's text, or <system-out> for a case that
    passed.
    """

    name: str
    classname: str
    time_seconds: float = 0.0
    failure: str | None = None
    error: str | None = None
    skipped: str | None = None
    details: str | None = None
    properties: dict[str, str] = field(default_factory=dict)


def _clean(value: object) -> str:
    return _INVALID_XML_CHARS.sub("\ufffd", str(value))


def write_junit(path: str | Path, suite_name: str, cases: list[JUnitCase]) -> None:
    """Write cases as one JUnit <testsuite> to path, creating its directory.

    Args:
        path: File to write
        suite_name: The <testsuite> name
        cases: Test cases, in report order
    """
    counts = {
        "tests": len(cases),
        "failures": sum(c.failure is not None for c in cases),
        "errors": sum(c.error is not None for c in cases),
        "skipped": sum(c.skipped is not None for c in cases),
    }
    total_time = f"{sum(c.time_seconds for c in cases):.3f}"
    root = ET.Element("testsuites", {k: str(v) for k, v in counts.items()}, time=total_time)
    suite = ET.SubElement(
        root,
        "testsuite",
        {k: str(v) for k, v in counts.items()},
        name=_clean(suite_name),
        time=total_time,
    )
    for case in cases:
        tc = ET.SubElement(
            suite,
            "testcase",
            name=_clean(case.name),
            classname=_clean(case.classname),
            time=f"{case.time_seconds:.3f}",
        )
        if case.properties:
            props = ET.SubElement(tc, "properties")
            for key, value in case.properties.items():
                ET.SubElement(props, "property", name=_clean(key), value=_clean(value))
        outcome = None
        for tag, message in (
            ("failure", case.failure),
            ("error", case.error),
            ("skipped", case.skipped),
        ):
            if message is not None:
                outcome = ET.SubElement(tc, tag, message=_clean(message))
                break
        if case.details:
            if outcome is not None and outcome.tag != "skipped":
                outcome.text = _clean(case.details)
            else:
                ET.SubElement(tc, "system-out").text = _clean(case.details)
    ET.indent(root)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out, encoding="utf-8", xml_declaration=True)
