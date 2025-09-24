# tests/conftest.py
import os
import sys
import json
import pathlib
from datetime import datetime

# --- import shim so `import <your modules>` works from project root ---
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# ---------------------------------------------------------------------

# Where to write logs (env var wins)
DEFAULT_LOG_PATH = pathlib.Path("reports/custom_test_log.txt")
LOG_FILE = pathlib.Path(os.getenv("TEST_LOG_PATH", str(DEFAULT_LOG_PATH)))

def _append(msg: str) -> None:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(msg)

def _fmt(label: str, val) -> str:
    try:
        if isinstance(val, (dict, list, tuple)):
            return f"{label} {json.dumps(val, ensure_ascii=False, indent=2)}"
        return f"{label} {val}"
    except Exception:
        return f"{label} {val}"

def pytest_sessionstart(session):
    # Fresh header per run
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    LOG_FILE.write_text(
        f"=== PYTEST RUN START: {datetime.now().isoformat(timespec='seconds')} ===\n",
        encoding="utf-8",
    )

# ---------- GLOBAL LOGGER HOOK ----------
import pytest

@pytest.hookimpl(hookwrapper=True, tryfirst=True)
def pytest_runtest_makereport(item, call):
    """
    Single place to format and store results for every test.
    Uses your requested format and enriches it with markers, params, duration,
    and captured stdout/stderr. If a test calls `record_property("inputs"/"expected"/"actual", ...)`,
    those values are included automatically.
    """
    outcome = yield
    report = outcome.get_result()  # TestReport

    # Only log after the test body finishes
    if report.when != "call":
        return

    # Basic metadata
    try:
        name = item.name
        desc = (item.function.__doc__ or "No description provided").strip()
    except Exception:
        name = getattr(item, "name", "<unknown>")
        desc = "No description provided"

    # Parametrized inputs (fallback if no explicit inputs recorded)
    params = {}
    callspec = getattr(item, "callspec", None)
    if callspec:
        params = callspec.params

    # Markers and status
    markers = sorted(set(m.name for m in item.iter_markers()))
    passed = (report.outcome == "passed")
    duration = getattr(report, "duration", 0.0)

    # Captured output (fallback if no 'actual' provided)
    capstdout = getattr(report, "capstdout", "") or ""
    capstderr = getattr(report, "capstderr", "") or ""

    # Failure details (trimmed)
    longrepr = ""
    if not passed and getattr(report, "longrepr", None):
        longrepr = str(report.longrepr)
        if len(longrepr) > 4000:
            longrepr = longrepr[:4000] + " ... (truncated)"

    # Pull custom fields recorded by tests via `record_property`
    # report.user_properties is a list of (name, value) pairs
    user_props = dict(getattr(report, "user_properties", []) or [])
    expected = user_props.get("expected")
    actual = user_props.get("actual")
    inputs_detail = user_props.get("inputs")

    # Build the lines in your requested style
    lines = []
    lines.append(f"\n=== TEST CASE: {name} ===")
    lines.append(_fmt("• What is being tested:", desc))

    if inputs_detail is not None:
        lines.append(_fmt("• Input provided:", inputs_detail))
    elif params:
        lines.append(_fmt("• Input provided (params):", params))

    if expected is not None:
        lines.append(_fmt("• Expected output:", expected))
    else:
        lines.append("• Expected output: See assertions in the test")

    if actual is not None:
        lines.append(_fmt("• Actual output:  ", actual))
    else:
        lines.append(_fmt("• Actual output (captured stdout):", capstdout.strip() or "(none)"))

    # Extra diagnostic context (markers, duration, stderr, failures)
    lines.append(_fmt("• Markers:", markers))
    lines.append(f"• Duration: {duration:.4f}s")
    if capstderr.strip():
        lines.append(_fmt("• Captured stderr:", capstderr.strip()))
    lines.append(f"• Status: {'PASSED' if passed else 'FAILED'}")
    if longrepr:
        lines.append(_fmt("• Failure details:", longrepr))

    _append("\n".join(lines) + "\n")

# ---------- OPTIONAL: helper you can call inside tests ----------
def log_test_detail(name, description, inputs, expected, actual, passed):
    """
    Manual logger for special cases where you want explicit control
    (in addition to the automatic hook). Writes to log and prints to stdout.
    """
    msg = (
        f"\n=== TEST CASE: {name} ===\n"
        f"• What is being tested: {description}\n"
        f"• Input provided: {inputs}\n"
        f"• Expected output: {expected}\n"
        f"• Actual output:   {actual}\n"
        f"• Status: {'PASSED' if passed else 'FAILED'}\n"
    )
    print(msg)
    _append(msg)

# ---------- OPTIONAL: end-of-run summary ----------
def pytest_terminal_summary(terminalreporter, exitstatus):
    stats = terminalreporter.stats
    total = terminalreporter._numcollected
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    skipped = len(stats.get("skipped", []))
    xfailed = len(stats.get("xfailed", []))
    xpassed = len(stats.get("xpassed", []))

    summary = (
        f"\n=== PYTEST RUN END ===\n"
        f"Collected: {total}\n"
        f"Passed:    {passed}\n"
        f"Failed:    {failed}\n"
        f"Skipped:   {skipped}\n"
        f"XFailed:   {xfailed}\n"
        f"XPassed:   {xpassed}\n"
        f"Exit code: {exitstatus}\n"
    )
    _append(summary)
