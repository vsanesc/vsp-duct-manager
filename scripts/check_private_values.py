# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Víctor Sánchez Escalante

"""Pre-commit hook: block commits whose staged additions contain values from duct_config.yaml.

duct_config.yaml is private and gitignored. This hook reads it (if present), collects every numeric and
string value that is not already public (trivial, or shared with duct_config.example.yaml), and scans the
added lines of the staged diff for them. It reports file:line and the config key that matched, never the
value itself and never the offending line. If duct_config.yaml does not exist (for example in someone
else's clone) it passes silently.

Bypass for a reviewed false positive: ``SKIP=check-private-values git commit ...``.
"""
import math
import re
import subprocess
import sys
from pathlib import Path

PRIVATE_NAME = "duct_config.yaml"
EXAMPLE_NAME = "duct_config.example.yaml"

TRIVIAL_NUMBERS = (0.0, 1.0, 2.0)  # compared on absolute value
MIN_STRING_LENGTH = 3
REL_TOL = 1e-9

# Number tokens not glued to identifiers or to longer numbers. Comma-grouped thousands are matched as one
# token and also split on the commas, so "[111,222]" yields 111 and 222 as well as 111222.
NUMBER_RE = re.compile(r"(?<![\w.])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][+-]?\d+)?(?!\w|\.\d)")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")
FILE_EXT_RE = re.compile(r"\.\w{1,5}$")


def flatten(node, prefix=""):
    """Yield (key path, scalar) for every leaf of a loaded YAML document."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from flatten(value, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(node, (list, tuple)):
        for i, value in enumerate(node):
            yield from flatten(value, f"{prefix}[{i}]")
    else:
        yield prefix, node


def _as_number(value):
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def collect_private_values(private: dict, example: dict | None) -> tuple[list, list]:
    """Return (numbers, strings) as lists of (value, key path), excluding public and trivial values."""
    public_numbers: list[float] = []
    public_strings: set[str] = set()
    for _, value in flatten(example or {}):
        number = _as_number(value)
        if number is not None:
            public_numbers.append(abs(number))
        elif isinstance(value, str):
            public_strings.add(value.strip().lower())

    def is_public_number(x: float) -> bool:
        return any(math.isclose(x, p, rel_tol=REL_TOL, abs_tol=1e-12) for p in public_numbers)

    numbers: list[tuple[float, str]] = []
    strings: list[tuple[str, str]] = []
    for key, value in flatten(private):
        number = _as_number(value)
        if number is None and isinstance(value, str):
            try:  # a number written as a string still counts as a number
                number = float(value.replace(",", ""))
            except ValueError:
                number = None
        if number is not None:
            x = abs(number)
            if not math.isfinite(x) or any(math.isclose(x, t, abs_tol=1e-12) for t in TRIVIAL_NUMBERS):
                continue
            if is_public_number(x):
                continue
            numbers.append((x, key))
        elif isinstance(value, str):
            text = value.strip().lower()
            if len(text) < MIN_STRING_LENGTH or text in public_strings:
                continue
            variants = {text}
            if FILE_EXT_RE.search(text):  # file names also leak through their stem
                variants.add(FILE_EXT_RE.sub("", text))
            for variant in variants:
                if len(variant) >= MIN_STRING_LENGTH and variant not in public_strings:
                    strings.append((variant, key))
    return numbers, strings


def number_tokens(line: str):
    """Absolute values of every numeric literal in a line."""
    for match in NUMBER_RE.finditer(line):
        token = match.group(0)
        yield float(token.replace(",", ""))
        if "," in token:
            for part in token.split(","):
                yield float(part)


def find_hits(line: str, numbers: list, strings: list) -> set[str]:
    """Key paths of the private values present in a line."""
    hits: set[str] = set()
    tokens = list(number_tokens(line))
    for value, key in numbers:
        if any(math.isclose(t, value, rel_tol=REL_TOL, abs_tol=1e-12) for t in tokens):
            hits.add(key)
    lowered = line.lower()
    for text, key in strings:
        if text in lowered:
            hits.add(key)
    return hits


def added_lines(diff: str):
    """Yield (path, line number, text) for every added line of a ``git diff -U0`` output."""
    path = None
    lineno = 0
    for raw in diff.splitlines():
        if raw.startswith("+++ "):
            target = raw[4:]
            path = None if target == "/dev/null" else target[2:] if target.startswith("b/") else target
        elif raw.startswith("@@"):
            match = HUNK_RE.match(raw)
            lineno = int(match.group(1)) if match else 0
        elif raw.startswith("+") and path is not None:
            yield path, lineno, raw[1:]
            lineno += 1


def staged_diff(root: Path) -> str:
    result = subprocess.run(
        ["git", "-c", "core.quotepath=false", "diff", "--cached", "-U0", "--no-color",
         "--no-ext-diff", "--no-renames", "--diff-filter=ACMR"],
        cwd=root,
        capture_output=True,
        check=True,
    )
    return result.stdout.decode("utf-8", errors="replace")


def git_toplevel() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, check=True)
    return Path(out.stdout.decode("utf-8", errors="replace").strip())


def load_yaml(path: Path):
    import yaml

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> int:
    root = git_toplevel()
    private_path = root / PRIVATE_NAME
    if not private_path.is_file():
        return 0  # nothing private to protect here

    try:
        private = load_yaml(private_path) or {}
        example_path = root / EXAMPLE_NAME
        example = load_yaml(example_path) if example_path.is_file() else None
    except Exception:  # never echo parser output: it can quote the file
        print(f"check-private-values: cannot parse {PRIVATE_NAME} or {EXAMPLE_NAME}; fix it so the scan can run")
        return 1

    numbers, strings = collect_private_values(private, example)
    findings: list[tuple[str, int, str]] = []
    for path, lineno, text in added_lines(staged_diff(root)):
        if Path(path).name == PRIVATE_NAME:
            findings.append((path, lineno, "the private config itself is staged"))
            continue
        for key in sorted(find_hits(text, numbers, strings)):
            findings.append((path, lineno, f"matches a value from {PRIVATE_NAME} (key: {key})"))

    if not findings:
        return 0
    print(f"check-private-values: staged additions contain private values from {PRIVATE_NAME}:")
    for path, lineno, why in sorted(set(findings)):
        print(f"  {path}:{lineno}: {why}")
    print("Remove or replace them (use neutral values), unstage the file, and commit again.")
    print("Reviewed false positive? SKIP=check-private-values git commit ...")
    return 1


if __name__ == "__main__":
    sys.exit(main())
