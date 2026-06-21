#!/usr/bin/env python3
"""Fail on common secret files and project-specific infrastructure leaks."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
IGNORED_PARTS = {".git", ".venv", "data", "runs", "__pycache__"}
PATTERNS = {
    "GitHub token": re.compile(r"gh[pousr]_[A-Za-z0-9_]{30,}"),
    "private key": re.compile(r"-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----"),
    "non-loopback IPv4": re.compile(r"(?<![\d.])(?!(?:127|0)\.)(?:\d{1,3}\.){3}\d{1,3}(?![\d.])"),
}


def main() -> None:
    tracked = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).splitlines()
    findings = []
    for relative in tracked:
        if relative == "scripts/security_check.py":
            continue
        path = ROOT / relative
        if not path.is_file() or any(part in IGNORED_PARTS for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                findings.append(f"{relative}:{text.count(chr(10), 0, match.start()) + 1}: {name}")
    if findings:
        raise SystemExit("Potential secrets:\n" + "\n".join(findings))
    print(f"OK: scanned {len(tracked)} tracked files")


if __name__ == "__main__":
    main()
