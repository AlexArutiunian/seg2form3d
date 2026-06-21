#!/usr/bin/env python3
"""Minimal fallback runner for offline lab machines without pytest."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    passed = 0
    for path in sorted((ROOT / "tests").glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[path.stem] = module
        spec.loader.exec_module(module)
        for name in sorted(dir(module)):
            function = getattr(module, name)
            if name.startswith("test_") and callable(function):
                function()
                passed += 1
                print(f"PASS {path.name}::{name}")
    print(f"{passed} passed")


if __name__ == "__main__":
    main()
