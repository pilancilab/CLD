"""
Pytest configuration.

This repo isn't packaged (no pyproject.toml/setup.py), so when tests are run from
inside `tests/` (or any other cwd that isn't the repo root), `import cld` can fail.
Ensure the project root is on sys.path so the `cld/` package is importable.
"""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

