"""Ensure the repo root is importable so tests can `import src.control...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
