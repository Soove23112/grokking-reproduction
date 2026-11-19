import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from grokking.paths import AUTHORS_RUN  # noqa: E402

needs_authors_run = pytest.mark.skipif(not AUTHORS_RUN.exists(), reason="authors' trajectory file not downloaded")
