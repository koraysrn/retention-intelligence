"""Frontend build presence test.

This is skipped only because `frontend/dist` is not built in headless test
runs. It is **not** a data-related skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest

DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"


@pytest.mark.skipif(
    not DIST.exists(),
    reason="frontend/dist not built; this is not a data-related skip",
)
def test_frontend_dist_exists() -> None:
    assert (DIST / "index.html").exists()
