"""Pytest fixtures and test environment configuration."""

import os
from pathlib import Path

# Ensure tiktoken reads pre-cached offline encoding inside test runs
_cache_dir = Path(__file__).resolve().parent.parent / ".tiktoken_cache"
if _cache_dir.exists():
    os.environ["TIKTOKEN_CACHE_DIR"] = str(_cache_dir)
