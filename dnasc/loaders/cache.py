"""
dnasc/loaders/cache.py
───────────────────────
Parquet-based caching for the pipeline baseline.
"""

from __future__ import annotations
import os

import pandas as pd

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger

log = get_logger(__name__)


class CacheLoader:
    """Read, write, and merge the parquet baseline cache."""

    @staticmethod
    def load_cached_data() -> pd.DataFrame | None:
        if not PipelineConfig.ENABLE_CACHE:
            return None
        if not os.path.exists(PipelineConfig.CACHE_FILE):
            return None
        log.info("Loading cached data from %s...", PipelineConfig.CACHE_FILE)
        return pd.read_parquet(PipelineConfig.CACHE_FILE)

    @staticmethod
    def save_cached_data(df: pd.DataFrame) -> None:
        if not PipelineConfig.ENABLE_CACHE:
            return
        log.info("Saving cache → %s (%d rows)", PipelineConfig.CACHE_FILE, len(df))
        df.to_parquet(PipelineConfig.CACHE_FILE, index=False)

    @staticmethod
    def merge_with_cache(new_df: pd.DataFrame) -> pd.DataFrame:
        cached = CacheLoader.load_cached_data()
        if cached is None:
            return new_df
        merged = pd.concat([cached, new_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=["workorder_id"], keep="last")
        log.info("Merged with cache: %d total rows", len(merged))
        return merged
