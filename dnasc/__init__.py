"""
dnasc
══════
DNA Strain & Construction — pipeline package.

Public API
──────────
    from dnasc import run_pipeline
    from dnasc import render_dashboard
    from dnasc.config import PipelineConfig

Subpackages
───────────
    dnasc.extractors    — BigQuery extraction (BIOS, LSP, LIMS, OpTracker)
    dnasc.transformers  — Data transformation (lineage, processing, repair, validation)
    dnasc.loaders       — Caching and persistence
    dnasc.renderer      — HTML dashboard rendering
"""

from dnasc.pipeline import run_pipeline
from dnasc.renderer import render_dashboard
from dnasc.config import PipelineConfig

__all__ = ["run_pipeline", "render_dashboard", "PipelineConfig"]
__version__ = PipelineConfig.PIPELINE_VERSION
