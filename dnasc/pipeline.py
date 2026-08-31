"""
dnasc/pipeline.py
──────────────────
Top-level pipeline orchestrator.
Coordinates all extractors and transformers in the correct order.
This is the only file that knows the full execution sequence.
"""

from __future__ import annotations
import re
import time
import concurrent.futures

import pandas as pd
from google.cloud import bigquery

from dnasc.config import PipelineConfig
from dnasc.logger import get_logger
from dnasc import protocols as proto
from dnasc.extractors import (
    BIOSExtractor,
    LSPExtractor,
    LIMSExtractor,
    OpTrackerExtractor,
)
from dnasc.transformers import (
    EnrichmentTransformer,
    LineageTransformer,
    ProcessingTransformer,
    RepairTransformer,
    ValidationTransformer,
)
from dnasc.transformers.repair import (
    populate_synthetic_optracker_batch,
    resolve_downstream_plates,
    resolve_lims_streakouts,
    resolve_optracker_streakouts,
    _fetch_well_mapping,
)

# ── Antibiotic normalisation (module-level so tests can import) ───────────────
_AB_NORM = {
    'kan': 'Kan', 'kanamycin': 'Kan',
    'spec': 'Spec', 'spectinomycin': 'Spec',
    'carb': 'Carb', 'carbenicillin': 'Carb', 'amp': 'Carb', 'ampicillin': 'Carb',
}

def _norm_bios_ab(val):
    """Normalise a BIOS antibiotic string to Kan/Spec/Carb, or None if unrecognised."""
    if val is None or (isinstance(val, float) and pd.isna(val)): return None
    v = str(val).strip().lower()
    for k, canon in _AB_NORM.items():
        if re.search(r'\b' + re.escape(k) + r'\b', v): return canon
    return None

# Matches a neomycin-resistance marker (NeoR / Neo / Neomycin) as a standalone token,
# tolerating '_' and '-' delimiters, while NOT matching fluorescent-protein names that
# merely contain the letters "neo" (e.g. mNeonGreen, NeonGreen). The lookbehind/lookahead
# require the "neo…" run to not be flanked by other letters.
_NEO_MARKER_RE = re.compile(r'(?<![A-Za-z])neo(?:r|mycin)?(?![A-Za-z])', re.I)

def _lims_ab_raw_set(r):
    """Every bacterial-selection antibiotic LIMS flags on this plasmid, UNADJUSTED.
    Used to surface the data-quality case where LIMS lists more than one antibiotic on a
    single plasmid (see lims_double_marker) — independent of neo adjustment."""
    out = set()
    if r.get('lims_anti_kan')  is True: out.add('Kan')
    if r.get('lims_anti_spec') is True: out.add('Spec')
    if r.get('lims_anti_carb') is True: out.add('Carb')
    return out

def _lims_ab_set(r):
    """Set of canonical bacterial-selection antibiotics LIMS records for this plasmid.

    LIMS can flag more than one marker on a single plasmid. We drop Kan when the construct
    name/alias carries a neomycin marker (NeoR/Neo/Neomycin) — in that case anti_kan reflects
    the neo/kan-family cargo gene used for *mammalian* selection, not the bacterial cloning
    marker (which is Spec/Carb). mNeonGreen and similar fluorescent proteins are NOT treated
    as neomycin. Returns a set of {'Kan','Spec','Carb'}."""
    _text = f"{r.get('lims_plasmid_alias') or ''} {r.get('construct_name') or ''}"
    _has_neo = bool(_NEO_MARKER_RE.search(_text))
    out = set()
    if r.get('lims_anti_kan')  is True and not _has_neo: out.add('Kan')
    if r.get('lims_anti_spec') is True: out.add('Spec')
    if r.get('lims_anti_carb') is True: out.add('Carb')
    return out

log = get_logger(__name__)


def run_pipeline() -> pd.DataFrame:
    """
    Execute the full DNA SC data pipeline.

    Steps
    ─────
    1.  Parallel extraction  (BIOS, LSP workorders, LSP aliquots, OpTracker)
    2.  LSP merge & orphan recovery
    3.  Lineage bridging
    4.  Synthetic streakout creation
    5.  Core processing (JSON parsing, status enrichment, yield calc)
    6.  LSP root assignment
    7.  OpTracker aggregation
    8.  LIMS colony extraction
    9.  Final merges
    10. Synthetic OpTracker population
    11. Root repair & metadata backfill
    12. Smart filtering & UI enrichment

    Returns
    ───────
    pd.DataFrame  — fully enriched, render-ready dataset
    """
    pipeline_start = time.time()
    _step_times: dict[str, float] = {}
    log.info("=" * 70)
    log.info("PIPELINE START  version=%s", PipelineConfig.PIPELINE_VERSION)
    log.info("=" * 70)

    # ── STEP 1: Parallel extraction ───────────────────────────────────────────
    log.info("STEP 1 — Parallel extraction")
    t = time.time()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        f_bios  = pool.submit(BIOSExtractor.get_bios_workorders)
        f_lsp   = pool.submit(LSPExtractor.get_lsp_workorders)
        f_aliq  = pool.submit(LSPExtractor.get_lsp_aliquots)
        f_op    = pool.submit(OpTrackerExtractor.get_optracker_operations)
        f_oidx  = pool.submit(LSPExtractor.get_lsp_order_index)

        bios_df      = f_bios.result()
        lsp_df       = f_lsp.result()
        aliq_df      = f_aliq.result()
        lsp_idx_df   = f_oidx.result()
        optracker_raw, _excluded_pids = f_op.result()
    _step_times["1-extraction"] = time.time() - t
    log.info("Extraction complete in %.2fs", _step_times["1-extraction"])

    if _excluded_pids:
        before = len(bios_df)
        bios_df = bios_df[~bios_df["workorder_id"].isin(_excluded_pids)]
        dropped = before - len(bios_df)
        if dropped:
            log.info("Dropped %d BIOS workorder(s) for excluded process_ids", dropped)

    # ── STEP 2: LSP merge & orphan recovery ───────────────────────────────────
    log.info("STEP 2 — LSP merge & orphan recovery")
    t = time.time()
    lsp_full = _merge_lsp(lsp_df, aliq_df)
    _step_times["2-lsp-merge"] = time.time() - t
    log.info("LSP merge complete in %.2fs", _step_times["2-lsp-merge"])

    # ── Well mapping: fetch once, reuse in Steps 4 and 11 ────────────────────
    log.info("Fetching recursive well → workorder mapping (shared by Steps 4 + 11)...")
    t = time.time()
    try:
        _wm_client = bigquery.Client(project=PipelineConfig.PROJECT_ID)
        _well_mapping = _fetch_well_mapping(_wm_client, PipelineConfig.PROJECT_ID)
        _step_times["2b-well-mapping"] = time.time() - t
        log.info("Well mapping ready: %d wells in %.2fs", len(_well_mapping), _step_times["2b-well-mapping"])
    except Exception as _exc:
        _step_times["2b-well-mapping"] = time.time() - t
        log.warning("Well mapping prefetch failed (%s) — Steps 4/11 will fetch independently", _exc)
        _well_mapping = None

    # ── STEP 3–5: Lineage, synthetics, processing ─────────────────────────────
    log.info("STEP 3 — Lineage bridging")
    t = time.time()
    workorder_data = LineageTransformer.bridge_lsp_lineage(bios_df, lsp_full)
    _step_times["3-lineage"] = time.time() - t
    log.info("Lineage bridging complete in %.2fs", _step_times["3-lineage"])

    log.info("STEP 4 — Synthetic streakout creation")
    t = time.time()
    workorder_data = RepairTransformer.create_synthetic_streakouts(workorder_data, well_mapping=_well_mapping)
    _step_times["4-syn-streakouts"] = time.time() - t
    log.info("Synthetic streakout creation complete in %.2fs", _step_times["4-syn-streakouts"])

    log.info("STEP 5 — Core processing")
    t = time.time()
    processed = ProcessingTransformer.process_workorder_data(workorder_data)
    _step_times["5-processing"] = time.time() - t
    log.info("Core processing complete in %.2fs", _step_times["5-processing"])

    # ── STEP 6: LSP root assignment ───────────────────────────────────────────
    log.info("STEP 6 — LSP root assignment")
    t = time.time()
    processed = _assign_lsp_roots(processed)
    _step_times["6-lsp-roots"] = time.time() - t
    log.info("LSP root assignment complete in %.2fs", _step_times["6-lsp-roots"])

    # ── STEP 7: OpTracker aggregation ─────────────────────────────────────────
    log.info("STEP 7 — OpTracker aggregation")
    t = time.time()
    optracker_raw["process_id"] = (
        optracker_raw["process_id"].astype(str).str.strip('"').str.lower()
    )
    op_agg = (
        optracker_raw.groupby("process_id")
        .agg({
            "protocol_name":       list,
            "operation_state":     list,
            "operation_start":     list,
            "operation_ready":     list,
            "job_id":              list,
            "well_location":       list,
            "ngs_run_number":      list,
            "confirmed_input_ids":  "first",
            "input_dna_plasmids":   lambda x: next((v for v in x if v is not None and str(v) != 'nan'), None),
            "input_stock_wells":    lambda x: next((v for v in x if v is not None and str(v) != 'nan'), None),
        })
        .reset_index()
    )

    # Secondary aggregation keyed by LSP batch ID (e.g. "LSP-8403").
    # Synthetic LSP rows have no BIOS workorder UUID so they never match
    # op_agg; but OpTracker operations carry an "LSP Batch" parameter that
    # maps directly to the batch ID — use that as a fallback.
    _lsp_op_rows = optracker_raw[optracker_raw["lsp_batch_id_from_optracker"].notna()].copy()
    _lsp_op_rows["_lsp_key"] = _lsp_op_rows["lsp_batch_id_from_optracker"].str.strip().str.upper()
    lsp_op_agg = (
        _lsp_op_rows.groupby("_lsp_key")
        .agg({
            "protocol_name":    list,
            "operation_state":  list,
            "operation_start":  list,
            "operation_ready":  list,
            "job_id":           list,
            "well_location":    list,
            "ngs_run_number":   list,
        })
        .reset_index()
        .rename(columns={"_lsp_key": "lsp_batch_key"})
    )
    _step_times["7-optracker-agg"] = time.time() - t
    log.info("OpTracker aggregated in %.2fs", _step_times["7-optracker-agg"])

    # ── STEP 7b: LIMS streakout resolution ────────────────────────────────────
    # Must run before Step 8 so colony extraction includes the new synthetic rows.
    log.info("STEP 7b — LIMS streakout resolution")
    t = time.time()
    processed = resolve_lims_streakouts(processed)
    _step_times["7b-lims-streakouts"] = time.time() - t
    log.info("LIMS streakout resolution complete in %.2fs", _step_times["7b-lims-streakouts"])

    # ── STEP 8: LIMS colony extraction (parallel) ─────────────────────────────
    log.info("STEP 8 — LIMS colony extraction")
    t = time.time()
    wo_ids  = processed["workorder_id"].unique().tolist()
    pcr_ids = processed.loc[processed["type"] == "pcr_workorder", "workorder_id"].dropna().unique().tolist()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        f_colony   = pool.submit(LIMSExtractor.get_colony_data, wo_ids)
        f_picking  = pool.submit(LIMSExtractor.get_colony_picking_counts, wo_ids)
        f_comments = pool.submit(LIMSExtractor.get_well_comments, pcr_ids)
        colony_data    = f_colony.result()
        picking_counts = f_picking.result()
        well_comments  = f_comments.result()
    _step_times["8-colony-extraction"] = time.time() - t
    log.info("Colony data extracted in %.2fs", _step_times["8-colony-extraction"])

    # ── STEP 9: Final merges ──────────────────────────────────────────────────
    log.info("STEP 9 — Final merges")
    t = time.time()
    final_df = processed.merge(colony_data, on="workorder_id", how="left")
    if not picking_counts.empty:
        final_df = final_df.merge(picking_counts, on="workorder_id", how="left")
    if not well_comments.empty:
        final_df = final_df.merge(well_comments, on="workorder_id", how="left")

    # ── Antibiotic mismatch detection ────────────────────────────────────────
    # lims_anti_kan/spec/carb come directly from the BIOS BQ query (lims__src.plasmid JOIN).
    # LIMS can flag multiple markers on one plasmid. We derive the *set* of bacterial
    # selection antibiotics (dropping neo-derived Kan, see _lims_ab_set), then:
    #   • flag a mismatch only when the BIOS antibiotic isn't among that set, and
    #   • flag lims_double_marker when 2+ real markers remain (informational).
    _raw_sets  = final_df.apply(_lims_ab_raw_set, axis=1)   # unadjusted LIMS flags
    _lims_sets = final_df.apply(_lims_ab_set, axis=1)        # neo-adjusted → drives mismatch
    final_df['lims_antibiotic']  = _lims_sets.apply(lambda s: ', '.join(sorted(s)) if s else None)
    final_df['lims_all_markers'] = _raw_sets.apply(lambda s: ', '.join(sorted(s)) if s else None)
    # Data-quality flag: LIMS lists 2+ antibiotics on ONE plasmid. Even when we can resolve the
    # correct one (neo-derived Kan dropped so no actionable mismatch), the double-listing is a
    # BIOS/LIMS source-record error that should still be surfaced for correction.
    final_df['lims_double_marker'] = _raw_sets.apply(lambda s: len(s) >= 2)
    # True when the double-listing is specifically the NeoR/neo artifact (a Kan we dropped).
    final_df['lims_neo_kan_artifact'] = [
        ('Kan' in raw) and ('Kan' not in eff)
        for raw, eff in zip(_raw_sets, _lims_sets)
    ]
    _bios_norm = final_df['antibiotic'].apply(_norm_bios_ab)
    final_df['antibiotic_mismatch'] = [
        (b is not None) and bool(s) and (b not in s)
        for b, s in zip(_bios_norm, _lims_sets)
    ]

    final_df["join_key"] = (
        final_df["workorder_id"].astype(str)
        .str.replace("STBL3_", "", case=False, regex=False)
        .str.lower()
    )
    # Drop any op_agg columns already present in final_df before merging.
    # If these columns exist (e.g. as null scalars from an upstream change),
    # pandas suffix logic would keep the stale column as "protocol_name" and
    # put the aggregated lists in "protocol_name_raw_op" — then the dedup
    # guard would discard the lists silently. Explicit drop makes op_agg win.
    _op_merge_cols = [
        "protocol_name", "operation_state", "operation_start", "operation_ready",
        "job_id", "well_location", "ngs_run_number",
        "confirmed_input_ids", "input_dna_plasmids", "input_stock_wells",
    ]
    final_df = final_df.drop(
        columns=[c for c in _op_merge_cols if c in final_df.columns],
        errors="ignore",
    )
    final_df = final_df.merge(
        op_agg,
        left_on="join_key", right_on="process_id",
        how="left",
    )
    # For LSP rows that got no queue data from the primary merge,
    # fill from the lsp_op_agg keyed by lsp_batch_id_from_optracker.
    # Pass 1: synthetic LSP rows (workorder_id = "LSP-XXXX")
    # Pass 2: real LSP workorders (UUID workorder_id) using bios_batch_id
    import numpy as _np
    _op_cols = ["protocol_name", "operation_state", "operation_start", "operation_ready", "job_id", "well_location", "ngs_run_number"]
    _no_ops = ~final_df["protocol_name"].apply(lambda x: isinstance(x, (list, _np.ndarray)) and len(x) > 0)

    _syn_lsp_empty = (
        final_df["workorder_id"].astype(str).str.upper().str.startswith("LSP-") & _no_ops
    )
    if _syn_lsp_empty.any() and not lsp_op_agg.empty:
        _fill_idx  = final_df[_syn_lsp_empty].index
        _fill_keys = final_df.loc[_fill_idx, "workorder_id"].astype(str).str.upper()
        for _col in _op_cols:
            _col_map = lsp_op_agg.set_index("lsp_batch_key")[_col].to_dict()
            final_df.loc[_fill_idx, _col] = _fill_keys.map(_col_map)
        log.info(
            "Filled OpTracker queue data for %d synthetic LSPs via lsp_batch_id",
            _syn_lsp_empty.sum(),
        )

    # Pass 2: real LSP workorders (UUID) with bios_batch_id set but still no queue data
    if "bios_batch_id" in final_df.columns and not lsp_op_agg.empty:
        _no_ops2 = ~final_df["protocol_name"].apply(lambda x: isinstance(x, (list, _np.ndarray)) and len(x) > 0)
        _real_lsp_empty = (
            (final_df["type"] == "lsp_workorder") &
            ~final_df["workorder_id"].astype(str).str.upper().str.startswith("LSP-") &
            final_df["bios_batch_id"].notna() &
            _no_ops2
        )
        if _real_lsp_empty.any():
            _fill_idx2  = final_df[_real_lsp_empty].index
            _fill_keys2 = final_df.loc[_fill_idx2, "bios_batch_id"].astype(str).str.upper()
            for _col in _op_cols:
                _col_map = lsp_op_agg.set_index("lsp_batch_key")[_col].to_dict()
                final_df.loc[_fill_idx2, _col] = _fill_keys2.map(_col_map)
            log.info(
                "Filled OpTracker queue data for %d real LSPs via bios_batch_id",
                _real_lsp_empty.sum(),
            )

    final_df = _attach_lsp_order_index(final_df, lsp_idx_df)
    _step_times["9-merges"] = time.time() - t
    log.info("Final merges complete in %.2fs", _step_times["9-merges"])

    # Deduplicate columns introduced by the Step 9 merge before any concat steps
    if final_df.columns.duplicated().any():
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # ── STEP 9b: Resolve OpTracker-only streakouts ────────────────────────────
    log.info("STEP 9b — OpTracker streakout resolution")
    t = time.time()
    final_df = resolve_optracker_streakouts(final_df, optracker_raw)
    _step_times["9b-optracker-streakouts"] = time.time() - t
    log.info("OpTracker streakout resolution complete in %.2fs", _step_times["9b-optracker-streakouts"])

    # ── STEP 10: Synthetic OpTracker population ───────────────────────────────
    log.info("STEP 10 — Synthetic OpTracker population")
    t = time.time()
    final_df = populate_synthetic_optracker_batch(final_df)
    _step_times["10-syn-optracker"] = time.time() - t
    log.info("Synthetic OpTracker populated in %.2fs", _step_times["10-syn-optracker"])

    # ── STEP 11: Root repair & metadata backfill ──────────────────────────────
    log.info("STEP 11 — Root repair & metadata backfill")
    t = time.time()
    final_df = RepairTransformer.repair_data(final_df, well_mapping=_well_mapping)
    # Pull agar-derived synthetic rows (picks) onto the assembly root now that
    # repair has collapsed roots. Before _assign_lsp_roots so an LSP hanging off
    # a moved pick is re-resolved onto the same root in the next call.
    final_df = _reroot_synthetic_picks(final_df)
    # Re-run LSP root assignment after repair — Step 6 ran before Step 11
    # resolved STREAK synthetic row roots, so legacy LSPs sourced via
    # lsp_process_id=STREAK_well* were left self-rooted. Now that repair has
    # filled STREAK roots, re-resolve so those LSPs inherit the correct
    # root_work_order_id, experiment_name, and cloning_strain.
    final_df = _assign_lsp_roots(final_df)
    final_df = _finalize_metadata(final_df)
    _step_times["11-repair"] = time.time() - t
    log.info("Repair complete in %.2fs", _step_times["11-repair"])

    # ── STEP 12: Smart filtering & UI enrichment ──────────────────────────────
    log.info("STEP 12 — Smart filtering & UI enrichment")
    t = time.time()
    final_df = _filter_and_enrich(final_df)
    _step_times["12-filter-enrich"] = time.time() - t
    log.info("Filtering complete in %.2fs", _step_times["12-filter-enrich"])

    # ── Recompute derived columns that depend on root_work_order_id ──────────
    # is_fulfillment is first set at Step 5 for dedup, but root_work_order_id
    # changes during Step 11 repair.  Recompute here so it reflects the final
    # repaired root assignment rather than the pre-repair value.
    final_df["is_fulfillment"] = final_df["workorder_id"] == final_df["root_work_order_id"]

    # ── Dedup columns from merges ─────────────────────────────────────────────
    final_df = final_df.loc[:, ~final_df.columns.duplicated()]

    # ── Sanitize any cells that were corrupted by duplicate-column assignments ─
    # When a df has duplicate column names and a groupby/fillna is applied, some
    # cells may end up containing pd.Series objects instead of scalar values.
    # Extract the first non-null value from any such cells before serialization.
    import pandas as _pd
    for _col in final_df.select_dtypes(include="object").columns:
        _has_series = final_df[_col].map(lambda x: isinstance(x, (_pd.Series, _pd.DataFrame))).any()
        if _has_series:
            log.warning("Sanitizing Series-valued cells in column %s", _col)
            final_df[_col] = final_df[_col].map(
                lambda x: (x.dropna().iloc[0] if not x.dropna().empty else None)
                if isinstance(x, (_pd.Series, _pd.DataFrame)) else x
            )

    # ── Colony status overrides ───────────────────────────────────────────────
    # _bridge_status (Step 12) sets visual_status from wo_status + op states.
    # For colony types (GG, Gibson, Transformation, Streakout), LIMS colony
    # counts can override that — e.g. BIOS=SUCCEEDED but 0 LIMS colonies → FAILED.
    # Running this here keeps the stored visual_status in sync with what the
    # renderer would show, so parquet queries and the dashboard agree.
    t = time.time()
    final_df = _apply_colony_status_overrides(final_df)
    final_df = _detect_colony_repicks(final_df)
    _step_times["12b-colony-overrides"] = time.time() - t

    # ── Downstream plate resolution (must run after _detect_colony_repicks) ───
    # _detect_colony_repicks adds "Repick: Miniprep/Glycerol/Media" to protocol_name
    # and "Manual Repick" plate IDs to all_protocol_plates. resolve_downstream_plates
    # gates on has_repick (protocol_name check), so it must run after repick detection
    # or it always excludes repick workorders that already have original-pick NGS ops.
    log.info("STEP 10b — Downstream plate resolution (post-repick)")
    t = time.time()
    final_df = resolve_downstream_plates(final_df)
    _step_times["10b-downstream-plates"] = time.time() - t
    log.info("Downstream plate resolution complete in %.2fs", _step_times["10b-downstream-plates"])

    # ── Re-derive visual_status for repick workorders from downstream op states ─
    # _detect_colony_repicks sets visual_status=RUNNING unconditionally. Now that
    # resolve_downstream_plates has appended the repick's Rearray/Quant/NGS ops,
    # apply the same colony-status logic as _apply_colony_status_overrides uses
    # for a SUCCEEDED colony workorder — scoped to the post-repick op slice.
    # This mirrors how streakout status is derived: OpTracker states + seq_confirmed.
    def _repick_status(row):
        if row.get("visual_status") != "RUNNING":
            return row["visual_status"]
        try:
            tot = int(row.get("repick_total_colonies") or 0)
            if tot <= 0:
                return row["visual_status"]
            pn = row.get("protocol_name")
            st = row.get("operation_state")
            if hasattr(pn, "tolist"): pn = pn.tolist()
            if hasattr(st, "tolist"): st = st.tolist()
            if not (isinstance(pn, list) and isinstance(st, list)):
                return row["visual_status"]
            # Find the LAST repick op (handles >1 repick round)
            repick_idx = None
            for i, p in enumerate(pn):
                if p == proto.REPICK:
                    repick_idx = i
            if repick_idx is None:
                return row["visual_status"]
            post_pn = pn[repick_idx + 1:]
            post_st = st[repick_idx + 1:]
            if not post_pn:
                return "RUNNING"  # repick plates found, nothing downstream yet
            # The repick's own confirmed count, not the original pick's. A repick
            # only fires on a FAILED parent, so seq_confirmed here is 0 by
            # construction — reading it alone made every confirmed repick resolve
            # back to FAILED off the post-repick NGS op.
            seq = int(row.get("seq_confirmed") or 0) + int(row.get("repick_seq_confirmed") or 0)
            # Active-op check must precede SC/FA check — the post-repick slice also
            # contains original-pick NGS ops (SC/FA) re-appended by
            # resolve_downstream_plates, so checking SC/FA first would falsely FAIL
            # a workorder whose repick NGS is still running.
            if any(s in ("RU", "RD") for s in post_st):
                return "RUNNING"
            has_progress = any(
                p in proto.PROGRESS_PROTOS and s == "SC"
                for p, s in zip(post_pn, post_st)
            )
            if not has_progress:
                return "RUNNING"  # ops present but none SC yet
            return _seq_status_from_ops(post_pn, post_st, seq)
        except Exception:
            pass
        return row["visual_status"]

    _repick_mask = (
        (final_df["visual_status"] == "RUNNING") &
        (final_df.get("repick_total_colonies", pd.Series(0, index=final_df.index)).fillna(0).astype(int) > 0)
    )
    if _repick_mask.any():
        _before = final_df.loc[_repick_mask, "visual_status"].copy()
        final_df.loc[_repick_mask, "visual_status"] = final_df[_repick_mask].apply(_repick_status, axis=1)
        _flipped = (_before != final_df.loc[_repick_mask, "visual_status"]).sum()
        if _flipped:
            log.info("Repick status re-derived for %d workorders from downstream op states", _flipped)

    # ── Attempt anchor recompute (must run last, after all row filtering) ────
    # Uses normalized backbone/parts columns. Runs here rather than in Step 2
    # so anchors are never computed against workorders that are later filtered
    # out by repair or _filter_and_enrich — which would leave referencing rows
    # with a stale anchor pointing to a missing workorder.
    t = time.time()
    final_df = ProcessingTransformer._compute_attempt_anchors(final_df)
    _step_times["13-attempt-anchors"] = time.time() - t

    # Roll up each assembly's downstream verdict into `chain_status` — the single
    # source of truth both dashboard tabs read so the tracking and colony views
    # can never disagree on whether an attempt succeeded.
    t = time.time()
    final_df = ProcessingTransformer._compute_chain_status(final_df)
    _step_times["13b-chain-status"] = time.time() - t

    t = time.time()
    final_df = EnrichmentTransformer.compute_request_enrichment(final_df)
    _step_times["14-enrichment"] = time.time() - t

    elapsed = time.time() - pipeline_start
    log.info("─" * 70)
    log.info("STEP TIMING BREAKDOWN (slowest first):")
    for _sname, _st in sorted(_step_times.items(), key=lambda x: x[1], reverse=True):
        log.info("  %-30s %6.1fs  (%4.1f%%)", _sname, _st, 100 * _st / elapsed)
    log.info("─" * 70)
    log.info("=" * 70)
    log.info("PIPELINE COMPLETE  %.1fs  |  %d rows  |  %d experiments  |  %d requests",
             elapsed, len(final_df),
             final_df["experiment_name"].nunique(),
             final_df["req_id"].nunique())
    log.info("=" * 70)
    return final_df


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers  (logic extracted from the old monolith run_pipeline())
# ─────────────────────────────────────────────────────────────────────────────

_COLONY_TYPES = frozenset({
    "gibson_workorder", "golden_gate_workorder", "transformation_workorder",
    "transformation_offline_operation", "streakout_operation",
    # Manual repicks logged in LIMS under their own hand-typed process id
    # (e.g. PICK_25Aug26_well2176911) surface as their own optracker_operation
    # row rather than folding into the parent. get_colony_data already matches
    # them on well.process_id, so they carry real colony counts — they just
    # never got the status override that turns those counts into a verdict.
    "optracker_operation",
})


def _seq_status_from_ops(pn: list, ps: list, seq: int) -> str:
    """
    Derive visual_status from protocol/state lists when at least one
    progress-milestone op is already complete (SC) and no ops are still
    active (RU/RD).  Called by both _apply_colony_status_overrides and
    _repick_status to avoid duplicating this logic.

    Returns one of: SUCCEEDED, FAILED, IN_PROGRESS.
    Callers are responsible for the RUNNING/READY guard before calling this.
    """
    if seq > 0:
        return "SUCCEEDED"
    if any(p in proto.SEQ_PROTOS and s in ("SC", "FA") for p, s in zip(pn, ps)):
        return "FAILED"
    return "IN_PROGRESS"


def _apply_colony_status_overrides(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply LIMS colony-count overrides to visual_status for colony-type workorders.
    Also sets is_software_fail (BIOS=FAILED but seq confirmed → display SUCCEEDED).
    Must run after _bridge_status (which sets the initial visual_status).
    """
    import numpy as _np

    col_mask = df["type"].isin(_COLONY_TYPES)
    if not col_mask.any():
        df["is_software_fail"] = False
        return df

    def _override(row):
        wo = str(row.get("wo_status") or "").strip().upper()
        if wo in ("", "NAN", "NONE", "UNKNOWN"):
            return row["visual_status"], False, False

        try:
            tot = int(row.get("total_colonies")) if pd.notna(row.get("total_colonies")) else 0
        except (TypeError, ValueError):
            tot = 0
        try:
            seq = int(row.get("seq_confirmed")) if pd.notna(row.get("seq_confirmed")) else 0
        except (TypeError, ValueError):
            seq = 0

        # Manual-repick rows carry colony counts but none of the transformation-
        # shaped protocol sequence the SUCCEEDED block below assumes (no Gibson,
        # no STAR Transformation — just Miniprep/Glycerol/Rearray/Quant/NGS), so
        # running that logic on them would flip finished picks to IN_PROGRESS.
        # Only the seq-confirmed rescue applies: a pick with a confirmed colony
        # is a success regardless of what the last op state happened to be.
        if row.get("type") == "optracker_operation":
            if seq > 0:
                return "SUCCEEDED", wo == "FAILED", False
            return row["visual_status"], False, False

        if wo == "FAILED" and seq > 0:
            return "SUCCEEDED", True, False

        if wo in ("RUNNING", "IN_PROGRESS") and seq > 0:
            return "SUCCEEDED", False, False

        if wo == "SUCCEEDED":
            pn = row.get("protocol_name")
            ps = row.get("operation_state")
            if isinstance(pn, _np.ndarray): pn = list(pn)
            if isinstance(ps, _np.ndarray): ps = list(ps)

            if tot == 0:
                if isinstance(pn, list) and isinstance(ps, list):
                    transf_done = any(
                        p in proto.TRANSF_PROTOS and s in ("SC", "FA")
                        for p, s in zip(pn, ps)
                    )
                    if not transf_done:
                        if any(s == "RU" for s in ps): return "RUNNING", False, False
                        if any(s == "RD" for s in ps): return "READY", False, False
                        return "IN_PROGRESS", False, False
                    miniprep = next(
                        (s for p, s in zip(pn, ps) if p == proto.MINIPREP), None
                    )
                    if miniprep == "RD": return "READY", False, False
                    if miniprep == "RU": return "RUNNING", False, False
                return "FAILED", False, False

            if isinstance(pn, list) and len(pn) > 0:
                if isinstance(ps, list):
                    has_progress = any(
                        p in proto.PROGRESS_PROTOS and s == "SC"
                        for p, s in zip(pn, ps)
                    )
                    if not has_progress:
                        miniprep_s = next(
                            (s for p, s in zip(pn, ps) if p == proto.MINIPREP), None
                        )
                        if miniprep_s == "SC": return "FAILED", False, False
                        if miniprep_s == "RU": return "RUNNING", False, False
                        if miniprep_s == "RD": return "READY", False, False
                    else:
                        if any(s == "RU" for s in ps): return "RUNNING", False, False
                        if any(s == "RD" for s in ps): return "READY", False, False
                        return _seq_status_from_ops(pn, ps, seq), False, False
            else:
                if tot > 0 and seq == 0:
                    return "RUNNING", False, False

        return wo, False, False

    overrides = df.loc[col_mask].apply(_override, axis=1, result_type="expand")
    overrides.columns = ["visual_status", "is_software_fail", "is_seq_rollback"]
    df = df.copy()
    df.loc[col_mask, "visual_status"]    = overrides["visual_status"]
    df.loc[col_mask, "is_software_fail"] = overrides["is_software_fail"]
    df["is_software_fail"] = df["is_software_fail"].where(df["is_software_fail"].notna(), False)
    df["is_status_override"] = (
        df["visual_status"].astype(str).str.upper()
        != df["wo_status"].astype(str).str.upper()
    ) & df["wo_status"].notna()
    return df

def _detect_colony_repicks(df: pd.DataFrame) -> pd.DataFrame:
    """
    For FAILED GG/Gibson/Transformation workorders with NGS FA ops, check LIMS
    for manually-created miniprep plates made after the NGS failure. If found,
    append a synthetic 'Repick: Miniprep/Glycerol/Media'
    op to the timeline and flip visual_status to RUNNING.
    """
    import json as _json
    import numpy as _np
    from dnasc.extractors.lims import LIMSExtractor

    _COLONY_TYPES = frozenset({
        "gibson_workorder", "golden_gate_workorder",
        "transformation_workorder", "transformation_offline_operation",
    })
    _SEQ_PROTOCOLS = {"NGS Sequence Confirmation", "Fragment Analyzer"}

    failed_mask = df["type"].isin(_COLONY_TYPES) & (df["visual_status"] == "FAILED")
    if not failed_mask.any():
        return df

    workorder_cutoffs: dict[str, str] = {}
    for _, row in df[failed_mask].iterrows():
        pn = row.get("protocol_name")
        ps = row.get("operation_state")
        ts = row.get("operation_start")
        if isinstance(pn, _np.ndarray): pn = list(pn)
        if isinstance(ps, _np.ndarray): ps = list(ps)
        if isinstance(ts, _np.ndarray): ts = list(ts)
        if not (isinstance(pn, list) and isinstance(ps, list) and isinstance(ts, list)):
            continue
        fa_times = [t for p, s, t in zip(pn, ps, ts) if p in _SEQ_PROTOCOLS and s == "FA" and t]
        if not fa_times:
            continue
        workorder_cutoffs[str(row["workorder_id"])] = max(str(t) for t in fa_times)

    if not workorder_cutoffs:
        return df

    repick_df = LIMSExtractor.get_repick_plates(workorder_cutoffs)
    if repick_df.empty:
        return df

    log.info("_detect_colony_repicks: repick plates found for %d workorders", repick_df["workorder_id"].nunique())

    # Count distinct colony_numbers per workorder from repick plates
    _named = repick_df[repick_df["colony_number"].notna()]
    repick_colony_counts = _named.groupby("workorder_id")["colony_number"].nunique()

    # ...and how many of those colonies came back sequence-confirmed. A colony is
    # held in several wells (overnight, glycerol, miniprep) and only some carry
    # the seq_confirmed flag, so count DISTINCT colony_numbers with any True row
    # rather than counting rows.
    if "seq_confirmed" in _named.columns:
        repick_seq_counts = (
            _named[_named["seq_confirmed"] == True]
            .groupby("workorder_id")["colony_number"]
            .nunique()
        )
    else:
        repick_seq_counts = pd.Series(dtype=int)

    if "repick_total_colonies" not in df.columns:
        df["repick_total_colonies"] = 0
    if "repick_seq_confirmed" not in df.columns:
        df["repick_seq_confirmed"] = 0

    df = df.copy()
    for wo_id, plates in repick_df.groupby("workorder_id"):
        mask = df["workorder_id"].str.upper() == str(wo_id).upper()
        if not mask.any():
            continue
        repick_ts = plates["plate_created_at"].min()
        # One row per (plate, colony) — dedupe so a 6-colony plate is listed once.
        repick_plate_ids = ",".join(
            dict.fromkeys(plates["plate_id"].dropna().astype(str).tolist())
        )
        n_repick = int(repick_colony_counts.get(str(wo_id), 0))
        n_repick_seq = int(repick_seq_counts.get(str(wo_id), 0))

        def _append_val(arr, val):
            if isinstance(arr, _np.ndarray): arr = list(arr)
            return (arr + [val]) if isinstance(arr, list) else [val]

        def _inject_repick_plates(json_str, plate_ids_str):
            try:
                data = _json.loads(json_str) if pd.notna(json_str) and json_str not in ('{}', '') else {}
            except Exception:
                data = {}
            data["Manual Repick"] = plate_ids_str
            return _json.dumps(data)

        df.loc[mask, "protocol_name"]      = df.loc[mask, "protocol_name"].apply(_append_val, val="Repick: Miniprep/Glycerol/Media")
        df.loc[mask, "operation_state"]    = df.loc[mask, "operation_state"].apply(_append_val, val="RU")
        df.loc[mask, "operation_start"]    = df.loc[mask, "operation_start"].apply(_append_val, val=repick_ts)
        df.loc[mask, "operation_ready"]    = df.loc[mask, "operation_ready"].apply(_append_val, val=repick_ts)
        if "job_id" in df.columns:
            df.loc[mask, "job_id"] = df.loc[mask, "job_id"].apply(_append_val, val=_np.nan)
        for _pad_col in ("ngs_run_number", "well_location"):
            if _pad_col in df.columns:
                df.loc[mask, _pad_col] = df.loc[mask, _pad_col].apply(_append_val, val=None)
        df.loc[mask, "all_protocol_plates"] = df.loc[mask, "all_protocol_plates"].apply(
            _inject_repick_plates, plate_ids_str=repick_plate_ids
        )
        df.loc[mask, "repick_total_colonies"] = n_repick
        df.loc[mask, "repick_seq_confirmed"]  = n_repick_seq
        # Clamp original colony count to total minus repick (floor 0)
        def _clamp_original(tot, n_rp=n_repick):
            try: return max(0, int(tot) - n_rp)
            except Exception: return 0
        df.loc[mask, "total_colonies"] = df.loc[mask, "total_colonies"].apply(_clamp_original)
        df.loc[mask, "visual_status"]      = "RUNNING"

    return df


def _attach_lsp_order_index(
    final_df: pd.DataFrame, lsp_idx_df: pd.DataFrame
) -> pd.DataFrame:
    """Attach the team's "job id _ index" handle for an LSP (e.g. "9560_8").

    See LSPExtractor.get_lsp_order_index for where the handle comes from. An LSP
    row can be keyed three different ways depending on how it was recovered, so
    try each in turn: the LIMS batch id, the BIOS batch id (real workorders that
    lost the LIMS join), the workorder_id itself (synthetic "LSP-####" rows), and
    finally the LSP Order operation's Process param → the lsp_workorder UUID.
    """
    for _col in ("lsp_order_index", "lsp_order_number"):
        if _col not in final_df.columns:
            final_df[_col] = None

    if lsp_idx_df is None or lsp_idx_df.empty:
        return final_df

    _batch_keys = lsp_idx_df["lsp_batch_id"].astype(str).str.strip().str.upper()
    _proc_keys  = lsp_idx_df["lsp_order_process_id"].astype(str).str.strip()

    for _out, _src in (
        ("lsp_order_index",  "lsp_order_index"),
        ("lsp_order_number", "lsp_order_number"),
    ):
        _by_batch = dict(zip(_batch_keys, lsp_idx_df[_src]))
        _by_proc  = dict(zip(_proc_keys,  lsp_idx_df[_src]))
        _vals = pd.Series(pd.NA, index=final_df.index, dtype="object")
        for _key_col, _lookup, _upper in (
            ("lsp_batch_id",  _by_batch, True),
            ("bios_batch_id", _by_batch, True),
            ("workorder_id",  _by_batch, True),
            ("workorder_id",  _by_proc,  False),
        ):
            if _key_col not in final_df.columns:
                continue
            _need = _vals.isna()
            if not _need.any():
                break
            _keys = final_df.loc[_need, _key_col].astype(str).str.strip()
            if _upper:
                _keys = _keys.str.upper()
            _vals.loc[_need] = _keys.map(_lookup)
        final_df[_out] = _vals

    _hits = final_df["lsp_order_index"].notna().sum()
    log.info("Attached LSP order index to %d row(s)", _hits)
    return final_df


def _merge_lsp(lsp_df: pd.DataFrame, aliq_df: pd.DataFrame) -> pd.DataFrame:
    """Primary + secondary LSP merge with orphan recovery."""
    import re
    import pandas as pd

    cutoff = pd.Timestamp(PipelineConfig.LSP_CUTOFF_DATE, tz="UTC")
    aliq_df_copy = aliq_df.copy()
    aliq_df_copy["batch_created_at"] = pd.to_datetime(
        aliq_df_copy["batch_created_at"], errors="coerce"
    )

    lsp_full = lsp_df.merge(
        aliq_df_copy, how="outer",
        left_on="workorder_id", right_on="lsp_process_id",
        indicator=True,
    )
    lsp_full["batch_created_at"] = pd.to_datetime(lsp_full["batch_created_at"], errors="coerce")

    # Secondary pass — identity recovery for unlinked batches that map to a known BIOS workorder.
    # Covers: (a) post-cutoff batches where process_id mismatch prevents primary join,
    #         (b) pre-cutoff batches that still have a BIOS workorder (e.g. CANCELED/FAILED).
    batch_lookup = (
        lsp_df.dropna(subset=["bios_batch_id"])
        .set_index("bios_batch_id")["workorder_id"]
        .to_dict()
    )
    right_only_mask = lsp_full["_merge"] == "right_only"
    failed_mask = right_only_mask & (
        (lsp_full["batch_created_at"] >= cutoff) |
        lsp_full["lsp_batch_id"].isin(set(batch_lookup.keys()))
    )
    if failed_mask.any():
        log.info("Identity recovery for %d unlinked LSP rows", failed_mask.sum())
        lsp_full.loc[failed_mask, "workorder_id"] = (
            lsp_full.loc[failed_mask, "lsp_batch_id"].map(batch_lookup)
        )
        lsp_full.loc[failed_mask, "data_source"] = "LSP"
        lsp_full.loc[failed_mask, "type"] = "lsp_workorder"
        success_mask = failed_mask & lsp_full["workorder_id"].notna()
        lsp_full.loc[success_mask, "_merge"] = "both"

        # Copy wo_status from the matched lsp_df row so that CANCELED workorders
        # don't appear as IN_PROGRESS on the recovered aliquot row.
        # (The right_only row has wo_status=NaN; without this copy it bypasses
        # the canceled_no_work filter downstream.)
        for _status_col in ["wo_status", "request_id"]:
            if _status_col in lsp_df.columns:
                _col_lookup = (
                    lsp_df.dropna(subset=["bios_batch_id"])
                    .drop_duplicates("bios_batch_id")
                    .set_index("bios_batch_id")[_status_col]
                    .to_dict()
                )
                lsp_full.loc[success_mask, _status_col] = (
                    lsp_full.loc[success_mask, "lsp_batch_id"].map(_col_lookup)
                )

    # Recombine
    matched    = lsp_full[lsp_full["_merge"] == "both"].copy()
    pre_aliq   = lsp_full[lsp_full["_merge"] == "left_only"].copy()
    orphaned   = lsp_full[
        (lsp_full["_merge"] == "right_only") &
        (lsp_full["batch_created_at"] < cutoff)
    ].copy()

    if not orphaned.empty:
        orphaned["workorder_id"] = orphaned["lsp_batch_id"]
        orphaned["type"]         = "lsp_workorder"
        orphaned["data_source"]  = "SYNTHETIC_LSP"
        orphaned["wo_status"]    = "UNKNOWN"
        orphaned["wo_created_at"]= orphaned["batch_created_at"]
        orphaned["root_work_order_id"] = None

        # bios_batch_id metadata snap
        bios_lookup = (
            lsp_df.dropna(subset=["bios_batch_id"])
            .drop_duplicates("bios_batch_id")
            .set_index("bios_batch_id")
        )
        meta_cols = [
            "workorder_id", "wo_status", "req_id", "request_status",
            "request_created_at", "priority", "submitter_email",
            "construct_name", "delivery_format", "for_partner",
            "experiment_name", "root_work_order_id",
            "source_lsp_process_id", "source_workorder_id",
        ]
        for col in meta_cols:
            if col in bios_lookup.columns and col in orphaned.columns:
                orphaned[col] = orphaned[col].fillna(
                    orphaned["lsp_batch_id"].map(bios_lookup[col])
                )

        recovered = orphaned["workorder_id"].notna() & (orphaned["data_source"] == "SYNTHETIC_LSP")
        orphaned.loc[recovered, "data_source"] = "LSP"
        orphaned.loc[recovered, "type"]        = "lsp_workorder"

        # Strip SCALEUP_ prefix
        def _clean_scaleup(val):
            if pd.isna(val) or str(val) == "nan":
                return val
            cleaned = re.sub(r"^SCALEUP_", "", str(val), flags=re.IGNORECASE)
            uuid_re = r"[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}"
            return cleaned if re.match(uuid_re, cleaned, re.IGNORECASE) else val

        if "lsp_process_id" in orphaned.columns:
            cleaned = orphaned["lsp_process_id"].apply(_clean_scaleup)
            changed = cleaned != orphaned["lsp_process_id"]
            orphaned.loc[changed, "source_lsp_process_id"] = cleaned[changed]
            log.debug("Stripped SCALEUP_ prefix from %d orphaned LSPs", changed.sum())

    result = pd.concat([matched, pre_aliq, orphaned], ignore_index=True)
    result = result.drop("_merge", axis=1, errors="ignore")

    # Consolidate duplicate workorder_ids produced by identity recovery.
    # Identity recovery converts right_only aliq rows (yield data, null BIOS cols)
    # to "matched", but the original left_only pre_aliq row (BIOS cols, null yield)
    # still exists.  Sort so rows with lsp_batch_id (yield-bearing) come first,
    # then groupby.first() coalesces: yield from matched + BIOS from pre_aliq.
    if result["workorder_id"].duplicated().any():
        n_before = len(result)
        result = (
            result
            .sort_values("lsp_batch_id", ascending=True, na_position="last")
            .groupby("workorder_id", sort=False, as_index=False)
            .first()
        )
        log.info("Consolidated %d duplicate LSP rows", n_before - len(result))

    log.info("LSP merge result: %d rows", len(result))
    return result


# Synthetic rows created at Step 4 from an agar well (picks, offline
# transformations, streakouts). They all carry source_asm_process_id.
_SYNTHETIC_PICK_TYPES = frozenset({
    "optracker_operation", "transformation_offline_operation", "streakout_operation",
})


def _reroot_synthetic_picks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-root Step-4 synthetic rows onto their source workorder's assembly root.

    create_synthetic_streakouts runs at Step 4, before Step 11 collapses
    root_work_order_id onto the assembly-design root. A pick taken off a
    TRANSFORMATION's agar plate therefore copies a root that is still the
    transformation itself, and the row forms its own workflow group instead of
    appearing under the assembly that agar belongs to. A pick off a GIBSON's agar
    plate looks fine only by accident — a Gibson is already its own root.
    Streakouts escape this because they are resolved later (Steps 7b/9b), once
    roots are final; pAI-20778 shows both behaviours off the same transformation.

    Scoped deliberately: a row moves only if its root IS its source workorder,
    and only onto that source's own root. A synthetic row that streakout
    resolution deliberately re-rooted somewhere else is left alone.

    Must run before _assign_lsp_roots so LSPs hanging off a moved pick follow it.
    """
    import pandas as pd

    if "source_asm_process_id" not in df.columns:
        return df

    id_to_root = (
        df.dropna(subset=["root_work_order_id"])
        .drop_duplicates(subset=["workorder_id"])
        .set_index("workorder_id")["root_work_order_id"]
        .astype(str)
        .to_dict()
    )

    src      = df["source_asm_process_id"].astype(str)
    src_root = src.map(id_to_root)
    mask = (
        df["type"].isin(_SYNTHETIC_PICK_TYPES)
        & df["source_asm_process_id"].notna()
        & (df["root_work_order_id"].astype(str) == src)   # still self-rooted on its source
        & src_root.notna()
        & (src_root != src)                               # and that source has a real root
    )
    if mask.any():
        df = df.copy()
        df.loc[mask, "root_work_order_id"] = src_root[mask]
        log.info(
            "_reroot_synthetic_picks: %d synthetic rows moved onto their assembly root",
            int(mask.sum()),
        )
    return df


def _assign_lsp_roots(df: pd.DataFrame) -> pd.DataFrame:
    """Resolve root_work_order_id for LSP workorders via source link columns."""
    import pandas as pd

    id_to_root = (
        df.dropna(subset=["root_work_order_id"])
        .set_index("workorder_id")["root_work_order_id"]
        .astype(str)
        .to_dict()
    )

    def _resolve(row):
        if row["type"] != "lsp_workorder":
            return row["root_work_order_id"]
        for col in ["source_lsp_process_id", "lsp_process_id", "middle_root", "source_workorder_id"]:
            val = row.get(col)
            if pd.isna(val):
                continue
            val = str(val).strip()
            if val in ("nan", "None", "", str(row["workorder_id"])):
                continue
            if val.upper().startswith("LSP-"):
                continue
            if val in id_to_root:
                root = id_to_root[val]
                if pd.notna(root) and str(root) not in ("nan", "None", ""):
                    return root
            if len(val) > 20:
                return val
        return row.get("root_work_order_id") or row["workorder_id"]

    df["root_work_order_id"] = df.apply(_resolve, axis=1)

    # Backfill source_lsp_process_id for orphans
    lsp_mask     = df["type"] == "lsp_workorder"
    source_empty = lsp_mask & df["source_lsp_process_id"].isna()
    for col in ["lsp_process_id", "source_workorder_id", "middle_root"]:
        if col in df.columns:
            still = source_empty & df["source_lsp_process_id"].isna()
            df.loc[still, "source_lsp_process_id"] = df.loc[still, col]

    self_ref = df["source_lsp_process_id"].astype(str) == df["workorder_id"].astype(str)
    df.loc[self_ref, "source_lsp_process_id"] = None
    log.debug("LSP root assignment complete")
    return df


def _finalize_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Final metadata gap-fill after RepairTransformer."""
    # Temporal disqualifier: if a workorder was created BEFORE its assigned
    # request, the link is spurious (e.g. an LSP batch from wave5 reused by a
    # wave6 request that didn't exist yet).  Nullify req_id/experiment_name so
    # the root-based fill below can assign the correct experiment instead.
    if "wo_created_at" in df.columns and "request_created_at" in df.columns:
        wo_ts  = pd.to_datetime(df["wo_created_at"],       utc=True, errors="coerce")
        req_ts = pd.to_datetime(df["request_created_at"],  utc=True, errors="coerce")
        # Restrict to LSP workorders only — Gibson/Transformation rows are always
        # created FOR their request, not reused across experiments like LSP batches.
        # Exempt lsp_workorders with a direct wo.request_id link (lsp_own_request_id
        # not null): LSP Refill batches are done before the formal refill request is
        # submitted, so wo_created_at < request_created_at is intentional, not spurious.
        # The disqualifier is only for inherited req_ids (via root-based fill).
        direct_link = (
            df["lsp_own_request_id"].notna()
            if "lsp_own_request_id" in df.columns
            else pd.Series(False, index=df.index)
        )
        temporal_mismatch = (
            wo_ts.notna() & req_ts.notna() & (wo_ts < req_ts) &
            (df["type"] == "lsp_workorder") &
            ~direct_link
        )
        df.loc[temporal_mismatch, "req_id"]          = None
        df.loc[temporal_mismatch, "experiment_name"] = None

    # For req_id and experiment_name, fill only from the root row
    # (workorder_id == root_work_order_id), not from any sibling in the group.
    # Using transform("first") can pull a sibling LSP's req_id from a
    # different experiment when multiple requests share the same root
    # (e.g. a wave5 and wave6 LSP both sourced from the same Gibson clone).
    root_rows = df[df["workorder_id"] == df["root_work_order_id"]]
    for col in ["req_id", "experiment_name"]:
        if col in df.columns:
            root_map = root_rows.set_index("root_work_order_id")[col].to_dict()
            df[col] = df[col].fillna(df["root_work_order_id"].map(root_map))

    # experiment_name: also overwrite non-null stale values. Cross-experiment
    # syn_parts (shared across assembly designs) keep their original
    # experiment_name after req_id is assigned to a different experiment.
    # root_map still holds the experiment_name mapping from the loop above.
    # Exception: lsp_workorders with a direct wo.request_id link (lsp_own_request_id
    # not null) must keep their own request's experiment_name — overwriting from
    # the root's experiment (e.g. A581 BMR001 Gibson) would displace LSP Refill
    # entries from the "LSP Refill Requests" experiment.
    if "experiment_name" in df.columns:
        direct_lsp = (
            df["lsp_own_request_id"].notna()
            if "lsp_own_request_id" in df.columns
            else pd.Series(False, index=df.index)
        )
        from_root = df["root_work_order_id"].map(root_map)
        df["experiment_name"] = df["experiment_name"].where(
            direct_lsp,
            from_root.fillna(df["experiment_name"])
        )

    # SYNTHETIC_LSP orphans have no real request of their own — always force
    # their req_id/experiment_name from the root, overriding whatever bios_lookup
    # may have set (e.g. a wave5 orphan incorrectly inheriting a wave6 req_id).
    syn_mask = df["data_source"] == "SYNTHETIC_LSP"
    if syn_mask.any():
        for col in ["req_id", "experiment_name"]:
            if col in df.columns:
                df.loc[syn_mask, col] = df.loc[syn_mask, "root_work_order_id"].map(root_map)

    # Other metadata cols are safe to fill from any group member
    cols = ["request_status", "priority", "construct_name", "for_partner"]
    for col in cols:
        if col in df.columns:
            df[col] = df[col].fillna(
                df.groupby("root_work_order_id")[col].transform("first")
            )
    orphan_mask = (df["data_source"] == "SYNTHETIC_LSP") & df["req_id"].isna()
    df.loc[orphan_mask, "req_id"]         = "ORPHAN_LEGACY"
    df.loc[orphan_mask, "request_status"] = "SUCCEEDED"

    wip_mask = (df["data_source"] == "LSP") & df["req_id"].isna()
    df.loc[wip_mask, "req_id"]         = "ACTIVE_WIP"
    df.loc[wip_mask, "request_status"] = "IN_PROGRESS"

    # Recompute STOCK_ID from final root assignments.
    # root_STOCK_ID is computed in Step 5 (processing.py) before Step 6
    # (_assign_lsp_roots) corrects LSP root pointers, so it may be stale.
    #
    # For LSP rows, fill STOCK_ID from plasmid_id FIRST (the aliquot batch's
    # specific plasmid_id from LIMS).  This prevents a sibling LSP's STOCK_ID
    # from polluting other rows when the group-fill below runs — e.g. two LSPs
    # for the same request sharing root_work_order_id, one with pAI-X and one
    # with a null STOCK_ID that incorrectly inherits pAI-X.
    if "STOCK_ID" in df.columns and "plasmid_id" in df.columns:
        lsp_mask = df["type"] == "lsp_workorder"
        df.loc[lsp_mask, "STOCK_ID"] = df.loc[lsp_mask, "STOCK_ID"].fillna(
            df.loc[lsp_mask, "plasmid_id"]
        )

    # Propagate STOCK_ID from the root workorder down to all family members.
    if "STOCK_ID" in df.columns and "root_work_order_id" in df.columns:
        df["STOCK_ID"] = df["STOCK_ID"].fillna(
            df.groupby("root_work_order_id")["STOCK_ID"].transform("first")
        )

    # Backfill source_lsp_process_id and lsp_input_well from sibling LSP rows
    # sharing the same input_well_id. Legacy workorders (LSP-XXXX format) predate
    # BIOS and have these fields null even when newer workorders for the same
    # source well have them populated.
    # NOTE: this must run BEFORE the cloning_strain fill below so that
    # source_lsp_process_id is resolved before it's used as a lookup key.
    if "input_well_id" in df.columns and "source_lsp_process_id" in df.columns:
        lsp_well_mask = (df["type"] == "lsp_workorder") & df["input_well_id"].notna()
        if lsp_well_mask.any():
            for col in ["source_lsp_process_id", "lsp_input_well",
                        "order_well_plate_id", "order_well_position",
                        "order_well_count", "order_well_labware",
                        "order_well_protocol", "order_well_plate_location"]:
                if col not in df.columns:
                    continue
                filled = (
                    df[lsp_well_mask & df[col].notna()]
                    .drop_duplicates(subset=["input_well_id"])
                    .set_index("input_well_id")[col]
                )
                null_mask = lsp_well_mask & (df[col].isna() | (df[col].astype(str) == "None"))
                if null_mask.any():
                    df.loc[null_mask, col] = df.loc[null_mask, "input_well_id"].map(filled)

    # Backfill qubit_concentration_ngul from batch_comments for older batches
    # that stored qubit as free text ("Qbit Concentration: XXXX") before the
    # structured field was used.  Also backfill nanodrop_concentration_ngul
    # from the deprecated concentration_ngul field (pre-split schema).
    import re as _re
    if "batch_comments" in df.columns and "qubit_concentration_ngul" in df.columns:
        lsp_null_qubit = (df["type"] == "lsp_workorder") & df["qubit_concentration_ngul"].isna()
        if lsp_null_qubit.any():
            def _parse_qbit(c):
                m = _re.search(r"[Qq]bit\s+[Cc]oncentration[:\s]+([0-9.]+)", str(c))
                return float(m.group(1)) if m else None
            parsed = df.loc[lsp_null_qubit, "batch_comments"].apply(_parse_qbit)
            df.loc[lsp_null_qubit, "qubit_concentration_ngul"] = parsed

    if "deprecated_concentration_ngul" in df.columns and "nanodrop_concentration_ngul" in df.columns:
        lsp_null_nano = (df["type"] == "lsp_workorder") & df["nanodrop_concentration_ngul"].isna()
        if lsp_null_nano.any():
            df.loc[lsp_null_nano, "nanodrop_concentration_ngul"] = df.loc[
                lsp_null_nano, "deprecated_concentration_ngul"
            ]

    # Copy cloning_strain from source transformation to LSP rows where null.
    # LSPs sourced from internal transformations don't inherit strain via the
    # root group (they're in a different assembly group), so fill directly.
    if "cloning_strain" in df.columns and "source_lsp_process_id" in df.columns:
        src_strain = (
            df.dropna(subset=["cloning_strain"])
            .set_index("workorder_id")["cloning_strain"]
            .to_dict()
        )
        lsp_null = (df["type"] == "lsp_workorder") & df["cloning_strain"].isna()
        df.loc[lsp_null, "cloning_strain"] = df.loc[lsp_null, "source_lsp_process_id"].map(src_strain)

    return df


def _filter_and_enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Smart filtering, status bridging, and UI enrichment."""
    import pandas as pd

    # Guard against duplicate columns reaching here from upstream merges/concats.
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()]

    blacklist = set(PipelineConfig.LSP_BLACKLIST)
    active_statuses = {
        "NEW", "PLANNED", "IN_PROGRESS", "ACTIVE_WIP",
        "ORPHAN_LEGACY", "SUCCEEDED", "FULFILLED",
    }

    import numpy as np
    lsp_batch = df["lsp_batch_id"].astype(str)
    req_id_s  = df["req_id"].astype(str)
    status_s  = df["request_status"].astype(str)
    keep_mask = (
        ~lsp_batch.isin(blacklist)
        & (
            (df["req_id"].notna() & (
                status_s.isin(active_statuses)
                | req_id_s.str.contains("ORPHAN|ACTIVE", na=False)
            ))
            | lsp_batch.str.startswith("LSP-10")
            | (pd.to_numeric(df.get("total_volume_ul", pd.Series(0, index=df.index)), errors="coerce").fillna(0) > 1.0)
            | df["protocol_name"].map(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0)
        )
    )
    df = df[keep_mask].copy()

    # Remove LSP rows whose location is a test/fake placeholder
    if "location" in df.columns:
        _test_loc = df["location"].astype(str).str.upper()
        _fake_mask = (
            (df["type"] == "lsp_workorder") &
            _test_loc.str.contains(r"\bTEST\b|\bFAKE\b|TESTFAKE", na=False)
        )
        if _fake_mask.any():
            log.info("Removing %d LSP rows with test/fake locations", _fake_mask.sum())
            df = df[~_fake_mask].copy()

    # Remove canceled LSPs with no OpTracker work
    canceled_no_work = (
        (df["type"] == "lsp_workorder") &
        (df["wo_status"].astype(str).str.upper() == "CANCELED") &
        (~df["protocol_name"].apply(lambda x: isinstance(x, (list, np.ndarray)) and len(x) > 0))
    )
    df = df[~canceled_no_work].copy()
    log.info("Removed %d canceled LSPs with no OpTracker work", canceled_no_work.sum())

    # Lineage glue
    df["is_visible"] = True
    if "parent_id" not in df.columns:
        df["parent_id"] = None
    df["parent_id"] = df["parent_id"].fillna(df["source_lsp_process_id"])
    df["parent_id"] = df["parent_id"].fillna(df["root_work_order_id"])
    df.loc[df["workorder_id"] == df["parent_id"], "parent_id"] = None

    all_parents = set(df["root_work_order_id"].dropna().unique())
    df["is_leaf"] = (df["type"] == "lsp_workorder") | (~df["workorder_id"].isin(all_parents))

    # Dedup columns
    df = df.loc[:, ~df.columns.duplicated()]

    # STOCK_ID fallback
    if "STOCK_ID" in df.columns and "root_STOCK_ID" in df.columns:
        df["STOCK_ID"] = df["STOCK_ID"].fillna(df["root_STOCK_ID"])

    # Status bridge
    import numpy as np

    wo  = df["wo_status"].astype(str).str.strip().str.upper()
    req = df["request_status"].astype(str).str.strip().str.upper()
    _bad = {"NAN", "NONE", "", "UNKNOWN"}

    states     = df["operation_state"]
    is_not_lsp = df["type"] != "lsp_workorder"

    def _to_list(s):
        if isinstance(s, np.ndarray): return s.tolist()
        return s if isinstance(s, list) else []

    states_l   = states.map(_to_list)
    has_states = states_l.map(bool)
    has_ru     = has_states & states_l.map(lambda s: "RU" in s)
    has_rd     = has_states & states_l.map(lambda s: "RD" in s)

    # Last SC or FA in the states list (ops are not always in chronological order,
    # so scan reversed to find the most-recently-appended terminal state).
    def _last_terminal(s):
        for v in reversed(s):
            if v in ("SC", "FA"):
                return v
        return None

    last_term  = states_l.map(_last_terminal)
    has_sc     = has_states & is_not_lsp & (last_term == "SC")
    has_fa     = has_states & is_not_lsp & (last_term == "FA")
    wo_valid   = ~wo.isin(_bad)
    req_valid  = ~req.isin(_bad)

    # CANCELED wo_status always wins — ops may have FA states but the
    # workorder was explicitly canceled, so don't let ops override it.
    # np.select applies conditions in order; first match wins per row.
    df["visual_status"] = np.select(
        [
            wo == "CANCELED",
            wo == "DRAFT",
            has_ru,
            has_rd,
            has_sc,
            has_fa,
            wo_valid,
            req_valid,
            df["data_source"] == "SYNTHETIC_LSP",
        ],
        [
            "CANCELED", "DRAFT", "RUNNING", "READY",
            "SUCCEEDED", "FAILED",
            wo, req, "UNKNOWN",
        ],
        default="IN_PROGRESS",
    )

    nan_mask = df["wo_status"].isna() | df["wo_status"].astype(str).str.upper().isin(["NAN", "NONE", ""])
    df.loc[nan_mask, "wo_status"] = df.loc[nan_mask, "visual_status"]

    log.info("Filtering & enrichment complete: %d rows ready for render", len(df))
    return df
