"""
Regression tests for ProcessingTransformer._compute_chain_status.

An assembly attempt's verdict is the best status across the assembly workorder
and its downstream children — NOT the assembly's own wo status. A Gibson/GG can
be CANCELED or FAILED while a child transformation produced a seq-confirmed
colony (= SUCCEEDED). This `chain_status` column is the single source of truth
both dashboard tabs (tracking + colony) read, so they can never disagree.

Reported case (pAI-21543): Gibson a1763876 was CANCELED but its transformation
0ba2439c SUCCEEDED — the design verdict must be SUCCEEDED, not CANCELED.
"""
import pandas as pd

from dnasc.transformers.processing import ProcessingTransformer as P


def _chain(rows):
    out = P._compute_chain_status(pd.DataFrame(rows))
    return dict(zip(out["workorder_id"].astype(str), out["chain_status"]))


def test_canceled_gibson_with_succeeded_transformation_rolls_up():
    cs = _chain([
        {"workorder_id": "g1", "type": "gibson_workorder", "visual_status": "CANCELED",
         "source_asm_process_id": None},
        {"workorder_id": "t1", "type": "transformation_workorder", "visual_status": "SUCCEEDED",
         "source_asm_process_id": "g1"},
        {"workorder_id": "t2", "type": "transformation_workorder", "visual_status": "FAILED",
         "source_asm_process_id": "g1"},
    ])
    assert cs["g1"] == "SUCCEEDED"


def test_failed_gibson_failed_transformation_stays_failed():
    cs = _chain([
        {"workorder_id": "g1", "type": "gibson_workorder", "visual_status": "FAILED",
         "source_asm_process_id": None},
        {"workorder_id": "t1", "type": "transformation_workorder", "visual_status": "FAILED",
         "source_asm_process_id": "g1"},
    ])
    assert cs["g1"] == "FAILED"


def test_two_attempts_resolve_independently():
    # Mirrors pAI-21543: attempt 1 fully failed, attempt 2 canceled-but-succeeded.
    cs = _chain([
        {"workorder_id": "att1", "type": "gibson_workorder", "visual_status": "FAILED",
         "source_asm_process_id": None},
        {"workorder_id": "att2", "type": "gibson_workorder", "visual_status": "CANCELED",
         "source_asm_process_id": None},
        {"workorder_id": "x1", "type": "transformation_workorder", "visual_status": "FAILED",
         "source_asm_process_id": "att1"},
        {"workorder_id": "x2", "type": "transformation_workorder", "visual_status": "SUCCEEDED",
         "source_asm_process_id": "att2"},
    ])
    assert cs["att1"] == "FAILED"
    assert cs["att2"] == "SUCCEEDED"


def test_running_transformation_beats_canceled_assembly():
    cs = _chain([
        {"workorder_id": "g1", "type": "gibson_workorder", "visual_status": "CANCELED",
         "source_asm_process_id": None},
        {"workorder_id": "t1", "type": "transformation_workorder", "visual_status": "RUNNING",
         "source_asm_process_id": "g1"},
    ])
    assert cs["g1"] == "RUNNING"


def test_assembly_with_no_children_uses_own_status():
    cs = _chain([
        {"workorder_id": "g1", "type": "gibson_workorder", "visual_status": "RUNNING",
         "source_asm_process_id": None},
    ])
    assert cs["g1"] == "RUNNING"


def test_uuid_extracted_from_decorated_parent_ref():
    # Parent refs sometimes arrive with prefixes/suffixes around the UUID.
    cs = _chain([
        {"workorder_id": "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", "type": "gibson_workorder",
         "visual_status": "FAILED", "source_asm_process_id": None},
        {"workorder_id": "t1", "type": "transformation_workorder", "visual_status": "SUCCEEDED",
         "source_asm_process_id": "process:aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee/step"},
    ])
    assert cs["aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"] == "SUCCEEDED"


def test_non_assembly_rows_get_null_chain_status():
    cs = _chain([
        {"workorder_id": "g1", "type": "gibson_workorder", "visual_status": "SUCCEEDED",
         "source_asm_process_id": None},
        {"workorder_id": "t1", "type": "transformation_workorder", "visual_status": "SUCCEEDED",
         "source_asm_process_id": "g1"},
    ])
    assert cs["g1"] == "SUCCEEDED"
    assert pd.isna(cs["t1"])  # chain_status only populated on assembly rows
