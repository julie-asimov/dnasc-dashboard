"""
_reroot_synthetic_picks: agar-derived synthetic rows join their assembly group.

A pick logged off a transformation's agar plate is created at Step 4, before
roots collapse onto the assembly design, so it kept the transformation as its
own root and split into a separate workflow group. A pick off a Gibson's agar
looked fine only because a Gibson is already its own root.
"""
import pandas as pd

from dnasc.pipeline import _reroot_synthetic_picks


GIBSON = "gib-0000-0000-0000-000000000000"
TRANSF = "tfm-0000-0000-0000-000000000000"


def _df(rows):
    return pd.DataFrame(rows)


def _base():
    """A Gibson, and a transformation rooted on it."""
    return [
        {"workorder_id": GIBSON, "type": "gibson_workorder",
         "root_work_order_id": GIBSON, "source_asm_process_id": None},
        {"workorder_id": TRANSF, "type": "transformation_workorder",
         "root_work_order_id": GIBSON, "source_asm_process_id": GIBSON},
    ]


class TestRerootSyntheticPicks:

    def test_pick_off_transformation_moves_to_assembly_root(self):
        df = _df(_base() + [
            {"workorder_id": "PICK_x_well1", "type": "optracker_operation",
             "root_work_order_id": TRANSF, "source_asm_process_id": TRANSF},
        ])
        out = _reroot_synthetic_picks(df)
        assert out.loc[out.workorder_id == "PICK_x_well1",
                       "root_work_order_id"].iloc[0] == GIBSON

    def test_pick_off_gibson_is_unchanged(self):
        """Already correct — a Gibson is its own root."""
        df = _df(_base() + [
            {"workorder_id": "PICK_x_well2", "type": "optracker_operation",
             "root_work_order_id": GIBSON, "source_asm_process_id": GIBSON},
        ])
        out = _reroot_synthetic_picks(df)
        assert out.loc[out.workorder_id == "PICK_x_well2",
                       "root_work_order_id"].iloc[0] == GIBSON

    def test_deliberately_rerooted_row_is_left_alone(self):
        """
        Streakout resolution re-roots some synthetic rows on purpose. Those have
        a root that is NOT their source, and must not be dragged back.
        """
        other = "oth-0000-0000-0000-000000000000"
        df = _df(_base() + [
            {"workorder_id": "STREAK_well9", "type": "streakout_operation",
             "root_work_order_id": other, "source_asm_process_id": TRANSF},
        ])
        out = _reroot_synthetic_picks(df)
        assert out.loc[out.workorder_id == "STREAK_well9",
                       "root_work_order_id"].iloc[0] == other

    def test_real_workorders_never_move(self):
        df = _df(_base())
        out = _reroot_synthetic_picks(df)
        assert out.loc[out.workorder_id == TRANSF,
                       "root_work_order_id"].iloc[0] == GIBSON

    def test_unknown_source_is_left_alone(self):
        """Source not present in the frame — nothing to resolve against."""
        df = _df(_base() + [
            {"workorder_id": "PICK_x_well3", "type": "optracker_operation",
             "root_work_order_id": "ghost", "source_asm_process_id": "ghost"},
        ])
        out = _reroot_synthetic_picks(df)
        assert out.loc[out.workorder_id == "PICK_x_well3",
                       "root_work_order_id"].iloc[0] == "ghost"

    def test_missing_column_returns_frame_unchanged(self):
        df = pd.DataFrame([{"workorder_id": GIBSON, "type": "gibson_workorder",
                            "root_work_order_id": GIBSON}])
        out = _reroot_synthetic_picks(df)
        assert out["root_work_order_id"].iloc[0] == GIBSON
