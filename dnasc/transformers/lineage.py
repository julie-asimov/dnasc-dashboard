"""
dnasc/transformers/lineage.py
──────────────────────────────
Bridges LSP lineage with BIOS workorders.
Prevents "kidnapping" — if an LSP belongs to a different request than
its source workorder, the lineage chain is broken and it becomes its own root.
"""

from __future__ import annotations

import pandas as pd

from dnasc.logger import get_logger

log = get_logger(__name__)


class LineageTransformer:
    """Combine BIOS and LSP data with context-aware lineage linking."""

    @staticmethod
    def bridge_lsp_lineage(bios_df: pd.DataFrame, lsp_full_df: pd.DataFrame) -> pd.DataFrame:
        """
        Concatenate BIOS and LSP DataFrames, then resolve root_work_order_id
        for each LSP row using smart lineage rules.

        Rules
        ─────
        - If the LSP has no parent in the dataset → it is its own root.
        - If the LSP's parent belongs to a *different* request → break the
          chain (anti-kidnapping). The LSP becomes its own root.
        - Otherwise → inherit the parent's root.
        """
        log.info("Bridging LSP lineage (smart mode)...")
        combined = pd.concat([bios_df, lsp_full_df], ignore_index=True)

        non_lsp = combined["type"] != "lsp_workorder"

        # id → root mapping (non-LSP rows only)
        id_to_root: dict[str, str] = (
            combined.loc[non_lsp]
            .set_index("workorder_id")["root_work_order_id"]
            .to_dict()
        )

        # root → req_id mapping (non-LSP rows only)
        root_to_req: dict[str, str] = (
            combined.loc[non_lsp]
            .set_index("root_work_order_id")["req_id"]
            .to_dict()
        )

        def _resolve_root(row: pd.Series):
            if row["type"] != "lsp_workorder":
                return row["root_work_order_id"]

            source_id = row.get("source_lsp_process_id")

            # No parent → self is root
            if pd.isna(source_id) or source_id not in id_to_root:
                return row["workorder_id"]

            parent_root = id_to_root[source_id]
            parent_req  = root_to_req.get(parent_root)
            my_req      = row.get("req_id")

            # Anti-kidnapping: different requests → break chain
            if (
                pd.notna(my_req)
                and pd.notna(parent_req)
                and str(my_req) != str(parent_req)
            ):
                return row["workorder_id"]

            return parent_root

        combined["root_work_order_id"] = combined.apply(_resolve_root, axis=1)
        combined["root_work_order_id"] = combined["root_work_order_id"].fillna(
            combined["workorder_id"]
        )

        # Store the intermediate link for reference
        lsp_mask = combined["type"] == "lsp_workorder"
        combined.loc[lsp_mask, "middle_root"] = combined.loc[
            lsp_mask, "source_lsp_process_id"
        ]

        log.info("Combined dataset: %d rows", len(combined))
        return combined
