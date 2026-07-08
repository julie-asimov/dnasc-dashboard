"""
Tests for antibiotic mismatch detection — pipeline._norm_bios_ab, _lims_ab_set,
and the mismatch flag computed in run_pipeline Step 9.
"""
import numpy as np
import pandas as pd
import pytest

from dnasc.pipeline import _norm_bios_ab, _lims_ab_set


# ── _norm_bios_ab ─────────────────────────────────────────────────────────────

class TestNormBiosAb:
    def test_kanamycin_variants(self):
        assert _norm_bios_ab("KANAMYCIN") == "Kan"
        assert _norm_bios_ab("Kanamycin") == "Kan"
        assert _norm_bios_ab("kan")       == "Kan"
        assert _norm_bios_ab("KAN")       == "Kan"

    def test_spectinomycin_variants(self):
        assert _norm_bios_ab("SPECTINOMYCIN") == "Spec"
        assert _norm_bios_ab("Spectinomycin") == "Spec"
        assert _norm_bios_ab("spec")          == "Spec"
        assert _norm_bios_ab("SPEC")          == "Spec"

    def test_carb_variants(self):
        assert _norm_bios_ab("CARBENICILLIN") == "Carb"
        assert _norm_bios_ab("Carbenicillin") == "Carb"
        assert _norm_bios_ab("carb")          == "Carb"
        assert _norm_bios_ab("AMPICILLIN")    == "Carb"
        assert _norm_bios_ab("amp")           == "Carb"

    def test_none_and_nan(self):
        assert _norm_bios_ab(None)       is None
        assert _norm_bios_ab(float("nan")) is None
        assert _norm_bios_ab(np.nan)     is None

    def test_empty_string(self):
        assert _norm_bios_ab("") is None

    def test_unrecognised_returns_none(self):
        assert _norm_bios_ab("Chloramphenicol") is None
        assert _norm_bios_ab("Zeocin")          is None


# ── _lims_ab_set ──────────────────────────────────────────────────────────────

class TestLimsAbSet:
    def test_anti_kan(self):
        assert _lims_ab_set({"lims_anti_kan": True,  "lims_anti_spec": False, "lims_anti_carb": False}) == {"Kan"}

    def test_anti_spec(self):
        assert _lims_ab_set({"lims_anti_kan": False, "lims_anti_spec": True,  "lims_anti_carb": False}) == {"Spec"}

    def test_anti_carb(self):
        assert _lims_ab_set({"lims_anti_kan": False, "lims_anti_spec": False, "lims_anti_carb": True})  == {"Carb"}

    def test_all_false_returns_empty(self):
        assert _lims_ab_set({"lims_anti_kan": False, "lims_anti_spec": False, "lims_anti_carb": False}) == set()

    def test_all_missing_returns_empty(self):
        assert _lims_ab_set({}) == set()

    def test_pd_na_not_truthy(self):
        # pd.NA must not be treated as True (the original bug)
        assert _lims_ab_set({"lims_anti_kan": pd.NA, "lims_anti_spec": pd.NA, "lims_anti_carb": pd.NA}) == set()

    def test_none_not_truthy(self):
        assert _lims_ab_set({"lims_anti_kan": None, "lims_anti_spec": None, "lims_anti_carb": None}) == set()

    def test_two_markers_both_kept(self):
        # Genuine dual-marker plasmid (no neo) → both markers returned.
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": True}) == {"Kan", "Spec", "Carb"}

    def test_neor_alias_drops_kan(self):
        # pAI-22250 case: NeoR marker in alias → anti_kan is mammalian; drop it, keep Spec.
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
                             "lims_plasmid_alias": "pL1_SV40WT_NeoR_SV40pA_U1U2"}) == {"Spec"}

    def test_neo_token_alias_drops_kan(self):
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
                             "lims_plasmid_alias": "pMab SV40-Neo; hEF1a-LC"}) == {"Spec"}

    def test_neomycin_alias_drops_kan(self):
        # 'Neomycin' spelled out is still the neo marker → drop Kan.
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
                             "lims_plasmid_alias": "pSomeVector-Neomycin"}) == {"Spec"}

    def test_neo_alias_kan_only_returns_empty(self):
        # Neo alias + only anti_kan set → no verifiable bacterial antibiotic.
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
                             "lims_plasmid_alias": "pL2-SB SV40-Neo"}) == set()

    def test_no_neo_alias_kan_still_fires(self):
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
                             "lims_plasmid_alias": "pUC19"}) == {"Kan"}

    def test_mneongreen_not_treated_as_neo(self):
        # mNeonGreen is a fluorescent protein, NOT neomycin — Kan must be kept.
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
                             "lims_plasmid_alias": "pL2-mNeonGreen-Puro"}) == {"Kan"}

    def test_neo_matched_in_construct_name(self):
        # Neo marker present in construct_name (not alias) still drops Kan.
        assert _lims_ab_set({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
                             "construct_name": "pL1_NeoR_cassette"}) == {"Spec"}


# ── mismatch flag logic (inline simulation of pipeline Step 9) ────────────────

def _apply_mismatch(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame and apply the same mismatch logic as the pipeline."""
    df = pd.DataFrame(rows)
    lims_sets = df.apply(_lims_ab_set, axis=1)
    df["lims_antibiotic"]    = lims_sets.apply(lambda s: ", ".join(sorted(s)) if s else None)
    df["lims_double_marker"] = lims_sets.apply(lambda s: len(s) >= 2)
    bios_norm = df["antibiotic"].apply(_norm_bios_ab)
    df["antibiotic_mismatch"] = [
        (b is not None) and bool(s) and (b not in s)
        for b, s in zip(bios_norm, lims_sets)
    ]
    return df


class TestMismatchFlag:
    def test_kan_vs_spec_flags(self):
        df = _apply_mismatch([{
            "antibiotic": "KANAMYCIN",
            "lims_anti_kan": False, "lims_anti_spec": True, "lims_anti_carb": False,
        }])
        assert df["antibiotic_mismatch"].iloc[0] is True or df["antibiotic_mismatch"].iloc[0] == True

    def test_spec_vs_spec_no_flag(self):
        df = _apply_mismatch([{
            "antibiotic": "SPECTINOMYCIN",
            "lims_anti_kan": False, "lims_anti_spec": True, "lims_anti_carb": False,
        }])
        assert not df["antibiotic_mismatch"].iloc[0]

    def test_kan_vs_kan_no_flag(self):
        df = _apply_mismatch([{
            "antibiotic": "Kanamycin",
            "lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
        }])
        assert not df["antibiotic_mismatch"].iloc[0]

    def test_null_bios_antibiotic_no_flag(self):
        df = _apply_mismatch([{
            "antibiotic": None,
            "lims_anti_kan": False, "lims_anti_spec": True, "lims_anti_carb": False,
        }])
        assert not df["antibiotic_mismatch"].iloc[0]

    def test_null_lims_flags_no_flag(self):
        df = _apply_mismatch([{
            "antibiotic": "KANAMYCIN",
            "lims_anti_kan": False, "lims_anti_spec": False, "lims_anti_carb": False,
        }])
        assert not df["antibiotic_mismatch"].iloc[0]

    def test_pd_na_lims_flags_no_flag(self):
        df = _apply_mismatch([{
            "antibiotic": "KANAMYCIN",
            "lims_anti_kan": pd.NA, "lims_anti_spec": pd.NA, "lims_anti_carb": pd.NA,
        }])
        assert not df["antibiotic_mismatch"].iloc[0]

    def test_neor_dual_marker_correct_selection_no_flag(self):
        # pAI-22250: LIMS has Kan+Spec, but Kan is the NeoR mammalian marker.
        # BIOS correctly picked Spec → no mismatch, and no double-marker note (Kan dropped).
        df = _apply_mismatch([{
            "antibiotic": "SPECTINOMYCIN",
            "lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
            "lims_plasmid_alias": "pL1_SV40WT_NeoR_SV40pA_U1U2",
        }])
        assert not df["antibiotic_mismatch"].iloc[0]
        assert not df["lims_double_marker"].iloc[0]
        assert df["lims_antibiotic"].iloc[0] == "Spec"

    def test_genuine_dual_marker_correct_selection_notes_but_no_mismatch(self):
        # Real dual-marker plasmid (no neo). BIOS matches one of them → no mismatch,
        # but the informational two-marker flag is set.
        df = _apply_mismatch([{
            "antibiotic": "KANAMYCIN",
            "lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
            "lims_plasmid_alias": "pDualMarker",
        }])
        assert not df["antibiotic_mismatch"].iloc[0]
        assert df["lims_double_marker"].iloc[0]
        assert df["lims_antibiotic"].iloc[0] == "Kan, Spec"

    def test_dual_marker_bios_matches_neither_flags_mismatch(self):
        # Two markers Kan+Spec, BIOS says Carb → genuine mismatch.
        df = _apply_mismatch([{
            "antibiotic": "CARBENICILLIN",
            "lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
            "lims_plasmid_alias": "pDualMarker",
        }])
        assert df["antibiotic_mismatch"].iloc[0]

    def test_multiple_rows_mixed(self):
        df = _apply_mismatch([
            {"antibiotic": "KANAMYCIN",    "lims_anti_kan": False, "lims_anti_spec": True,  "lims_anti_carb": False},
            {"antibiotic": "SPECTINOMYCIN","lims_anti_kan": False, "lims_anti_spec": True,  "lims_anti_carb": False},
            {"antibiotic": "KANAMYCIN",    "lims_anti_kan": True,  "lims_anti_spec": False, "lims_anti_carb": False},
            {"antibiotic": None,           "lims_anti_kan": False, "lims_anti_spec": True,  "lims_anti_carb": False},
        ])
        flags = df["antibiotic_mismatch"].tolist()
        assert flags[0] == True   # Kan vs Spec → mismatch
        assert flags[1] == False  # Spec vs Spec → ok
        assert flags[2] == False  # Kan vs Kan → ok
        assert flags[3] == False  # null BIOS → no flag
