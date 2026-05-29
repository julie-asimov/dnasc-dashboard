"""
Tests for antibiotic mismatch detection — pipeline._norm_bios_ab, _lims_ab,
and the mismatch flag computed in run_pipeline Step 9.
"""
import numpy as np
import pandas as pd
import pytest

from dnasc.pipeline import _norm_bios_ab, _lims_ab


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


# ── _lims_ab ──────────────────────────────────────────────────────────────────

class TestLimsAb:
    def test_anti_kan(self):
        assert _lims_ab({"lims_anti_kan": True,  "lims_anti_spec": False, "lims_anti_carb": False}) == "Kan"

    def test_anti_spec(self):
        assert _lims_ab({"lims_anti_kan": False, "lims_anti_spec": True,  "lims_anti_carb": False}) == "Spec"

    def test_anti_carb(self):
        assert _lims_ab({"lims_anti_kan": False, "lims_anti_spec": False, "lims_anti_carb": True})  == "Carb"

    def test_all_false_returns_none(self):
        assert _lims_ab({"lims_anti_kan": False, "lims_anti_spec": False, "lims_anti_carb": False}) is None

    def test_all_missing_returns_none(self):
        assert _lims_ab({}) is None

    def test_pd_na_not_truthy(self):
        # pd.NA must not be treated as True (the original bug)
        assert _lims_ab({"lims_anti_kan": pd.NA, "lims_anti_spec": pd.NA, "lims_anti_carb": pd.NA}) is None

    def test_none_not_truthy(self):
        assert _lims_ab({"lims_anti_kan": None, "lims_anti_spec": None, "lims_anti_carb": None}) is None

    def test_priority_kan_over_spec(self):
        # Kan checked first — multiple True flags shouldn't happen in practice
        # but priority must be deterministic.
        assert _lims_ab({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": True}) == "Kan"

    def test_neo_alias_skips_kan(self):
        # Plasmid with Neo in alias → anti_kan is mammalian marker, not bacterial; use Spec instead.
        assert _lims_ab({"lims_anti_kan": True, "lims_anti_spec": True, "lims_anti_carb": False,
                         "lims_plasmid_alias": "pMab SV40-Neo; hEF1a-LC"}) == "Spec"

    def test_neo_alias_kan_only_returns_none(self):
        # Neo alias + only anti_kan set → no valid bacterial antibiotic identifiable.
        assert _lims_ab({"lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
                         "lims_plasmid_alias": "pL2-SB SV40-Neo"}) is None

    def test_no_neo_alias_kan_still_fires(self):
        assert _lims_ab({"lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
                         "lims_plasmid_alias": "pUC19"}) == "Kan"

    def test_neo_word_boundary_not_partial(self):
        # 'Neomycin' as full word should not suppress Kan (it's not the marker abbreviation).
        assert _lims_ab({"lims_anti_kan": True, "lims_anti_spec": False, "lims_anti_carb": False,
                         "lims_plasmid_alias": "pSomeVector-Neomycin"}) == "Kan"


# ── mismatch flag logic (inline simulation of pipeline Step 9) ────────────────

def _apply_mismatch(rows: list[dict]) -> pd.DataFrame:
    """Build a minimal DataFrame and apply the same mismatch logic as the pipeline."""
    df = pd.DataFrame(rows)
    df["lims_antibiotic"] = df.apply(_lims_ab, axis=1)
    bios_norm = df["antibiotic"].apply(_norm_bios_ab)
    df["antibiotic_mismatch"] = (
        df["lims_antibiotic"].notna()
        & bios_norm.notna()
        & (bios_norm != df["lims_antibiotic"])
    )
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
