"""Load cleaned polymer cohorts from MongoDB without changing the source data."""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")

DB_NAME = "PolymerPrediction"
COLLECTIONS = {"bicerano": "biceranoPolymers", "polyinfo": "polyInfo_kaggle"}
EXPECTED_COUNTS = {"bicerano": (315, 312), "polyinfo": (7284, 7077)}
EXPECTED_MD_COMMON = 135
MD_COLUMNS = {
    "biceranoPolymers_analysis_summary_base.csv": ("end_to_end_md", "Rg_md", "density_md"),
    "biceranoPolymers_analysis_summary_250.csv": ("end_to_end_md_250", "Rg_md_250", "density_md_250"),
    "biceranoPolymers_analysis_summary_350.csv": ("end_to_end_md_350", "Rg_md_350", "density_md_350"),
    "single_chain.csv": ("Rg", "tau", "trans", "gauche", "cis", "mean_angle", "var_dihedral", "circular_var"),
}


def normalize_star(smiles):
    """Match the wildcard normalization used by the original importer."""
    smiles = re.sub(r"\(\s*\*\s*\)", "[*]", smiles)
    smiles = smiles.replace("[Ce]", "[*]").replace("[Th]", "[*]")
    return re.sub(r"(?<!\[)\*(?!\])", "[*]", smiles)


def canonical_smiles(smiles):
    """Return the canonical RDKit SMILES, or None if parsing fails."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(normalize_star(smiles.strip()))
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def clean_frame(frame, dataset, *, check_counts=True):
    """Apply uniform SMILES/Tg rules and retain an auditable exclusion table."""
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    required = {"sample_id", "smiles", "Tg_K"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = frame.copy().reset_index(drop=True)
    df["sample_id"] = df.sample_id.astype(str)
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id in source data")

    df["Tg_K"] = pd.to_numeric(df.Tg_K, errors="coerce")
    df["canonical_smiles"] = df.smiles.map(canonical_smiles)
    df["decision"] = "keep"

    # Dataset-level checks run before any split or model preprocessing.
    df.loc[df.smiles.astype("string").str.contains(r"\[\[", na=False), "decision"] = "exclude_double_bracket_smiles"
    df.loc[df.canonical_smiles.isna() & df.decision.eq("keep"), "decision"] = "exclude_rdkit_invalid"
    df.loc[df.Tg_K.isna(), "decision"] = "exclude_missing_Tg"
    df.loc[df.Tg_K.notna() & ~np.isfinite(df.Tg_K), "decision"] = "exclude_nonfinite_Tg"

    valid = df.decision.eq("keep")
    conflicting = df.loc[valid].groupby("canonical_smiles").Tg_K.nunique().gt(1)
    df.loc[valid & df.canonical_smiles.isin(conflicting[conflicting].index), "decision"] = "exclude_conflicting_Tg_group"

    # Identical targets: choose one sample deterministically, never by Tg.
    remaining = df.loc[df.decision.eq("keep")].sort_values("sample_id")
    duplicate_rows = remaining.duplicated("canonical_smiles")
    df.loc[remaining.index[duplicate_rows], "decision"] = "exclude_identical_Tg_duplicate"

    kept = df.loc[df.decision.eq("keep")].drop(columns="decision").sort_values("sample_id").reset_index(drop=True)
    excluded = df.loc[df.decision.ne("keep")].sort_values("sample_id").reset_index(drop=True)

    if kept.canonical_smiles.isna().any() or kept.canonical_smiles.duplicated().any() or not np.isfinite(kept.Tg_K).all():
        raise RuntimeError("Cleaning left invalid or duplicate observations")
    if check_counts and (len(df), len(kept)) != EXPECTED_COUNTS[dataset]:
        raise RuntimeError(f"Unexpected {dataset} counts: {len(df)} raw, {len(kept)} cleaned")
    return kept, excluded


def common_md_ids(md_dir):
    """Select IDs with all required finite bulk and single-chain MD features."""
    ids_by_file = []
    for name, columns in MD_COLUMNS.items():
        path = Path(md_dir) / name
        table = pd.read_csv(path, dtype={"polymer_id": str})
        if missing := {"polymer_id", *columns} - set(table.columns):
            raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
        if table.polymer_id.dropna().duplicated().any():
            raise ValueError(f"Duplicate polymer_id in {path}")
        values = table[list(columns)].apply(pd.to_numeric, errors="coerce")
        mask = table.polymer_id.notna() & np.isfinite(values.to_numpy(dtype=float)).all(axis=1)
        ids_by_file.append(set(table.loc[mask, "polymer_id"]))
    return set.intersection(*ids_by_file)


def load_cohort(cohort, *, md_dir=None, client=None, db_name=DB_NAME, check_counts=True):
    """Return (cleaned samples, excluded samples) from the original MongoDB."""
    dataset = "bicerano" if cohort == "md_common" else cohort
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown cohort: {cohort}")

    if client is None:
        from dotenv import load_dotenv
        from pymongo import MongoClient
        load_dotenv()
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise EnvironmentError("Set MONGO_URI in .env or environment")
        with MongoClient(uri) as mongo:
            return load_cohort(cohort, md_dir=md_dir, client=mongo, db_name=db_name, check_counts=check_counts)

    fields = {"smiles": 1, "polymer_name": 1, "properties.Tg": 1}
    docs = client[db_name][COLLECTIONS[dataset]].find({}, fields)
    frame = pd.DataFrame({
        "sample_id": str(doc["_id"]), "smiles": doc.get("smiles"),
        "polymer_name": doc.get("polymer_name"),
        "Tg_K": (doc.get("properties") or {}).get("Tg"),
    } for doc in docs)
    kept, excluded = clean_frame(frame, dataset, check_counts=check_counts)

    if cohort == "md_common":
        if md_dir is None:
            raise ValueError("Pass md_dir for the MD common cohort")
        ids = common_md_ids(md_dir)
        unavailable = kept.loc[~kept.sample_id.isin(ids)].copy()
        unavailable["decision"] = "exclude_md_unavailable_or_nonfinite"
        excluded = pd.concat([excluded, unavailable], ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        kept = kept.loc[kept.sample_id.isin(ids)].reset_index(drop=True)
        if check_counts and len(kept) != EXPECTED_MD_COMMON:
            raise RuntimeError(f"Expected {EXPECTED_MD_COMMON} MD samples, got {len(kept)}")

    return kept.assign(Tg=kept.Tg_K), excluded
