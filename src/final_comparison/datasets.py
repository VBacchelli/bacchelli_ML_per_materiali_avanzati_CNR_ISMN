"""Load cleaned polymer cohorts from MongoDB without modifying the source data."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.error")

DB_NAME = "PolymerPrediction"
COLLECTIONS = {"bicerano": "biceranoPolymers", "polyinfo": "polyInfo_kaggle", "extended": "extendedPolymers"}
MD_COLUMNS = {
    "biceranoPolymers_analysis_summary_base.csv": ("end_to_end_md", "Rg_md", "density_md"),
    "biceranoPolymers_analysis_summary_250.csv": ("end_to_end_md_250", "Rg_md_250", "density_md_250"),
    "biceranoPolymers_analysis_summary_350.csv": ("end_to_end_md_350", "Rg_md_350", "density_md_350"),
    "single_chain.csv": ("Rg", "tau", "trans", "gauche", "cis", "mean_angle", "var_dihedral", "circular_var"),
}


def canonical_smiles(smiles):
    """Return canonical RDKit SMILES, or None if parsing fails."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True) if mol else None


def clean_frame(frame, dataset, *, target="Tg", check_counts=True):
    """Apply the same structure and duplicate rules to the selected target."""
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    required = {"sample_id", "smiles"}
    required.add("Tg_K" if target == "Tg" and "target_value" not in frame else "target_value")
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = frame.copy().reset_index(drop=True)
    df["sample_id"] = df.sample_id.astype(str)
    if df.sample_id.duplicated().any():
        raise ValueError("Duplicate sample_id in source data")

    source = "target_value" if "target_value" in df else "Tg_K"
    df["target_value"] = pd.to_numeric(df[source], errors="coerce")
    if target == "Tg":
        df["Tg_K"] = df.target_value
    df["canonical_smiles"] = df.smiles.map(canonical_smiles)
    df["decision"] = "keep"

    # Target-specific cleaning happens before model preprocessing or data splits.
    df.loc[df.smiles.astype("string").str.contains(r"\[\[", na=False), "decision"] = "exclude_double_bracket_smiles"
    df.loc[df.canonical_smiles.isna() & df.decision.eq("keep"), "decision"] = "exclude_rdkit_invalid"
    df.loc[df.target_value.isna(), "decision"] = f"exclude_missing_{target}"
    df.loc[df.target_value.notna() & ~np.isfinite(df.target_value), "decision"] = f"exclude_nonfinite_{target}"

    valid = df.decision.eq("keep")
    conflicting = df.loc[valid].groupby("canonical_smiles").target_value.nunique().gt(1)
    df.loc[valid & df.canonical_smiles.isin(conflicting[conflicting].index), "decision"] = f"exclude_conflicting_{target}_group"

    # If multiple samples have the same structure and target, keep one by ID.
    remaining = df.loc[df.decision.eq("keep")].sort_values("sample_id")
    df.loc[remaining.index[remaining.duplicated("canonical_smiles")], "decision"] = f"exclude_identical_{target}_duplicate"

    kept = df.loc[df.decision.eq("keep")].drop(columns="decision").sort_values("sample_id").reset_index(drop=True)
    excluded = df.loc[df.decision.ne("keep")].sort_values("sample_id").reset_index(drop=True)
    if kept.canonical_smiles.isna().any() or kept.canonical_smiles.duplicated().any() or not np.isfinite(kept.target_value).all():
        raise RuntimeError("Cleaning left invalid or duplicate observations")
    return kept, excluded


def common_md_ids(md_dir):
    """Find samples with complete finite bulk and single-chain MD descriptors."""
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


def load_cohort(dataset, *, target="Tg", md_dir=None, client=None, db_name=DB_NAME, check_counts=True):
    """Load a cleaned target cohort; optionally require complete MD descriptors."""
    if dataset not in COLLECTIONS:
        raise ValueError(f"Unknown dataset: {dataset}")
    if not isinstance(target, str) or not target.isidentifier():
        raise ValueError("target must be a property name (e.g. Tg or density)")

    if client is None:
        from dotenv import load_dotenv
        from pymongo import MongoClient
        load_dotenv()
        uri = os.getenv("MONGO_URI")
        if not uri:
            raise EnvironmentError("Set MONGO_URI in .env or environment")
        with MongoClient(uri) as mongo:
            return load_cohort(dataset, target=target, md_dir=md_dir, client=mongo, db_name=db_name, check_counts=check_counts)

    fields = {"smiles": 1, "polymer_name": 1, f"properties.{target}": 1}
    docs = client[db_name][COLLECTIONS[dataset]].find({}, fields)
    frame = pd.DataFrame(({
        "sample_id": str(doc["_id"]), "smiles": doc.get("smiles"),
        "polymer_name": doc.get("polymer_name"),
        "target_value": (doc.get("properties") or {}).get(target),
    } for doc in docs), columns=["sample_id", "smiles", "polymer_name", "target_value"])
    if target == "Tg":
        frame["Tg_K"] = frame.target_value
    kept, excluded = clean_frame(frame, dataset, target=target, check_counts=check_counts)

    if md_dir is not None:
        available = kept.sample_id.isin(common_md_ids(md_dir))
        excluded = pd.concat([
            excluded, kept.loc[~available].assign(decision="exclude_md_unavailable_or_nonfinite")
        ], ignore_index=True).sort_values("sample_id").reset_index(drop=True)
        kept = kept.loc[available].reset_index(drop=True)

    return kept.assign(**{target: kept.target_value}), excluded
