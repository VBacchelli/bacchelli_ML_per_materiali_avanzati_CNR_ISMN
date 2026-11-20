"""Freeze and verify shared, sample-ID-based evaluation splits."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

SEED = 28
N_FOLDS = 5
VALIDATION_FRACTION = 0.125
SCHEMA_VERSION = 1


def _digest(data):
    payload = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path):
    """Hash a source file without loading it as model input."""
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _samples(frame, target):
    value_col = "Tg_K" if target == "Tg" else "target_value"
    required = {"sample_id", "smiles", "canonical_smiles", value_col}
    if missing := required - set(frame.columns):
        raise ValueError(f"Missing cleaned columns: {sorted(missing)}")

    df = frame[list(required)].copy()
    df["sample_id"] = df.sample_id.astype(str)
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")
    if df.sample_id.duplicated().any() or df.canonical_smiles.duplicated().any():
        raise ValueError("Duplicate sample ID or canonical structure in cleaned cohort")
    if df.smiles.isna().any() or df.canonical_smiles.isna().any() or not np.isfinite(df[value_col]).all():
        raise ValueError("Invalid structures or target values in cleaned cohort")

    df = df.sort_values("sample_id").reset_index(drop=True)
    records = [
        {
            "sample_id": row.sample_id,
            "fingerprint": _digest({
                "sample_id": row.sample_id,
                "smiles": row.smiles,
                "canonical_smiles": row.canonical_smiles,
                "target": target,
                "value": float(getattr(row, value_col)).hex(),
            }),
        }
        for row in df.itertuples(index=False)
    ]
    return df, records


def create_manifest(frame, *, cohort, target="Tg", seed=SEED,
                    n_folds=N_FOLDS, validation_fraction=VALIDATION_FRACTION,
                    source=None, source_hashes=None, target_unit=None):
    """Assign one outer test fold per sample, plus fixed inner validations."""
    df, samples = _samples(frame, target)
    if len(df) < n_folds * 2:
        raise ValueError("Cohort too small for five outer and inner folds")
    if not (0 < validation_fraction < 1):
        raise ValueError("validation_fraction must be between 0 and 1")

    ids = df.sample_id.to_numpy()
    outer = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds = []
    for fold, (train_idx, test_idx) in enumerate(outer.split(ids), start=1):
        for index in test_idx:
            samples[int(index)]["outer_fold"] = fold

        train_ids = ids[train_idx]
        inner = KFold(n_splits=n_folds, shuffle=True, random_state=seed + fold)
        inner_val = [sorted(train_ids[val].tolist()) for _, val in inner.split(train_ids)]
        _, stop_val = train_test_split(
            train_ids, test_size=validation_fraction, random_state=seed + fold,
            shuffle=True,
        )
        folds.append({
            "fold": fold,
            "inner_validation_ids": inner_val,
            "early_stop_validation_ids": sorted(stop_val.tolist()),
        })

    document = {
        "schema_version": SCHEMA_VERSION,
        "cohort": cohort,
        "target": target,
        "target_unit": target_unit or ("K" if target == "Tg" else None),
        "source": source or {},
        "source_sha256": source_hashes or {},
        "seed": seed,
        "n_outer_folds": n_folds,
        "n_inner_folds": n_folds,
        "early_stop_validation_fraction": validation_fraction,
        "split_method": "shuffled_KFold_over_sorted_sample_ids",
        "dataset_sha256": _digest([
            {"sample_id": row["sample_id"], "fingerprint": row["fingerprint"]}
            for row in samples
        ]),
        "samples": samples,
        "folds": folds,
    }
    document["manifest_sha256"] = _digest(document)
    validate_manifest(frame, document)
    return document


def fold_ids(manifest, fold):
    """Return stable train/test and inner validation sample IDs."""
    entries = {row["sample_id"]: row["outer_fold"] for row in manifest["samples"]}
    if not 1 <= fold <= manifest["n_outer_folds"]:
        raise ValueError(f"Unknown outer fold: {fold}")
    train = sorted(sid for sid, test_fold in entries.items() if test_fold != fold)
    test = sorted(sid for sid, test_fold in entries.items() if test_fold == fold)
    config = manifest["folds"][fold - 1]
    return {
        "train_ids": train,
        "test_ids": test,
        "inner_validation_ids": config["inner_validation_ids"],
        "early_stop_validation_ids": config["early_stop_validation_ids"],
        "early_stop_train_ids": sorted(set(train) - set(config["early_stop_validation_ids"])),
    }


def validate_manifest(frame, manifest):
    """Fail closed if cleaned MongoDB data or any split has changed."""
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported manifest version")
    checksum = manifest.get("manifest_sha256")
    if checksum != _digest({k: v for k, v in manifest.items() if k != "manifest_sha256"}):
        raise ValueError("Manifest checksum mismatch")

    _, current = _samples(frame, manifest["target"])
    saved = [{k: row[k] for k in ("sample_id", "fingerprint")}
             for row in manifest["samples"]]
    if current != saved or manifest["dataset_sha256"] != _digest(saved):
        raise ValueError("Cleaned MongoDB cohort differs from frozen manifest")

    n_folds = manifest["n_outer_folds"]
    if len(manifest["folds"]) != n_folds:
        raise ValueError("Missing outer folds")
    if any(row["outer_fold"] not in range(1, n_folds + 1) for row in manifest["samples"]):
        raise ValueError("Invalid outer fold assignment")

    all_ids = {row["sample_id"] for row in manifest["samples"]}
    canonical = frame.set_index(frame.sample_id.astype(str))["canonical_smiles"]
    for fold in range(1, n_folds + 1):
        if manifest["folds"][fold - 1]["fold"] != fold:
            raise ValueError("Fold numbering is inconsistent")
        split = fold_ids(manifest, fold)
        train, test = set(split["train_ids"]), set(split["test_ids"])
        if not test or not train or train & test or train | test != all_ids:
            raise ValueError("Incomplete or overlapping outer split")
        if set(canonical.loc[list(train)]) & set(canonical.loc[list(test)]):
            raise ValueError("Structural leakage between train and test")

        inner_groups = split["inner_validation_ids"]
        if len(inner_groups) != manifest["n_inner_folds"]:
            raise ValueError("Missing inner folds")
        flat_inner = [sid for group in inner_groups for sid in group]
        if not all(inner_groups) or set(flat_inner) != train or len(flat_inner) != len(train):
            raise ValueError("Invalid inner CV assignments")
        stop_val = split["early_stop_validation_ids"]
        if not stop_val or len(stop_val) >= len(train) or not set(stop_val) <= train:
            raise ValueError("Early-stopping validation is outside outer train")
        if len(stop_val) != len(set(stop_val)):
            raise ValueError("Repeated early-stopping sample IDs")
    return True


def save_manifest(path, document):
    """Create once; repeated calls may verify but never replace a manifest."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(document, indent=2, ensure_ascii=False) + "\n"
    try:
        if document.get("manifest_sha256") != _digest({
            k: v for k, v in document.items() if k != "manifest_sha256"
        }):
            raise ValueError("Invalid manifest checksum")
        with path.open("x", encoding="utf-8") as handle:
            handle.write(content)
        return True
    except FileExistsError:
        existing = load_manifest(path)
        if existing != document:
            raise FileExistsError(f"Frozen manifest differs; do not overwrite: {path}")
        return False


def load_manifest(path, frame=None, source_paths=None):
    """Read a manifest and optionally validate data and MD input hashes."""
    with open(path, encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("manifest_sha256") != _digest({
        k: v for k, v in manifest.items() if k != "manifest_sha256"
    }):
        raise ValueError("Manifest checksum mismatch")
    if frame is not None:
        validate_manifest(frame, manifest)
    if source_paths is not None:
        current = {name: file_sha256(path) for name, path in source_paths.items()}
        if current != manifest["source_sha256"]:
            raise ValueError("MD source files differ from the frozen manifest")
    return manifest


def fold_summary(manifest):
    rows = []
    for fold in range(1, manifest["n_outer_folds"] + 1):
        split = fold_ids(manifest, fold)
        rows.append({
            "cohort": manifest["cohort"], "target": manifest["target"],
            "fold": fold, "train": len(split["train_ids"]),
            "early_stop_train": len(split["early_stop_train_ids"]),
            "early_stop_val": len(split["early_stop_validation_ids"]),
            "test": len(split["test_ids"]),
        })
    return pd.DataFrame(rows)
