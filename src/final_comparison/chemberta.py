"""Frozen ChemBERTa embeddings and leak-safe GBR comparison on Phase-02 splits."""

import hashlib
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
import pandas as pd
from joblib import Memory
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from .gbr import PARAM_GRID, frozen_inner_cv, validate_frozen_splits

MODEL_NAME = "seyonec/ChemBERTa-zinc-base-v1"
REPRESENTATIONS = ("full", "pca95")


def embedding_path(root, collection, variant="raw"):
    if variant not in ("raw", "clean"):
        raise ValueError(f"Unknown ChemBERTa variant: {variant}")
    return Path(root) / f"chemberta_{collection}_{variant}.csv"


def smiles_digest(frame, variant="raw"):
    """Fingerprint the exact sample-ID/SMILES mapping used for extraction."""
    if variant not in ("raw", "clean"):
        raise ValueError(f"Unknown ChemBERTa variant: {variant}")
    ordered = frame[["sample_id", "smiles"]].copy()
    ordered["sample_id"] = ordered.sample_id.astype(str)
    if ordered.sample_id.duplicated().any():
        raise ValueError("Duplicate sample IDs")
    ordered = ordered.sort_values("sample_id")
    data = [[sid, s if variant == "raw" else s.replace("[*]", "")]
            for sid, s in ordered.itertuples(index=False, name=None)]
    return hashlib.sha256(json.dumps(data, ensure_ascii=False,
                                      separators=(",", ":")).encode("utf-8")).hexdigest()


def extract_embeddings(frame, path, *, variant="raw", model_name=MODEL_NAME,
                       batch_size=32, device=None):
    """Extract mean-pooled embeddings without using the target; never overwrite cache."""
    import torch
    from transformers import AutoModel, AutoTokenizer

    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Existing embedding file will not be overwritten: {path}")
    if variant not in ("raw", "clean"):
        raise ValueError(f"Unknown variant: {variant}")
    if batch_size < 1 or frame.empty:
        raise ValueError("Positive batch size and nonempty input required")
    ordered = frame[["sample_id", "smiles"]].copy().sort_values("sample_id")
    if ordered.sample_id.astype(str).duplicated().any() or ordered.smiles.isna().any():
        raise ValueError("Invalid sample IDs or SMILES")
    texts = ordered.smiles.astype(str).tolist()
    if variant == "clean":
        texts = [s.replace("[*]", "") for s in texts]

    extraction_started = perf_counter()
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name).to(device).eval()
    chunks = []
    for start in range(0, len(texts), batch_size):
        inputs = tokenizer(texts[start:start + batch_size], return_tensors="pt",
                           padding=True, truncation=True)
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = encoder(**inputs).last_hidden_state
            mask = inputs["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(1) / mask.sum(1).clamp(min=1)
        chunks.append(pooled.float().cpu().numpy())
    values = np.concatenate(chunks)
    if not np.isfinite(values).all():
        raise ValueError("ChemBERTa generated nonfinite embeddings")
    result = pd.DataFrame(values, columns=[f"emb_{i}" for i in range(values.shape[1])])
    result.insert(0, "_id", ordered.sample_id.astype(str).to_numpy())
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".csv.part")
    if temp.exists():
        raise FileExistsError(f"Incomplete extraction exists: {temp}")
    try:
        result.to_csv(temp, index=False)
        temp.replace(path)
        meta = {
            "model_name": model_name,
            "model_revision": getattr(encoder.config, "_commit_hash", None),
            "pooling": "attention_mask_mean",
            "variant": variant,
            "smiles_sha256": smiles_digest(ordered, variant),
            "samples": len(ordered),
            "dimensions": int(values.shape[1]),
            "csv_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "extraction_seconds": perf_counter() - extraction_started,
        }
        path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    finally:
        if temp.exists():
            temp.unlink()
    return result, meta


def load_embeddings(path, ids, *, source_frame=None, variant="raw", model_name=MODEL_NAME):
    """Load cached embeddings, strictly align to manifest IDs, never drop samples."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing embedding file: {path}")
    table = pd.read_csv(path, dtype={"_id": str, "sample_id": str})
    key = "_id" if "_id" in table else "sample_id"
    if key not in table:
        raise ValueError(f"Missing _id or sample_id in {path}")
    if table[key].isna().any() or table[key].duplicated().any():
        raise ValueError("Missing or duplicate embedding sample IDs")
    cols = [c for c in table.columns if c != key]
    if not cols or any(c != f"emb_{i}" for i, c in enumerate(cols)):
        raise ValueError("Embedding columns must be consecutive emb_0 ... emb_N")
    indexed = table.set_index(key)
    requested = pd.Index(list(map(str, ids)), name="sample_id")
    if not requested.is_unique or requested.empty:
        raise ValueError("Invalid requested sample IDs")
    missing = requested.difference(indexed.index)
    if len(missing):
        raise ValueError(f"Embedding cache lacks {len(missing)} manifest IDs; first: {missing[:3].tolist()}")
    X = indexed.loc[requested, cols].apply(pd.to_numeric, errors="coerce")
    X.index = requested
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite embeddings in frozen cohort")
    meta_path = path.with_suffix(".meta.json")
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_name") != model_name or meta.get("variant") != variant:
            raise ValueError("Embedding cache model or variant mismatch")
        if meta.get("csv_sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError("Embedding CSV changed after extraction")
        if source_frame is not None and meta.get("smiles_sha256") != smiles_digest(source_frame, variant):
            raise ValueError("Cached embeddings were extracted from different source SMILES/IDs")
    else:
        warnings.warn(f"Legacy embedding CSV has no provenance metadata: {path}; verify its model/pooling manually",
                      UserWarning, stacklevel=2)
    return X


def benchmark_chemberta(frame, manifest, X, *, representation="full", fold_ids=None,
                         log=None, param_grid=None, seed=28, cache_dir=None):
    """Frozen 5x5 nested CV with optional training-fold PCA and OOF logging."""
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown ChemBERTa representation: {representation}")
    ids, stored_folds = validate_frozen_splits(frame, manifest, fold_ids=fold_ids)
    if seed != manifest.get("seed"):
        raise ValueError("The model seed must match the frozen manifest seed")
    if not X.index.is_unique or set(X.index.astype(str)) != set(ids):
        raise ValueError("Embedding ID coverage must match the frozen cohort exactly")
    X = X.loc[ids]
    if not np.isfinite(X.to_numpy(dtype=float)).all():
        raise ValueError("Nonfinite embedding values")
    frame = frame.set_index("sample_id", drop=False).loc[ids]
    y = frame.target_value.astype(float)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Nonfinite target values")
    grid = param_grid or {"regressor__" + k.removeprefix("gbr__"): v
                          for k, v in PARAM_GRID.items()}
    oof, fold_rows = [], []
    for outer_fold, tr, te in stored_folds:
        pipeline = Pipeline([
            ("reduce", "passthrough" if representation == "full" else
             PCA(n_components=0.95, svd_solver="full")),
            ("regressor", GradientBoostingRegressor(random_state=seed)),
        ], memory=Memory(location=cache_dir, verbose=0) if cache_dir else None)
        search = GridSearchCV(pipeline, grid, cv=frozen_inner_cv(manifest, outer_fold, tr),
                              scoring="neg_mean_absolute_error", n_jobs=4, refit=True, verbose=2)
        started = perf_counter()
        search.fit(X.loc[tr], y.loc[tr])
        elapsed = perf_counter() - started
        prediction = search.predict(X.loc[te])
        n_components = (X.shape[1] if representation == "full" else
                        search.best_estimator_.named_steps["reduce"].n_components_)
        mae = mean_absolute_error(y.loc[te], prediction)
        oof.extend({"sample_id": sid, "fold": outer_fold,
                    "y_true": float(y.loc[sid]), "y_pred": float(pred)}
                   for sid, pred in zip(te, prediction))
        fold_rows.append({
            "fold": outer_fold, "n_train": len(tr), "n_test": len(te),
            "n_components": int(n_components),
            "inner_MAE": -search.best_score_, "MAE": mae,
            "R2": r2_score(y.loc[te], prediction),
            "MAE_over_sigma": mae / y.std(ddof=1),
            "fit_seconds": elapsed, "best_params": str(search.best_params_),
        })
    predictions = pd.DataFrame(oof).sort_values("sample_id").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    if (len(predictions) != len(ids) or predictions.sample_id.duplicated().any()
            or set(predictions.sample_id) != set(ids)):
        raise ValueError("OOF coverage differs from frozen manifest")
    mae = mean_absolute_error(predictions.y_true, predictions.y_pred)
    summary = {
        "n_samples": len(ids), "n_outer": 5, "n_inner": 5,
        "MAE": float(mae), "R2": float(r2_score(predictions.y_true, predictions.y_pred)),
        "MAE_over_sigma": float(mae / y.std(ddof=1)),
        "fold_MAE_mean": float(folds.MAE.mean()),
        "fold_MAE_std": float(folds.MAE.std(ddof=1)),
        "fit_seconds": float(folds.fit_seconds.sum()),
        "n_components_mean": float(folds.n_components.mean()),
    }
    if log is not None:
        log.dataset(n_polymers=len(ids), target_min=y.min(), target_max=y.max(),
                    target_mean=y.mean(), target_std=y.std(ddof=1))
        log.table("oof_predictions", predictions, index=False)
        log.table("nested_cv_folds", folds, index=False)
        log.result("nested_cv", {
            **summary,
            "base_MAE_mean": float(folds.MAE.mean()),
            "base_MAE_std": float(folds.MAE.std(ddof=1)),
            "base_R2_mean": float(folds.R2.mean()),
            "base_R2_std": float(folds.R2.std(ddof=1)),
        })
    return predictions, folds, summary
