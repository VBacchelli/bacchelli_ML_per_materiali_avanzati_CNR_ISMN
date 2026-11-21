"""Leakage-safe descriptor Gradient Boosting for frozen polymer splits."""

from itertools import combinations
from time import perf_counter

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline


PARAM_GRID = {
    "gbr__n_estimators": [100, 200, 400],
    "gbr__learning_rate": [0.01, 0.05, 0.1],
    "gbr__max_depth": [2, 3, 4],
    "gbr__subsample": [0.8, 1.0],
}


def featurize_structures(frame, use_mordred=False):
    """Compute structure-only features once; no target values are accessed."""
    names = [name for name, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(names)
    morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    columns = names + [f"fp_{i}" for i in range(1024)] + [f"maccs_{i}" for i in range(167)]
    mols = [Chem.MolFromSmiles(s) for s in frame.smiles]
    if any(mol is None for mol in mols):
        raise ValueError("The cleaned cohort contains an invalid SMILES")

    values = [
        list(calc.CalcDescriptors(mol))
        + list(morgan.GetFingerprint(mol))
        + list(MACCSkeys.GenMACCSKeys(mol))
        for mol in mols
    ]
    X = pd.DataFrame(values, index=frame.sample_id.astype(str), columns=columns)
    X.index.name = "sample_id"

    if use_mordred:
        from mordred import Calculator, descriptors

        mordred = Calculator(descriptors, ignore_3D=True).pandas(mols, nproc=1, quiet=True)
        mordred.index = X.index
        mordred.columns = [f"mrd_{name}" for name in mordred.columns]
        X = pd.concat([X, mordred.apply(pd.to_numeric, errors="coerce")], axis=1)

    if not X.columns.is_unique or not X.index.is_unique:
        raise ValueError("Duplicate descriptor names or sample IDs")
    return X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)


class FeatureFilter(BaseEstimator, TransformerMixin):
    """Filter and impute columns using only the current training fold."""

    def __init__(self, min_prev=0.02, max_nan=0.20):
        self.min_prev = min_prev
        self.max_nan = max_nan

    def fit(self, X, y=None):
        X = X.replace([np.inf, -np.inf], np.nan)
        binary = [c for c in X if c.startswith(("fp_", "maccs_"))]
        continuous = [c for c in X if c not in binary]
        prevalence = X[binary].mean()
        self.keep_bin_ = prevalence[prevalence.between(self.min_prev, 1 - self.min_prev)].index.tolist()
        cont = X[continuous].loc[:, X[continuous].isna().mean() <= self.max_nan]
        cont = cont.loc[:, cont.std(ddof=0) > 0]

        # The historical Mordred de-duplication must also use training rows only.
        rdkit = [c for c in cont if not c.startswith("mrd_")]
        signatures = {self._signature(cont[c]) for c in rdkit}
        self.keep_cont_ = []
        for col in cont:
            if col.startswith("mrd_"):
                original = col[4:]
                if original in cont:
                    both = cont[[col, original]].dropna()
                    if len(both) and np.allclose(both[col], both[original], rtol=1e-5, atol=1e-8):
                        continue
                if self._signature(cont[col]) in signatures:
                    continue
            self.keep_cont_.append(col)

        self.selected_ = self.keep_cont_ + self.keep_bin_
        if not self.selected_:
            raise ValueError("No features survived the training-fold filters")
        self.medians_ = X[self.keep_cont_].median()
        return self

    @staticmethod
    def _signature(series):
        return np.round(series.to_numpy(dtype=float), 6).tobytes()

    def transform(self, X):
        X = X[self.selected_].replace([np.inf, -np.inf], np.nan).copy()
        X[self.keep_cont_] = X[self.keep_cont_].fillna(self.medians_)
        X[self.keep_bin_] = X[self.keep_bin_].fillna(0)
        if not np.isfinite(X.to_numpy(dtype=float)).all():
            raise ValueError("Non-finite descriptors remain after imputation")
        return X


class MIPermutationSelector(BaseEstimator, TransformerMixin):
    """Select features within fit, never using GridSearchCV validation labels."""

    def __init__(self, n_mi=50, n_splits=5, n_repeats=10, seed=28):
        self.n_mi = n_mi
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed)
        discrete = np.asarray(X.columns.str.startswith(("fp_", "maccs_")))

        importance = pd.Series(0.0, index=X.columns)
        frequency = pd.Series(0, index=X.columns)

        for train, validation in cv.split(X):
            mi = mutual_info_regression(
                X.iloc[train], y[train],
                discrete_features=discrete,
                n_neighbors=5, random_state=self.seed,
            )
            top = X.columns[np.argsort(-mi, kind="stable")[:self.n_mi]]

            model = GradientBoostingRegressor(random_state=self.seed).fit(
                X.iloc[train][top], y[train]
            )
            scores = permutation_importance(
                model, X.iloc[validation][top], y[validation],
                scoring="neg_mean_absolute_error",
                n_repeats=self.n_repeats,
                random_state=self.seed,
            ).importances_mean

            importance.loc[top] += scores
            frequency.loc[top] += 1

        ranked = (
            importance.loc[frequency.gt(0)] / self.n_splits
        ).sort_values(ascending=False)

        self.selected_ = (
            ranked[ranked > ranked.mean()].index.tolist()
            or ranked.index[:1].tolist()
        )
        return self

    def transform(self, X):
        return X[self.selected_]


def _outer_fold_record(manifest, outer_fold):
    """Look up a one-based outer fold in the immutable Phase-02 manifest."""
    folds = manifest["folds"]
    if isinstance(folds, dict):
        try:
            return folds[str(outer_fold)] if str(outer_fold) in folds else folds[outer_fold]
        except KeyError as exc:
            raise ValueError(f"Missing saved outer fold {outer_fold}") from exc
    if not isinstance(folds, list) or len(folds) != manifest["n_outer_folds"]:
        raise ValueError("Unexpected frozen outer-fold structure")
    labeled = [fold for fold in folds if isinstance(fold, dict) and any(
        name in fold for name in ("outer_fold", "fold", "fold_id")
    )]
    if labeled:
        matches = [fold for fold in labeled if str(next(
            fold[key] for key in ("outer_fold", "fold", "fold_id") if key in fold
        )) == str(outer_fold)]
        if len(matches) != 1:
            raise ValueError(f"Cannot identify saved outer fold {outer_fold}")
        return matches[0]
    return folds[outer_fold - 1]


def _validation_ids(entry):
    """Read a saved validation-ID list without constructing new splits."""
    if isinstance(entry, dict):
        for key in ("validation_ids", "inner_validation_ids", "val_ids", "test_ids"):
            if key in entry:
                return entry[key]
        raise ValueError("Inner fold has no validation IDs")
    return entry


def frozen_inner_cv(manifest, outer_fold, train_ids):
    """Use stored inner validation IDs; infer each train set by complement."""
    train_ids = [str(sid) for sid in train_ids]
    if len(set(train_ids)) != len(train_ids):
        raise ValueError("Repeated outer-training sample IDs")
    record = _outer_fold_record(manifest, outer_fold)
    if not isinstance(record, dict) or "inner_validation_ids" not in record:
        raise ValueError(f"Outer fold {outer_fold} lacks frozen inner_validation_ids")
    saved = record["inner_validation_ids"]
    if isinstance(saved, dict):
        if set(map(str, saved)) != {str(i) for i in range(1, 6)}:
            raise ValueError("Expected five numbered saved inner validation folds")
        saved = [saved.get(str(i), saved.get(i)) for i in range(1, 6)]
    if not isinstance(saved, list) or len(saved) != 5:
        raise ValueError("Expected five frozen inner validation folds")

    id_to_pos = {sid: i for i, sid in enumerate(train_ids)}
    expected = set(id_to_pos)
    observed = []
    pairs = []
    for item in saved:
        val_ids = _validation_ids(item)
        if not isinstance(val_ids, (list, tuple)):
            raise ValueError("Inner validation fold must contain a list of sample IDs")
        val_ids = [str(sid) for sid in val_ids]
        val_set = set(val_ids)
        if not val_ids or len(val_set) != len(val_ids) or not val_set <= expected:
            raise ValueError("Invalid frozen inner validation IDs")
        observed.extend(val_ids)
        pairs.append((
            np.array([i for i, sid in enumerate(train_ids) if sid not in val_set], dtype=int),
            np.array([id_to_pos[sid] for sid in val_ids], dtype=int),
        ))
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Inner validations must cover outer training exactly once")
    return pairs


def frozen_outer_ids(manifest, outer_fold):
    """Read one-based outer folds from the Phase-02 sample records."""
    if outer_fold not in range(1, manifest["n_outer_folds"] + 1):
        raise ValueError(f"Invalid frozen outer fold: {outer_fold}")
    records = manifest["samples"]
    if any(record.get("outer_fold") not in range(1, manifest["n_outer_folds"] + 1)
           for record in records):
        raise ValueError("A sample has an invalid frozen outer-fold assignment")
    train = [str(item["sample_id"]) for item in records if item["outer_fold"] != outer_fold]
    test = [str(item["sample_id"]) for item in records if item["outer_fold"] == outer_fold]
    return train, test


def validate_frozen_splits(frame, manifest, *, fold_ids=None):
    """Validate all frozen outer and inner folds; do not create or modify splits.

    The optional Phase-02 fold_ids helper is cross-checked against the assignments
    embedded in samples. It may return either (train_ids, test_ids) or a mapping.
    """
    if manifest.get("n_outer_folds") != 5 or manifest.get("n_inner_folds") != 5:
        raise ValueError("Expected exactly five frozen outer and inner folds")
    ids = [str(item["sample_id"]) for item in manifest["samples"]]
    actual = frame.sample_id.astype(str).tolist()
    if len(ids) != len(set(ids)) or len(actual) != len(set(actual)) or set(ids) != set(actual):
        raise ValueError("Manifest sample IDs differ from the cleaned MongoDB cohort")

    observed_test = []
    verified = []
    expected = set(ids)
    for outer_fold in range(1, 6):
        train, test = frozen_outer_ids(manifest, outer_fold)
        a, b = set(train), set(test)
        if (not a or not b or len(a) != len(train) or len(b) != len(test)
                or a & b or a | b != expected):
            raise ValueError(f"Invalid stored outer partition {outer_fold}")

        if fold_ids is not None:
            actual_split = fold_ids(manifest, outer_fold)
            if isinstance(actual_split, dict):
                provider_train = actual_split["train_ids"]
                provider_test = actual_split["test_ids"]
            elif isinstance(actual_split, (tuple, list)) and len(actual_split) == 2:
                provider_train, provider_test = actual_split
            else:
                raise ValueError("Unexpected Phase-02 fold_ids output")
            if set(map(str, provider_train)) != a or set(map(str, provider_test)) != b:
                raise ValueError(f"Phase-02 fold_ids disagrees with outer fold {outer_fold}")

        frozen_inner_cv(manifest, outer_fold, train)
        observed_test.extend(test)
        verified.append((outer_fold, train, test))
    if len(observed_test) != len(ids) or set(observed_test) != expected:
        raise ValueError("Outer tests must cover the manifest exactly once")
    return ids, verified


def benchmark_gbr(frame, manifest, *, fold_ids=None, use_mordred=False, log=None,
                  param_grid=None, seed=28, cache_dir=None):
    """Run frozen 5x5 nested CV and return OOF predictions and fold diagnostics."""
    from joblib import Memory

    ids, stored_folds = validate_frozen_splits(frame, manifest, fold_ids=fold_ids)
    if seed != manifest.get("seed"):
        raise ValueError("The model seed must match the frozen manifest seed")
    frame = frame.set_index("sample_id", drop=False).loc[ids]
    X = featurize_structures(frame, use_mordred=use_mordred)
    y = frame.target_value.astype(float)
    if not np.isfinite(y.to_numpy()).all():
        raise ValueError("Non-finite target values")
    oof, fold_rows, selected_rows = [], [], []

    for outer_fold, tr, te in stored_folds:
        inner = frozen_inner_cv(manifest, outer_fold, tr)
        pipe = Pipeline([
            ("filter", FeatureFilter()),
            ("select", MIPermutationSelector(seed=seed)),
            ("gbr", GradientBoostingRegressor(random_state=seed)),
        ], memory=Memory(location=cache_dir, verbose=0) if cache_dir else None)
        search = GridSearchCV(
            pipe, param_grid or PARAM_GRID, scoring="neg_mean_absolute_error",
            cv=inner, n_jobs=1, refit=True,
        )
        started = perf_counter()
        search.fit(X.loc[tr], y.loc[tr])
        elapsed = perf_counter() - started
        prediction = search.predict(X.loc[te])
        picked = search.best_estimator_.named_steps["select"].selected_
        selected_rows.extend({"fold": outer_fold, "feature": f} for f in picked)
        oof.extend({"sample_id": sid, "fold": outer_fold,
                    "y_true": float(y.loc[sid]), "y_pred": float(p)}
                   for sid, p in zip(te, prediction))
        fold_rows.append({
            "fold": outer_fold, "n_train": len(tr), "n_test": len(te),
            "n_selected": len(picked), "inner_MAE": -search.best_score_,
            "MAE": mean_absolute_error(y.loc[te], prediction),
            "R2": r2_score(y.loc[te], prediction),
            "MAE_over_sigma": mean_absolute_error(y.loc[te], prediction) / y.std(ddof=1),
            "fit_seconds": elapsed, "best_params": str(search.best_params_),
        })

    predictions = pd.DataFrame(oof).sort_values("sample_id").reset_index(drop=True)
    folds = pd.DataFrame(fold_rows)
    selected = pd.DataFrame(selected_rows)
    if len(predictions) != len(ids) or set(predictions.sample_id) != set(ids) or predictions.sample_id.duplicated().any():
        raise ValueError("The OOF predictions do not cover the manifest exactly once")
    groups = [set(selected.loc[selected.fold.eq(k), "feature"]) for k in range(1, 6)]
    jaccard = np.mean([len(a & b) / len(a | b) if a | b else 1.0 for a, b in combinations(groups, 2)])
    summary = {
        "n_samples": len(ids), "n_outer": 5, "n_inner": 5,
        "MAE": float(mean_absolute_error(predictions.y_true, predictions.y_pred)),
        "R2": float(r2_score(predictions.y_true, predictions.y_pred)),
        "MAE_over_sigma": float(mean_absolute_error(predictions.y_true, predictions.y_pred) / y.std(ddof=1)),
        "fold_MAE_mean": float(folds.MAE.mean()),
        "fold_MAE_std": float(folds.MAE.std(ddof=1)),
        "selection_jaccard": float(jaccard),
        "fit_seconds": float(folds.fit_seconds.sum()),
    }
    if log is not None:
        log.dataset(n_polymers=len(ids), target_min=y.min(), target_max=y.max(),
                    target_mean=y.mean(), target_std=y.std(ddof=1))
        log.table("oof_predictions", predictions, index=False)
        log.table("nested_cv_folds", folds, index=False)
        log.table("selected_features", selected, index=False)
        counts = selected.feature.value_counts().rename_axis("feature").reset_index(name="n_folds")
        counts["frequency"] = counts.n_folds / 5
        log.table("selection_stability", counts, index=False)
        log.result("nested_cv", {
            **summary,
            # Existing RunLogger reads these fields when updating runs/index.csv.
            "base_MAE_mean": float(folds.MAE.mean()),
            "base_MAE_std": float(folds.MAE.std(ddof=1)),
            "base_R2_mean": float(folds.R2.mean()),
            "base_R2_std": float(folds.R2.std(ddof=1)),
        })
        log.result("selection_stability", {"jaccard_mean": jaccard})
    return predictions, folds, selected, summary
