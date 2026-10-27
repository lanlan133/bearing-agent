"""Offline, device-disjoint experiments; independent of deployment imports.

The envelope branch describes *spectral* envelopes, not time-domain envelopes.
All learned preprocessing/resampling is fitted inside the relevant training fold.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss,
    confusion_matrix, f1_score, log_loss, precision_score, recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES = tuple(f"f{i}" for i in range(1, 18))
ENVELOPE_FEATURES = tuple(f"f{i}" for i in range(10, 15))
SPECTRUM_FEATURES = tuple(f for f in FEATURES if f not in ENVELOPE_FEATURES)
BRANCH_COLUMNS = ("p_envelope", "p_spectrum", "p_full")
LABEL_MAP = {"非轴承润滑不良": 0, "轴承润滑不良": 1}
LABEL_NAMES = {value: key for key, value in LABEL_MAP.items()}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    sampling_rate: float = 10240.0
    test_size: float = 0.2
    folds: int = 5
    inner_folds: int = 3
    trees: int = 300
    jobs: int = 2
    threshold: float = 0.5


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                               allow_nan=False), encoding="utf-8")


def feature_matrix(frame: pd.DataFrame, names=FEATURES) -> np.ndarray:
    missing = set(names) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    values = frame.loc[:, list(names)].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Features contain nonnumeric, NaN or infinite values")
    return values


def load_experiment_data(path: Path, sampling_rate: float):
    raw = pd.read_csv(path)
    required = {*FEATURES, "sample_index", "equipment_id", "label", "sampling_rate"}
    if required - set(raw.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(raw.columns))}")
    rates = pd.to_numeric(raw["sampling_rate"], errors="coerce")
    frame = raw.loc[np.isclose(rates, sampling_rate, rtol=0, atol=1e-6)].copy()
    if frame.empty:
        raise ValueError(f"No samples at {sampling_rate} Hz")
    if frame["equipment_id"].isna().any():
        raise ValueError("Missing equipment_id; cannot establish group boundaries")
    frame["equipment_id"] = frame["equipment_id"].astype(str).str.strip()
    if frame["equipment_id"].str.lower().isin(["", "nan", "none", "[]"]).any():
        raise ValueError("Empty or placeholder equipment_id")
    ids = pd.to_numeric(frame["sample_index"], errors="coerce").to_numpy(float)
    if not np.isfinite(ids).all() or (ids < 0).any() or (ids != np.floor(ids)).any():
        raise ValueError("sample_index must contain nonnegative integers")
    frame["sample_index"] = ids.astype(int)
    if frame["sample_index"].duplicated().any():
        raise ValueError("sample_index is not unique")
    frame["label"] = frame["label"].astype(str).str.strip()
    if set(frame["label"]) - set(LABEL_MAP):
        raise ValueError(f"Unknown labels: {set(frame['label']) - set(LABEL_MAP)}")
    frame["target"] = frame["label"].map(LABEL_MAP).astype(int)
    if frame["target"].nunique() != 2:
        raise ValueError("Both binary classes are required")
    feature_matrix(frame)
    # Keep identical IDs from different sources together until identity is confirmed.
    frame["group_id"] = frame["equipment_id"]
    return raw, frame.sort_values("sample_index").reset_index(drop=True)


def legacy_consistency(current: pd.DataFrame, directory: Path) -> dict:
    result = {}
    for name in ("train", "test"):
        path = directory / f"{name}.csv"
        if not path.exists():
            continue
        old = pd.read_csv(path)
        if "sample_index" not in old or old["sample_index"].duplicated().any():
            result[name] = {"error": "Missing or duplicate sample_index"}
            continue
        matched = old.merge(current, on="sample_index", suffixes=("_old", "_current"))
        changes = {}
        for feature in FEATURES:
            if feature not in old:
                changes[feature] = "missing"
                continue
            before = pd.to_numeric(matched[f"{feature}_old"], errors="coerce")
            now = pd.to_numeric(matched[f"{feature}_current"], errors="coerce")
            count = int((~np.isclose(before, now, rtol=1e-8, atol=1e-10)).sum())
            if count:
                changes[feature] = count
        result[name] = {"rows": len(old), "matched_current_rows": len(matched),
                        "changed_features": changes, "used_for_training": False}
    return result


def audit_dataset(path: Path, config: ExperimentConfig, output: Path,
                  raw_file: Path | None = None, verify_raw_count: int = 8):
    raw, frame = load_experiment_data(path, config.sampling_rate)
    X = feature_matrix(frame)
    devices = frame.groupby("group_id", sort=True).agg(
        samples=("target", "size"), positive=("target", "sum"),
        label_count=("target", "nunique"),
    )
    devices["negative"] = devices["samples"] - devices["positive"]
    if "source" in frame:
        sources = frame.groupby("group_id")["source"]
        devices["source_count"] = sources.nunique(dropna=False)
        devices["sources"] = sources.agg(lambda s: " | ".join(sorted(set(s.fillna("<missing>").astype(str)))))
    devices.to_csv(output / "device_summary.csv", encoding="utf-8-sig")
    feature_stats = pd.DataFrame(X, columns=FEATURES).describe().T
    feature_stats["unique_values"] = [frame[f].nunique() for f in FEATURES]
    feature_stats.to_csv(output / "feature_statistics.csv", encoding="utf-8-sig")
    duplicate_mask = frame.duplicated(list(FEATURES), keep=False)
    duplicates = frame.loc[duplicate_mask]
    duplicates.to_csv(output / "duplicate_feature_rows.csv", index=False, encoding="utf-8-sig")
    cross_group_duplicates = int((duplicates.groupby(list(FEATURES))["group_id"].nunique() > 1).sum())
    if cross_group_duplicates:
        raise ValueError("Identical feature vectors cross equipment groups; resolve provenance before training")
    checks = []
    if raw_file is not None and raw_file.exists() and verify_raw_count > 0:
        from scipy.io import loadmat
        from msm_features import analyze_signal
        mat = loadmat(raw_file)
        if "data" not in mat:
            raise ValueError("Raw MAT file must contain data")
        selected = frame.sample(n=min(verify_raw_count, len(frame)), random_state=config.seed)
        for _, row in selected.iterrows():
            sample_index = int(row["sample_index"])
            signal = np.asarray(mat["data"][sample_index, 3], dtype=float).ravel()
            fresh = analyze_signal(signal, float(row["sampling_rate"]))["features"]
            changed = [f for f in FEATURES if not np.isclose(
                fresh[f], row[f], rtol=1e-7, atol=1e-10)]
            checks.append({"sample_index": sample_index, "changed_features": changed})
        write_json(output / "raw_feature_checks.json", checks)
        if any(check["changed_features"] for check in checks):
            raise ValueError("Raw signal recomputation disagrees with features.csv; see raw_feature_checks.json")
    warning_messages = []
    duplicate_count = int(frame.duplicated(list(FEATURES)).sum())
    mixed_groups = int((devices["label_count"] > 1).sum())
    multisource = int((devices["source_count"] > 1).sum()) if "source_count" in devices else 0
    if duplicate_count:
        warning_messages.append(f"{duplicate_count} repeated feature rows kept; all are within a device group")
    if multisource:
        warning_messages.append(f"{multisource} equipment IDs appear across sources; conservatively grouped by equipment_id")
    if mixed_groups:
        warning_messages.append(f"{mixed_groups} device groups contain both labels; see device_summary.csv")
    if not checks:
        warning_messages.append("Raw signal recomputation was not performed")
    legacy = legacy_consistency(frame, path.parent)
    if any(value.get("changed_features") for value in legacy.values()):
        warning_messages.append("Historical train.csv/test.csv contain stale features; never used by this experiment")
    if raw_file is not None and raw_file.exists():
        raw_identity = {"path": str(raw_file.resolve()), "sha256": sha256_file(raw_file)}
    else:
        raw_identity = None
    audit = {
        "feature_file": str(path.resolve()), "feature_sha256": sha256_file(path),
        "raw_file": raw_identity, "all_rows": len(raw), "eligible_rows": len(frame),
        "excluded_sampling_rate_rows": len(raw) - len(frame),
        "sampling_rate_counts": {str(k): int(v) for k, v in raw["sampling_rate"].value_counts(dropna=False).items()},
        "sampling_rate": config.sampling_rate, "devices": len(devices),
        "class_counts": {str(k): int(v) for k, v in frame["target"].value_counts().items()},
        "devices_per_class": {str(k): int(v) for k, v in frame.groupby("target")["group_id"].nunique().items()},
        "grouping": "equipment_id, conservative across sources", "mixed_label_groups": mixed_groups,
        "multisource_groups": multisource, "duplicate_feature_rows": duplicate_count,
        "cross_group_duplicate_feature_sets": cross_group_duplicates,
        "constant_features": [f for f in FEATURES if frame[f].nunique() <= 1],
        "feature_groups": {"spectral_envelope": list(ENVELOPE_FEATURES),
                           "spectrum_roi": list(SPECTRUM_FEATURES), "full": list(FEATURES)},
        "raw_recomputed_samples": len(checks), "legacy_tables": legacy,
        "quality_flags": {col: {str(k): int(v) for k, v in frame[col].value_counts(dropna=False).items()}
                          for col in ("feature_valid", "roi_active", "gaussian_success") if col in frame},
        "warnings": warning_messages,
    }
    write_json(output / "data_audit.json", audit)
    return frame, audit


def group_holdout(y, groups, config: ExperimentConfig):
    """Select on class/sample balance only, never on fitted model performance."""
    if not 0 < config.test_size < 1:
        raise ValueError("test_size must be between 0 and 1")
    best = None
    splitter = GroupShuffleSplit(n_splits=200, test_size=config.test_size,
                                 random_state=config.seed)
    for train, test in splitter.split(np.zeros(len(y)), y, groups):
        if len(np.unique(y[train])) != 2 or len(np.unique(y[test])) != 2:
            continue
        # Leave enough development groups for outer and inner grouped folds.
        if min(len(np.unique(groups[train][y[train] == c])) for c in (0, 1)) < 4:
            continue
        score = abs(len(test) / len(y) - config.test_size) + abs(y[test].mean() - y.mean())
        if best is None or score < best[0]:
            best = (score, train, test)
    if best is None:
        raise ValueError("Cannot create a two-class device holdout with enough development groups")
    _, train, test = best
    assert not set(groups[train]) & set(groups[test])
    return train, test


def grouped_folds(y, groups, requested: int, seed: int):
    """Never fall back to row-level CV when groups/classes are insufficient."""
    if requested < 2:
        raise ValueError("At least two grouped folds are required")
    class_groups = min(len(np.unique(groups[y == c])) for c in (0, 1))
    for count in range(min(requested, class_groups), 1, -1):
        for offset in range(20):
            splitter = StratifiedGroupKFold(n_splits=count, shuffle=True,
                                            random_state=seed + offset)
            splits = list(splitter.split(np.zeros(len(y)), y, groups))
            if all(len(np.unique(y[a])) == len(np.unique(y[b])) == 2
                   and not set(groups[a]) & set(groups[b]) for a, b in splits):
                return splits
    raise ValueError("Insufficient two-class device groups for grouped CV")


class FoldADASYN(BaseEstimator):
    """Fit only on a fold's training rows; record any necessary fallback."""

    def __init__(self, random_state=42):
        self.random_state = random_state

    def fit_resample(self, X, y):
        self.n_input_ = len(y)
        self.n_features_in_ = X.shape[1]
        self.fallback_reason_ = None
        counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
        if min(counts) == 0:
            raise ValueError("Resampling requires two classes")
        if counts[0] == counts[1]:
            self.method_ = "none_already_balanced"
            self.n_output_ = len(y)
            return X, y
        if min(counts) < 2:
            self.fallback_reason_ = "Minority class has fewer than two training rows"
        else:
            try:
                sampler = ADASYN(n_neighbors=min(5, int(min(counts)) - 1),
                                 sampling_strategy="minority", random_state=self.random_state)
                X_new, y_new = sampler.fit_resample(X, y)
                self.method_ = "adasyn"
                self.n_output_ = len(y_new)
                return X_new, y_new
            except (ValueError, RuntimeError) as error:
                # Separable/small folds may not support ADASYN's neighbour allocation.
                self.fallback_reason_ = f"{type(error).__name__}: {error}"
        X_new, y_new = RandomOverSampler(random_state=self.random_state).fit_resample(X, y)
        self.method_ = "random_oversampler_fallback"
        self.n_output_ = len(y_new)
        return X_new, y_new


def base_classifiers(config: ExperimentConfig, seed: int):
    return {
        "rf": RandomForestClassifier(n_estimators=config.trees, max_features="sqrt",
                                     random_state=seed, n_jobs=config.jobs),
        "xgb": XGBClassifier(n_estimators=config.trees, max_depth=3, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             objective="binary:logistic", eval_metric="logloss",
                             random_state=seed, n_jobs=config.jobs),
        "svm": SVC(kernel="rbf", C=2.0, gamma="scale", probability=True,
                   random_state=seed),
    }


def fit_pipeline(X, y, classifier, seed):
    pipeline = Pipeline([("scale", StandardScaler()), ("resample", FoldADASYN(seed)),
                         ("model", classifier)])
    pipeline.fit(X, y)
    return pipeline


def resampling_record(pipeline):
    sampler = pipeline.named_steps["resample"]
    return {"input_rows": sampler.n_input_, "output_rows": sampler.n_output_,
            "method": sampler.method_, "fallback_reason": sampler.fallback_reason_}


class FullFeatureStacking:
    """RF/XGB/SVM + original 17 features -> LR, with grouped inner OOF."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def fit(self, X, y, groups):
        self.training_groups_ = sorted(set(groups))
        splits = grouped_folds(y, groups, self.config.inner_folds, self.config.seed)
        oof = np.full((len(y), 3), np.nan)
        self.fold_records_ = []
        self.resampling_records_ = []
        for fold, (train, valid) in enumerate(splits):
            record = {"fold": fold, "train_groups": sorted(set(groups[train])),
                      "valid_groups": sorted(set(groups[valid]))}
            self.fold_records_.append(record)
            for j, (name, model) in enumerate(base_classifiers(self.config, self.config.seed + fold).items()):
                pipeline = fit_pipeline(X[train], y[train], model, self.config.seed + fold)
                oof[valid, j] = pipeline.predict_proba(X[valid])[:, 1]
                self.resampling_records_.append({"stage": f"inner_{fold}_{name}", **resampling_record(pipeline)})
        if not np.isfinite(oof).all():
            raise RuntimeError("Incomplete inner OOF probabilities")
        self.meta_scale_ = StandardScaler().fit(X)
        meta_X = np.column_stack((oof, self.meta_scale_.transform(X)))
        self.meta_ = LogisticRegression(C=1, class_weight="balanced", max_iter=3000)
        self.meta_.fit(meta_X, y)  # Original rows only; no synthetic meta validation rows.
        self.models_ = {}
        for name, model in base_classifiers(self.config, self.config.seed).items():
            pipeline = fit_pipeline(X, y, model, self.config.seed)
            self.models_[name] = pipeline
            self.resampling_records_.append({"stage": f"refit_{name}", **resampling_record(pipeline)})
        return self

    def predict_proba(self, X):
        base = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_.values()])
        return self.meta_.predict_proba(np.column_stack((base, self.meta_scale_.transform(X))))


class MultiViewJudge:
    """Three-branch predictor: 5D envelope + 12D spectrum/ROI + full 17D Stacking."""

    def __init__(self, config: ExperimentConfig):
        self.config = config

    def fit(self, frame: pd.DataFrame):
        X = feature_matrix(frame)
        y = frame["target"].to_numpy(int)
        groups = frame["group_id"].to_numpy(str)
        envelope_indices = [FEATURES.index(f) for f in ENVELOPE_FEATURES]
        spectrum_indices = [FEATURES.index(f) for f in SPECTRUM_FEATURES]
        self.oof_probabilities_ = np.full((len(y), 3), np.nan)
        self.oof_fold_ = np.full(len(y), -1, dtype=int)
        self.fold_records_ = []
        self.resampling_records_ = []
        self.outer_inner_records_ = []
        splits = grouped_folds(y, groups, self.config.folds, self.config.seed)
        for fold, (train, valid) in enumerate(splits):
            print(f"Outer grouped OOF {fold + 1}/{len(splits)}: train={len(train)}, valid={len(valid)}", flush=True)
            fold_config = ExperimentConfig(**{**asdict(self.config), "seed": self.config.seed + fold})
            self.fold_records_.append({"fold": fold,
                "train_groups": sorted(set(groups[train])), "valid_groups": sorted(set(groups[valid])),
                "train_sample_ids": frame.iloc[train]["sample_index"].tolist(),
                "valid_sample_ids": frame.iloc[valid]["sample_index"].tolist()})
            for column, indices in enumerate((envelope_indices, spectrum_indices)):
                model = base_classifiers(fold_config, fold_config.seed)["rf"]
                pipeline = fit_pipeline(X[train][:, indices], y[train], model, fold_config.seed)
                self.oof_probabilities_[valid, column] = pipeline.predict_proba(X[valid][:, indices])[:, 1]
                self.resampling_records_.append({"stage": f"outer_{fold}_{BRANCH_COLUMNS[column]}", **resampling_record(pipeline)})
            full = FullFeatureStacking(fold_config).fit(X[train], y[train], groups[train])
            self.oof_probabilities_[valid, 2] = full.predict_proba(X[valid])[:, 1]
            self.outer_inner_records_.append({"outer_fold": fold, "inner_folds": full.fold_records_})
            self.resampling_records_.extend({"stage": f"outer_{fold}/{r['stage']}",
                **{key: value for key, value in r.items() if key != "stage"}}
                for r in full.resampling_records_)
            self.oof_fold_[valid] = fold
        if not np.isfinite(self.oof_probabilities_).all() or (self.oof_fold_ < 0).any():
            raise RuntimeError("Incomplete outer OOF probabilities")
        # Preserve empirical class prevalence; do not rebalance the Judge inputs.
        self.judge_two_ = LogisticRegression(C=1, max_iter=3000).fit(self.oof_probabilities_[:, :2], y)
        self.judge_three_ = LogisticRegression(C=1, max_iter=3000).fit(self.oof_probabilities_, y)
        print("Refitting all three branches on development devices only", flush=True)
        self.envelope_ = fit_pipeline(X[:, envelope_indices], y,
            base_classifiers(self.config, self.config.seed)["rf"], self.config.seed)
        self.spectrum_ = fit_pipeline(X[:, spectrum_indices], y,
            base_classifiers(self.config, self.config.seed)["rf"], self.config.seed)
        self.full_ = FullFeatureStacking(self.config).fit(X, y, groups)
        for name, pipeline in (("envelope", self.envelope_), ("spectrum", self.spectrum_)):
            self.resampling_records_.append({"stage": f"final_{name}", **resampling_record(pipeline)})
        self.resampling_records_.extend({"stage": f"final_full/{r['stage']}",
            **{key: value for key, value in r.items() if key != "stage"}}
            for r in self.full_.resampling_records_)
        self.training_groups_ = sorted(set(groups))
        return self

    def predict_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        if "sampling_rate" in frame:
            rates = pd.to_numeric(frame["sampling_rate"], errors="coerce")
            if not np.isclose(rates, self.config.sampling_rate, rtol=0, atol=1e-6).all():
                raise ValueError(f"Prediction rows must use {self.config.sampling_rate:g} Hz; select eligible rows explicitly")
        probabilities = np.column_stack((
            self.envelope_.predict_proba(feature_matrix(frame, ENVELOPE_FEATURES))[:, 1],
            self.spectrum_.predict_proba(feature_matrix(frame, SPECTRUM_FEATURES))[:, 1],
            self.full_.predict_proba(feature_matrix(frame))[:, 1],
        ))
        result = frame.loc[:, [col for col in ("sample_index", "equipment_id", "group_id", "source", "target")
                               if col in frame]].reset_index(drop=True).copy()
        for i, col in enumerate(BRANCH_COLUMNS):
            result[col] = probabilities[:, i]
        result["p_judge_two"] = self.judge_two_.predict_proba(probabilities[:, :2])[:, 1]
        result["p_judge_three"] = self.judge_three_.predict_proba(probabilities)[:, 1]
        result["prediction"] = (result["p_judge_three"] >= self.config.threshold).astype(int)
        result["diagnosis"] = result["prediction"].map(LABEL_NAMES)
        return result


def probability_metrics(y, p, threshold=0.5):
    predicted = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    both = len(np.unique(y)) == 2
    return {"n": len(y), "threshold": threshold,
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)) if both else None,
        "average_precision": float(average_precision_score(y, p)) if both else None,
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}


def calibration_rows(y, p):
    rows = []
    bins = np.minimum((np.asarray(p) * 10).astype(int), 9)
    for index in range(10):
        selected = bins == index
        rows.append({"bin": index, "left": index / 10, "right": (index + 1) / 10,
                     "n": int(selected.sum()),
                     "mean_probability": float(np.mean(np.asarray(p)[selected])) if selected.any() else None,
                     "positive_rate": float(np.mean(np.asarray(y)[selected])) if selected.any() else None})
    return rows
