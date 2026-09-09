from pathlib import Path
import json
import re

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_DIR = PROJECT_ROOT / "reports" / "paper_stacking_validation"
REPORT_DIR = PROJECT_ROOT / "reports" / "review_policy"
POLICY_FILE = PROJECT_ROOT / "models" / "review_policy.json"

PREDICTION_GLOB = "seed_*_predictions.csv"
PROBABILITY_COLUMN = "stacking_probability"
LABEL_COLUMN = "y_true"
BASE_THRESHOLD = 0.5
MAX_REVIEW_RATE = 0.15


def _safe_ratio(numerator, denominator):
    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def _seed_sort_key(path):
    match = re.fullmatch(r"seed_(.+)_predictions", path.stem)
    if match is None:
        return (1, path.stem)
    seed_text = match.group(1)
    try:
        return (0, int(seed_text))
    except ValueError:
        return (1, seed_text)


def load_stacking_predictions():
    prediction_files = sorted(
        PREDICTION_DIR.glob(PREDICTION_GLOB),
        key=_seed_sort_key,
    )
    if not prediction_files:
        raise FileNotFoundError(
            "没有找到 Paper Stacking 验证预测文件: "
            f"{PREDICTION_DIR / PREDICTION_GLOB}"
        )

    frames = []
    required_columns = {LABEL_COLUMN, PROBABILITY_COLUMN}

    for prediction_file in prediction_files:
        frame = pd.read_csv(prediction_file)
        missing_columns = required_columns - set(frame.columns)
        if missing_columns:
            raise ValueError(
                f"{prediction_file.name} 缺少字段: {sorted(missing_columns)}"
            )

        seed_match = re.fullmatch(
            r"seed_(.+)_predictions",
            prediction_file.stem,
        )
        seed = seed_match.group(1) if seed_match else prediction_file.stem
        frame = frame.copy()
        frame["validation_seed"] = seed
        frame["prediction_file"] = prediction_file.name
        frames.append(frame)

    predictions = pd.concat(frames, ignore_index=True)
    predictions[LABEL_COLUMN] = pd.to_numeric(
        predictions[LABEL_COLUMN], errors="raise"
    ).astype(int)
    predictions[PROBABILITY_COLUMN] = pd.to_numeric(
        predictions[PROBABILITY_COLUMN], errors="raise"
    ).astype(float)

    labels = set(predictions[LABEL_COLUMN].unique())
    if not labels.issubset({0, 1}):
        raise ValueError(f"{LABEL_COLUMN} 必须为二分类 0/1，实际为: {labels}")

    probabilities = predictions[PROBABILITY_COLUMN].to_numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{PROBABILITY_COLUMN} 包含 NaN 或无穷值")
    if ((probabilities < 0.0) | (probabilities > 1.0)).any():
        raise ValueError(f"{PROBABILITY_COLUMN} 必须位于 [0, 1]")

    return predictions, prediction_files


def classification_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)

    return {
        "accuracy": _safe_ratio(tp + tn, len(y_true)),
        "precision": precision,
        "recall": recall,
        "f1": _safe_ratio(2.0 * precision * recall, precision + recall),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def evaluate_policy(y_true, probability, lower, upper):
    y_true = np.asarray(y_true, dtype=int)
    probability = np.asarray(probability, dtype=float)
    base_prediction = (probability >= BASE_THRESHOLD).astype(int)

    errors = base_prediction != y_true
    false_negatives = (y_true == 1) & (base_prediction == 0)
    false_positives = (y_true == 0) & (base_prediction == 1)
    review = (probability >= lower) & (probability <= upper)
    automatic = ~review

    error_count = int(errors.sum())
    fn_count = int(false_negatives.sum())
    fp_count = int(false_positives.sum())
    covered_errors = int((errors & review).sum())
    covered_fn = int((false_negatives & review).sum())
    covered_fp = int((false_positives & review).sum())
    automatic_count = int(automatic.sum())
    automatic_accuracy = (
        float((base_prediction[automatic] == y_true[automatic]).mean())
        if automatic_count
        else float("nan")
    )

    return {
        "lower": float(lower),
        "upper": float(upper),
        "review_count": int(review.sum()),
        "review_rate": float(review.mean()),
        "auto_count": automatic_count,
        "auto_accuracy": automatic_accuracy,
        "error_count": error_count,
        "covered_errors": covered_errors,
        "error_coverage": _safe_ratio(covered_errors, error_count),
        "fn_count": fn_count,
        "covered_fn": covered_fn,
        "fn_coverage": _safe_ratio(covered_fn, fn_count),
        "residual_fn": fn_count - covered_fn,
        "fp_count": fp_count,
        "covered_fp": covered_fp,
        "fp_coverage": _safe_ratio(covered_fp, fp_count),
        "residual_fp": fp_count - covered_fp,
    }


def search_review_policy(y_true, probability):
    results = []
    # Stacking 概率更集中在 0 和 1 附近，使用 0.01 步长可避免
    # 旧 XGBoost 策略的 0.05 粗网格漏掉有效的低概率复核带。
    lower_candidates = np.arange(0.01, 0.50, 0.01)
    upper_candidates = np.arange(0.51, 1.00, 0.01)

    for lower in lower_candidates:
        for upper in upper_candidates:
            results.append(
                evaluate_policy(
                    y_true=y_true,
                    probability=probability,
                    lower=round(float(lower), 2),
                    upper=round(float(upper), 2),
                )
            )
    return pd.DataFrame(results)


def select_policy(result_df):
    candidates = result_df[
        result_df["review_rate"] <= MAX_REVIEW_RATE
    ].copy()
    if candidates.empty:
        raise RuntimeError(
            f"没有满足复核率 <= {MAX_REVIEW_RATE:.0%} 的候选策略"
        )

    # 设备安全场景优先拦截漏诊，其次覆盖全部错误；覆盖率相同后，
    # 再选择自动区准确率更高、人工复核负担更低的策略。
    candidates = candidates.sort_values(
        by=["fn_coverage", "error_coverage", "auto_accuracy", "review_rate"],
        ascending=[False, False, False, True],
    )
    return candidates.iloc[0], candidates.head(10)


def build_seed_metrics(predictions, lower, upper):
    rows = []
    for seed, frame in predictions.groupby("validation_seed", sort=False):
        metrics = evaluate_policy(
            y_true=frame[LABEL_COLUMN].to_numpy(),
            probability=frame[PROBABILITY_COLUMN].to_numpy(),
            lower=lower,
            upper=upper,
        )
        metrics["seed"] = seed
        metrics["samples"] = len(frame)
        rows.append(metrics)

    columns = ["seed", "samples"] + [
        column for column in rows[0] if column not in {"seed", "samples"}
    ]
    return pd.DataFrame(rows)[columns]


def save_outputs(predictions, prediction_files, grid, top10, best):
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    POLICY_FILE.parent.mkdir(parents=True, exist_ok=True)

    lower = float(best["lower"])
    upper = float(best["upper"])
    probability = predictions[PROBABILITY_COLUMN].to_numpy()
    y_true = predictions[LABEL_COLUMN].to_numpy()
    y_pred = (probability >= BASE_THRESHOLD).astype(int)

    output = predictions.copy()
    output["review_required"] = (
        (probability >= lower) & (probability <= upper)
    )
    output["base_prediction"] = y_pred
    output["base_correct"] = y_pred == y_true

    seed_metrics = build_seed_metrics(predictions, lower, upper)
    base_metrics = classification_metrics(y_true, y_pred)

    grid.to_csv(
        REPORT_DIR / "review_policy_grid.csv",
        index=False,
        encoding="utf-8-sig",
    )
    top10.to_csv(
        REPORT_DIR / "top10_review_policies.csv",
        index=False,
        encoding="utf-8-sig",
    )
    seed_metrics.to_csv(
        REPORT_DIR / "stacking_review_policy_by_seed.csv",
        index=False,
        encoding="utf-8-sig",
    )
    output.to_csv(
        REPORT_DIR / "stacking_validation_predictions_with_review.csv",
        index=False,
        encoding="utf-8-sig",
    )

    unique_samples = (
        int(predictions["sample_index"].nunique())
        if "sample_index" in predictions.columns
        else None
    )
    policy = {
        "version": "2.0",
        "method": "paper_stacking_multi_seed_holdout_review",
        "model_name": "MSM-ADASYN-Stacking",
        "probability_field": PROBABILITY_COLUMN,
        "base_threshold": BASE_THRESHOLD,
        "lower_threshold": lower,
        "upper_threshold": upper,
        "selection": {
            "max_review_rate": MAX_REVIEW_RATE,
            "priority": [
                "fn_coverage",
                "error_coverage",
                "auto_accuracy",
                "review_rate",
            ],
            "threshold_step": 0.01,
        },
        "validation_source": {
            "directory": str(PREDICTION_DIR.relative_to(PROJECT_ROOT)),
            "file_pattern": PREDICTION_GLOB,
            "files": [path.name for path in prediction_files],
            "seed_count": len(prediction_files),
            "prediction_rows": len(predictions),
            "unique_sample_count": unique_samples,
            "pooling": "equal-weight prediction rows across held-out seed splits",
        },
        "validation_metrics": {
            "review_count": int(best["review_count"]),
            "review_rate": float(best["review_rate"]),
            "fn_coverage": float(best["fn_coverage"]),
            "fp_coverage": float(best["fp_coverage"]),
            "error_coverage": float(best["error_coverage"]),
            "auto_accuracy": float(best["auto_accuracy"]),
            "residual_fn": int(best["residual_fn"]),
            "residual_fp": int(best["residual_fp"]),
            "per_seed_review_rate_mean": float(seed_metrics["review_rate"].mean()),
            "per_seed_review_rate_std": float(seed_metrics["review_rate"].std(ddof=0)),
            "per_seed_auto_accuracy_mean": float(seed_metrics["auto_accuracy"].mean()),
            "per_seed_auto_accuracy_std": float(seed_metrics["auto_accuracy"].std(ddof=0)),
        },
        "base_validation_metrics": base_metrics,
        "decision_rule": {
            "normal": "p < lower_threshold",
            "review": "lower_threshold <= p <= upper_threshold",
            "fault": "p > upper_threshold",
        },
    }

    with open(POLICY_FILE, "w", encoding="utf-8") as handle:
        json.dump(policy, handle, ensure_ascii=False, indent=2)

    return policy, seed_metrics


def main():
    print("=" * 80)
    print("基于 Paper MSM-ADASYN-Stacking 验证结果构建 Review Policy")
    print("=" * 80)

    predictions, prediction_files = load_stacking_predictions()
    y_true = predictions[LABEL_COLUMN].to_numpy()
    probability = predictions[PROBABILITY_COLUMN].to_numpy()
    print(f"预测文件数: {len(prediction_files)}")
    print(f"验证预测行数: {len(predictions)}")
    print(f"概率字段: {PROBABILITY_COLUMN}")

    grid = search_review_policy(y_true, probability)
    best, top10 = select_policy(grid)
    policy, seed_metrics = save_outputs(
        predictions, prediction_files, grid, top10, best
    )

    metrics = policy["validation_metrics"]
    print("\n推荐 Review Policy")
    print(f"复核区间: [{policy['lower_threshold']:.2f}, {policy['upper_threshold']:.2f}]")
    print(f"复核率: {metrics['review_rate']:.2%}")
    print(f"漏诊覆盖率: {metrics['fn_coverage']:.2%}")
    print(f"误报覆盖率: {metrics['fp_coverage']:.2%}")
    print(f"总错误覆盖率: {metrics['error_coverage']:.2%}")
    print(f"自动区准确率: {metrics['auto_accuracy']:.2%}")
    print(f"剩余漏诊/误报: {metrics['residual_fn']}/{metrics['residual_fp']}")
    print(
        "跨 seed 复核率范围: "
        f"{seed_metrics['review_rate'].min():.2%} ~ "
        f"{seed_metrics['review_rate'].max():.2%}"
    )
    print(f"\nPolicy 已保存: {POLICY_FILE}")
    print(f"诊断报告已保存: {REPORT_DIR}")


if __name__ == "__main__":
    main()
