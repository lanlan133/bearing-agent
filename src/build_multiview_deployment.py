"""Train the production three-branch Judge on all eligible 10240 Hz samples.

This intentionally differs from validation: it has no held-out test set because
its sole purpose is to create the version used by the Streamlit application.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np

from multiview_experiment import (
    FEATURES,
    PROJECT_ROOT,
    ExperimentConfig,
    MultiViewJudge,
    feature_matrix,
    load_experiment_data,
    sha256_file,
)


DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "MultiView_Judge_Bundle.joblib"


def build_bundle(
    data_file: Path,
    output_file: Path = DEFAULT_OUTPUT,
    seed: int = 42,
    trees: int = 300,
    jobs: int = 2,
    shap_background_size: int = 64,
) -> Path:
    """Fit every branch and the Judge using all eligible training devices."""

    if output_file.exists():
        raise FileExistsError(
            f"Refusing to overwrite deployment model: {output_file}. "
            "Remove it explicitly before rebuilding."
        )
    if shap_background_size < 1:
        raise ValueError("shap_background_size must be positive")

    config = ExperimentConfig(seed=seed, trees=trees, jobs=jobs)
    _, frame = load_experiment_data(data_file, config.sampling_rate)
    model = MultiViewJudge(config).fit(frame)

    X = feature_matrix(frame)
    rng = np.random.default_rng(seed)
    background_size = min(shap_background_size, len(X))
    background_indices = np.sort(rng.choice(len(X), size=background_size, replace=False))
    bundle = {
        "format_version": 1,
        "model_name": "三路 Judge（5D 包络 + 12D 频谱/ROI + 完整 17D Stacking）",
        "model": model,
        "config": asdict(config),
        "feature_columns": list(FEATURES),
        "feature_sha256": sha256_file(data_file),
        "training_sample_ids": frame["sample_index"].astype(int).tolist(),
        "training_group_count": int(frame["group_id"].nunique()),
        "shap_background_raw": X[background_indices],
        "shap_config": {
            "algorithm": "kernel_shap",
            "background_source": "original_real_training_samples",
            "background_size": int(background_size),
            "nsamples": 256,
            "explained_output": "p_full (完整 17D Stacking 分支概率)",
        },
    }
    output_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_file)
    return output_file


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT / "data" / "features" / "features.csv",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=int, default=300)
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--shap-background-size", type=int, default=64)
    args = parser.parse_args(argv)
    output = build_bundle(
        data_file=args.data,
        output_file=args.output,
        seed=args.seed,
        trees=args.trees,
        jobs=args.jobs,
        shap_background_size=args.shap_background_size,
    )
    print(f"Deployment bundle written: {output}")


if __name__ == "__main__":
    main()
