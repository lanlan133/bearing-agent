import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from diagnosis_tool import (
    explain_prediction,
    predict_complete_stacking_probability,
)
from stacking_predictor import predict_paper_stacking


MODEL_FILE = PROJECT_ROOT / "models" / "Paper_Stacking_Bundle.joblib"
FEATURE_FILE = PROJECT_ROOT / "data" / "features" / "features.csv"
FEATURE_NAMES = [f"f{i}" for i in range(1, 18)]


def test_complete_stacking_shap():
    bundle = joblib.load(MODEL_FILE)
    sample = (
        pd.read_csv(FEATURE_FILE, nrows=1)[FEATURE_NAMES]
        .to_numpy(dtype=float)[0]
    )

    direct_probability = float(
        predict_complete_stacking_probability(
            bundle,
            sample,
        )[0]
    )
    deployed_probability = float(
        predict_paper_stacking(sample)["final_probability"]
    )

    assert np.isclose(
        direct_probability,
        deployed_probability,
        rtol=0.0,
        atol=1e-12,
    )
    assert bundle["shap_config"]["explained_model"] == (
        "complete_stacking_probability"
    )
    assert bundle["shap_background_raw"].shape[1] == 17

    evidence = explain_prediction(
        bundle,
        sample,
        top_n=17,
    )
    shap_sum = sum(item["shap_value"] for item in evidence)
    background_probability = float(
        np.mean(
            predict_complete_stacking_probability(
                bundle,
                bundle["shap_background_raw"],
            )
        )
    )

    assert np.isclose(
        background_probability + shap_sum,
        direct_probability,
        rtol=0.0,
        atol=1e-6,
    )


if __name__ == "__main__":
    test_complete_stacking_shap()
    print("完整Stacking SHAP测试通过")
