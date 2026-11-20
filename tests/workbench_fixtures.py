"""Small MAT fixtures and deterministic diagnosis results for workbench tests."""

import numpy as np
from scipy.io import savemat


def write_mat(path, count=3):
    data = np.empty((count, 12), dtype=object)
    t = np.arange(1024) / 10240
    for index in range(count):
        signal = (index + 1) * np.sin(2 * np.pi * (2000 + index * 100) * t)
        data[index] = [
            f"M{index + 1}H", 1500, 10240, signal,
            "非轴承润滑不良", f"device_{index}", "测试设备", 0,
            "2026-11-17", 0, 0, "fixture",
        ]
    savemat(path, {"data": data})
    return path


def diagnosis_report():
    return {
        "diagnosis": "非轴承润滑不良", "decision": "非轴承润滑不良",
        "model_diagnosis": "非轴承润滑不良", "model_prediction": 0,
        "model_name": "三路 Judge", "judge_probability": 0.2, "risk_probability": 0.2,
        "full_stacking_probability": 0.9, "stacking_probability": 0.9,
        "envelope_probability": 0.1, "spectrum_probability": 0.1,
        "classification_threshold": 0.5, "sampling_rate": 10240,
        "base_probabilities": {"rf": 0.1, "xgb": 0.2, "svm": 0.3},
        "features": {"f2": 0.82, "f7": 0.12},
        "roi": {"active": False}, "gaussian": {"success": False},
        "top_evidence": [dict(feature="f7", feature_name="谱平坦度", feature_value=0.12, shap_value=-0.1)],
        "domain": {"validated": True, "message": "采样率匹配。"},
        "maintenance_advice": ["继续进行趋势监测。"],
    }
