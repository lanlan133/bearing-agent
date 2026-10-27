"""Signal-statistic definitions and truthful, branch-specific narrative output."""

import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnosis_evidence import (
    build_evidence_details,
    build_evidence_narrative,
    build_judge_conclusion,
    calculate_signal_statistics,
)


class TestSignalStatistics(unittest.TestCase):
    def test_sinusoid_matches_known_rms_peak_factor_and_pearson_kurtosis(self):
        signal = 2 * np.sin(np.arange(1000) * 2 * np.pi / 100)
        stats = calculate_signal_statistics(signal)
        self.assertAlmostEqual(stats["rms"], np.sqrt(2))
        self.assertAlmostEqual(stats["crest_factor"], np.sqrt(2))
        self.assertAlmostEqual(stats["kurtosis_pearson"], 1.5)
        amplified = calculate_signal_statistics(signal * 100)
        self.assertAlmostEqual(amplified["rms"], stats["rms"] * 100)
        self.assertAlmostEqual(amplified["kurtosis_pearson"], stats["kurtosis_pearson"])

    def test_impulse_kurtosis_and_offset_use_centered_population_moments(self):
        signal = np.array([0.0] * 99 + [10.0])
        stats = calculate_signal_statistics(signal)
        centered = signal - signal.mean()
        expected = np.mean(centered ** 4) / np.mean(centered ** 2) ** 2
        self.assertAlmostEqual(stats["kurtosis_pearson"], expected)
        shifted = calculate_signal_statistics(signal + 5)
        self.assertAlmostEqual(shifted["kurtosis_pearson"], expected)
        self.assertNotEqual(shifted["rms"], stats["rms"])

    def test_zero_and_constant_are_serializable_without_fabricating_statistics(self):
        for value in (0, 8):
            stats = calculate_signal_statistics(np.full(32, value))
            self.assertIsNone(stats["kurtosis_pearson"])
            self.assertEqual(stats["crest_factor"], None if value == 0 else 1)
            json.dumps(stats, allow_nan=False)
        for invalid in ([], [np.nan], [np.inf]):
            with self.assertRaises(ValueError):
                calculate_signal_statistics(invalid)


class TestEvidenceNarrative(unittest.TestCase):
    def test_shap_strength_is_not_feature_count_and_is_explicitly_partial(self):
        evidence = [dict(feature="f14", feature_name="上包络RMS", feature_value=0.2, shap_value=0.5)]
        evidence += [dict(feature=f"f{i}", feature_name="特征", feature_value=0.1, shap_value=-0.01)
                     for i in range(1, 5)]
        text = build_evidence_narrative({"top_evidence": evidence})
        self.assertIn("合并作用为提高润滑不良概率", text)
        self.assertIn("其余特征", text)
        self.assertIn("不直接解释三路 Judge", text)
        self.assertNotIn("4 项推动", text)

    def test_full_positive_does_not_override_judge_and_roi_is_not_global_energy(self):
        report = dict(
            full_stacking_probability=0.9, judge_probability=0.2,
            envelope_probability=0.1, spectrum_probability=0.1,
            classification_threshold=0.5, diagnosis="非轴承润滑不良", sampling_rate=10240,
            signal_statistics=calculate_signal_statistics([1, -1] * 32),
            features={"f2": 0.82, "f7": 0.12, "f8": 1000},
            roi=dict(active=True, start=2192, end=2982, peak_frequency=2240, centroid=2590),
            gaussian=dict(success=True, rmse=0.06),
        )
        text = build_evidence_narrative(report)
        self.assertIn("时域 Pearson 峭度为 1.00", text)
        self.assertIn("1.23–5.12 kHz", text)
        self.assertIn("2.19–2.98 kHz", text)
        self.assertIn("不等于全部信号能量", text)
        self.assertNotIn("说明频谱稳定", text)
        self.assertIn("三路判断存在分歧", text)
        self.assertIn("最终输出“非轴承润滑不良”", text)
        self.assertIn("Judge 最终概率为 20.00%", text)
        self.assertIn("历史基线", text)

    def test_inclusive_threshold_and_missing_reports(self):
        text = build_judge_conclusion(dict(judge_probability=0.5, classification_threshold=0.5))
        self.assertIn("最终输出“轴承润滑不良”", text)
        missing = build_evidence_narrative({})
        self.assertIn("没有原始时域统计", missing)
        self.assertIn("未提供 Judge 概率", missing)
        self.assertNotIn("0.00%", missing)
        self.assertNotIn("较平稳", missing)

    def test_detail_panel_retains_exact_evidence_and_undefined_values(self):
        details = build_evidence_details(dict(
            signal_statistics=calculate_signal_statistics(np.zeros(8)),
            top_evidence=[dict(feature="f7", feature_name="谱平坦度", feature_value=0.1194, shap_value=-0.0905)],
        ))
        self.assertIn("p_full", details)
        self.assertIn("0.1194", details)
        self.assertIn("-0.0905", details)
        self.assertIn("未提供", details)


if __name__ == "__main__":
    unittest.main()
