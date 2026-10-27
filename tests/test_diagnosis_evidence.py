"""Truthful spectral narrative output and removal of time-domain details."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diagnosis_evidence import (
    build_evidence_details,
    build_evidence_narrative,
    build_judge_conclusion,
)


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
            signal_statistics={"rms": 0.21, "kurtosis_pearson": 4.86},
            features={"f2": 0.82, "f7": 0.12, "f8": 1000},
            roi=dict(active=True, start=2192, end=2982, peak_frequency=2240, centroid=2590),
            gaussian=dict(success=True, rmse=0.06),
        )
        text = build_evidence_narrative(report)
        self.assertNotIn("信号状态", text)
        self.assertNotIn("时域", text)
        self.assertNotIn("4.86", text)
        self.assertIn("1.23–5.12 kHz", text)
        self.assertIn("2.19–2.98 kHz", text)
        self.assertIn("不等于全部信号能量", text)
        self.assertNotIn("说明频谱稳定", text)
        self.assertIn("三路判断存在分歧", text)
        self.assertIn("最终输出“非轴承润滑不良”", text)
        self.assertIn("Judge 最终概率为 20.00%", text)

    def test_inclusive_threshold_and_missing_reports(self):
        text = build_judge_conclusion(dict(judge_probability=0.5, classification_threshold=0.5))
        self.assertIn("最终输出“轴承润滑不良”", text)
        missing = build_evidence_narrative({})
        self.assertNotIn("时域", missing)
        self.assertNotIn("信号状态", missing)
        self.assertIn("未提供 Judge 概率", missing)
        self.assertNotIn("0.00%", missing)
        self.assertNotIn("较平稳", missing)

    def test_detail_panel_preserves_shap_but_ignores_legacy_time_statistics(self):
        details = build_evidence_details(dict(
            signal_statistics={"rms": 0.21, "kurtosis_pearson": 4.86},
            top_evidence=[dict(feature="f7", feature_name="谱平坦度", feature_value=0.1194, shap_value=-0.0905)],
        ))
        self.assertIn("p_full", details)
        self.assertIn("0.1194", details)
        self.assertIn("-0.0905", details)
        self.assertNotIn("时域", details)
        self.assertNotIn("峰值因子", details)
        self.assertNotIn("4.86", details)
        self.assertNotIn("未提供", details)


if __name__ == "__main__":
    unittest.main()
