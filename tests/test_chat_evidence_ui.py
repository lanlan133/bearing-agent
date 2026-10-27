"""The real chat entrypoint must retain narrative and collapsed details on reruns."""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from streamlit.testing.v1 import AppTest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diagnosis_evidence import calculate_signal_statistics


class TestChatEvidenceUI(unittest.TestCase):
    def test_diagnosis_has_four_paragraphs_collapsed_details_and_analysis_after_rerun(self):
        report = {
            "diagnosis": "非轴承润滑不良", "model_diagnosis": "非轴承润滑不良",
            "judge_probability": 0.2, "risk_probability": 0.2,
            "full_stacking_probability": 0.9, "stacking_probability": 0.9,
            "envelope_probability": 0.1, "spectrum_probability": 0.1,
            "classification_threshold": 0.5, "sampling_rate": 10240,
            "base_probabilities": {"rf": 0.1, "xgb": 0.2, "svm": 0.3},
            "signal_statistics": calculate_signal_statistics(np.sin(np.arange(64))),
            "features": {"f2": 0.82, "f7": 0.12},
            "roi": {"active": False}, "gaussian": {"success": False},
            "top_evidence": [dict(feature="f7", feature_name="谱平坦度", feature_value=0.12, shap_value=-0.1)],
            "domain": {"validated": True, "message": "采样率匹配。"},
            "maintenance_advice": ["继续进行趋势监测。"],
        }
        result = {
            "status": "completed", "answer": "诊断已完成。",
            "context": {"last_report": report, "last_sample_number": 2},
            "observations": [{"success": True, "tool": "diagnose_sample"}],
        }
        sample = dict(equipment_id="test_device", measurement_point="M2H", sampling_rate=10240, speed=1500)
        with patch("agent_runtime.run_agent_task", return_value=result) as runtime, \
                patch("data_loader.load_single_sample", return_value=sample):
            app = AppTest.from_file(str(PROJECT_ROOT / "app_chat.py"), default_timeout=30)
            app.session_state["mat_file_path"] = "fixture.mat"
            app.session_state["total_samples"] = 3
            app.run()
            self.assertFalse(app.exception)
            app.chat_input[0].set_value("2").run()
            self.assertFalse(app.exception)
            runtime.assert_called_once()
            message = app.session_state["messages"][-1]
            for title in ("信号状态", "频谱与 ROI 状态", "模型证据", "综合结论"):
                self.assertIn(f"**{title}：**", message["content"])
            self.assertIn("时域 Pearson 峭度", message["evidence_details"])
            self.assertIn("后续建议：继续进行趋势监测。", message["agent_analysis"])
            self.assertEqual(len(app.expander), 1)
            self.assertFalse(app.expander[0].proto.expanded)
            self.assertIn("详细时域指标", app.expander[0].label)
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.expander), 1)
            self.assertEqual(app.session_state["messages"][-1], message)
            self.assertIn("最终输出“非轴承润滑不良”", message["content"])


if __name__ == "__main__":
    unittest.main()
