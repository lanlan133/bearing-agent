"""Test the real five-region entrypoint and its sample/chart/report synchronization."""

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from workbench_state import WorkbenchState
from workbench_fixtures import write_mat, diagnosis_report


class TestWorkbenchUI(unittest.TestCase):
    def prepared_app(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        fixture = write_mat(Path(temp.name) / "fixture.mat")
        state = WorkbenchState(upload_directory=temp.name)
        app = AppTest.from_file(str(ROOT / "app_chat.py"), default_timeout=40)
        app.session_state["workbench"] = state
        app.run()
        app.file_uploader[0].set_value(("fixture.mat", fixture.read_bytes(), "application/octet-stream")).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        return app

    def test_empty_state_and_inline_chat(self):
        app = AppTest.from_file(str(ROOT / "app_chat.py"), default_timeout=40).run()
        self.assertFalse(app.exception)
        self.assertTrue(app.button(key="diagnose_current").disabled)
        self.assertTrue(app.chat_input[0].disabled)
        self.assertEqual(len(app.get("vega_lite_chart")), 4)
        content = "\n".join(item.value for item in app.markdown)
        for title in ("时域波形", "全频段频谱", "完整包络", "ROI 扩展"):
            self.assertIn(f"**{title}**", content)
        self.assertIn("#### 频谱包络与 ROI", content)
        # The input belongs to the main column tree, not the global bottom region.
        self.assertEqual(len(app.main.get("chat_input")), 1)

    def test_diagnose_switch_cache_and_collapsed_details(self):
        with patch("agent_tools.run_diagnosis_agent", return_value=diagnosis_report()) as diagnose:
            app = self.prepared_app()
            self.assertEqual(len(app.get("vega_lite_chart")), 4)
            app.button(key="diagnose_current").click().run()
            self.assertFalse(app.exception)
            diagnose.assert_called_once()
            content = "\n".join(item.value for item in app.markdown)
            for title in ("关键依据", "后续建议"):
                self.assertIn(f"#### {title}", content)
            self.assertIn("润滑不良概率 20.00%", content)
            self.assertIn("低于 **50%** 的分类阈值", content)
            self.assertIn("谱平坦度降低该分支的润滑不良概率", content)
            self.assertNotIn("时域指标", content)
            self.assertIn("- 继续进行趋势监测。", content)
            self.assertNotIn("AI Agent 分析", content)
            details = app.expander[0]
            self.assertEqual(details.label, "展开查看模型详情")
            self.assertFalse(details.proto.expanded)
            detail_content = "\n".join(item.value for item in details.get("markdown"))
            self.assertIn("完整 17D Stacking 90.00%", detail_content)
            self.assertIn("完整 17D Stacking 分支概率 p_full", detail_content)
            self.assertIn("| f2 | 0.8200 | — | — |", detail_content)
            app.button(key="next_sample").click().run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state["workbench"].sample_number, 2)
            self.assertIsNone(app.session_state["workbench"].report)
            self.assertEqual(app.session_state["workbench"].sample["equipment_id"], "device_1")
            self.assertEqual(len(app.get("vega_lite_chart")), 4)
            self.assertFalse(app.success)
            app.button(key="previous_sample").click().run()
            self.assertFalse(app.exception)
            self.assertIsNotNone(app.session_state["workbench"].report)
            diagnose.assert_called_once()

    def test_natural_language_updates_current_sample_and_api_failure_preserves_report(self):
        app = self.prepared_app()
        result = {
            "status": "completed", "answer": "第3条诊断已完成。",
            "context": {"last_report": diagnosis_report(), "last_sample_number": 3},
            "observations": [{"success": True, "tool": "diagnose_sample"}],
        }
        with patch("agent_runtime.run_agent_task", return_value=result) as planner:
            app.chat_input[0].set_value("请诊断第3条并解释原因").run()
            self.assertFalse(app.exception)
            planner.assert_called_once()
            state = app.session_state["workbench"]
            self.assertEqual(state.sample_number, 3)
            self.assertEqual(app.number_input[0].value, 3)
            self.assertEqual(state.sample["equipment_id"], "device_2")
            self.assertEqual(state.messages[-1]["sample_number"], 3)
        with patch("agent_runtime.run_agent_task", side_effect=RuntimeError("API timeout")):
            app.chat_input[0].set_value("解释一下三路概率").run()
            self.assertFalse(app.exception)
            self.assertIsNotNone(app.session_state["workbench"].report)
            self.assertIn("API timeout", app.session_state["workbench"].messages[-1]["content"])


if __name__ == "__main__":
    unittest.main()
