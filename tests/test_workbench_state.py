"""State consistency, MAT reuse and plot semantics of the five-region workbench."""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agent_planner import build_planner_context
from agent_tools import get_sample_info, _context_dataset
from signal_charts import envelope_window, peak_preserving_indices, signal_frames, waveform_chart, spectrum_chart, roi_chart
from workbench_state import WorkbenchState
from workbench_fixtures import write_mat, diagnosis_report


class TestWorkbenchState(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = write_mat(Path(self.directory.name) / "samples.mat")
        self.state = WorkbenchState()
        self.state.activate_path(self.path)

    def test_sample_switch_invalidates_graphs_and_restores_only_matching_report(self):
        state = self.state
        with patch("data_loader.load_mat_dataset", side_effect=AssertionError("unexpected reread")), \
                patch("agent_tools.run_diagnosis_agent", return_value=diagnosis_report()) as diagnose:
            first = state.ensure_signal()
            state.diagnose()
            state.diagnose()
            diagnose.assert_called_once()
            get_sample_info(state.context(), 2)
            state.select_sample(2)
            self.assertIsNone(state.report)
            self.assertIsNone(state.analysis)
            second = state.ensure_signal()
            self.assertEqual(state.sample["equipment_id"], "device_1")
            self.assertIsNot(first, second)
            state.select_sample(1)
            self.assertEqual(state.report["judge_probability"], 0.2)
            self.assertEqual(state.report["model_prediction"], 0)
            state.diagnose()
            diagnose.assert_called_once()

    def test_new_file_clears_old_results_and_same_names_are_isolated(self):
        content = self.path.read_bytes()
        other = WorkbenchState()
        self.state.activate_upload("same.mat", content, Path(self.directory.name) / "uploads")
        other.activate_upload("same.mat", content, Path(self.directory.name) / "uploads")
        self.assertNotEqual(self.state.mat_file_path, other.mat_file_path)
        self.assertFalse(self.state.activate_upload("same.mat", content, Path(self.directory.name) / "uploads"))
        self.state.remember_report(diagnosis_report())
        new_file = write_mat(Path(self.directory.name) / "new.mat", count=2)
        self.state.activate_path(new_file)
        self.assertIsNone(self.state.report)
        self.assertFalse(self.state.reports)
        self.assertEqual(self.state.sample_number, 1)
        for invalid in (0, 3, 1.5, float("nan")):
            with self.assertRaises(ValueError):
                self.state.select_sample(invalid)

    def test_private_data_not_sent_to_planner_and_history_follows_current_sample(self):
        state = self.state
        state.add_message("user", "第一条的问题")
        state.select_sample(2)
        state.add_message("user", "第二条的问题")
        state.remember_report(diagnosis_report())
        context = build_planner_context(state.context())
        encoded = json.dumps(context, ensure_ascii=False)
        self.assertNotIn("_mat_dataset", encoded)
        self.assertNotIn("第一条的问题", encoded)
        self.assertIn("第二条的问题", encoded)
        self.assertIn("top_evidence", encoded)
        with patch("agent_tools.load_mat_dataset", return_value="fresh") as read:
            result = _context_dataset({"_mat_dataset": state.dataset, "_mat_dataset_path": "different.mat"}, self.path)
            self.assertEqual(result, "fresh")
            read.assert_called_once()

    def test_plot_coordinates_envelopes_and_peak_preserving_sampling(self):
        state = self.state
        analysis = state.ensure_signal()
        waveform, spectrum, local = signal_frames(state.sample, analysis)
        self.assertEqual(waveform.time.iloc[0], 0)
        self.assertAlmostEqual(waveform.time.iloc[-1], 1023 / 10240)
        self.assertLessEqual(spectrum.frequency.max(), 5120)
        self.assertIsNotNone(local)
        self.assertEqual(set(local.curve), {"频谱曲线", "上谱包络", "下谱包络"})
        self.assertEqual(local.frequency.min(), analysis["frequency"].min())
        self.assertEqual(local.frequency.max(), analysis["frequency"].max())
        self.assertLessEqual(len(local), 5000)
        for curve, key in (("上谱包络", "upper"), ("下谱包络", "lower")):
            self.assertEqual(local.loc[local.curve == curve, "amplitude"].max(), analysis["envelopes"][key].max())
        for chart in (waveform_chart(waveform), spectrum_chart(spectrum, analysis["roi"]), roi_chart(local)):
            chart.to_dict(validate=True)
        values = np.zeros(50000)
        values[12345], values[43210] = 20, -30
        selected = peak_preserving_indices(values)
        self.assertIn(12345, selected)
        self.assertIn(43210, selected)
        self.assertLessEqual(len(selected), 1800)
        analysis = {**analysis, "roi": {"active": False}}
        self.assertIsNotNone(signal_frames(state.sample, analysis)[2])

    def test_expanded_roi_and_local_y_axis_keep_contour_visible(self):
        analysis = self.state.ensure_signal()
        start, end = envelope_window(analysis, view="roi")
        roi = analysis["roi"]
        self.assertLess(start, roi["start"])
        self.assertGreater(end, roi["end"])
        local = signal_frames(self.state.sample, analysis, envelope_view="roi")[2]
        self.assertTrue(local.frequency.between(start, end).all())
        self.assertLess(local.frequency.min(), roi["start"])
        self.assertGreater(local.frequency.max(), roi["end"])
        small = local.copy()
        small["amplitude"] *= 0.001
        spec = roi_chart(small, roi=roi).to_dict(validate=True)
        y_domain = spec["layer"][-1]["encoding"]["y"]["scale"]["domain"]
        self.assertLess(y_domain[1], 0.01)
        self.assertGreater(y_domain[1], small.amplitude.max())
        self.assertEqual(spec["height"], 280)
        self.assertEqual(spec["layer"][-1]["encoding"]["x"]["title"], "频率 / kHz")
        x_scale = spec["layer"][-1]["encoding"]["x"]["scale"]
        self.assertFalse(x_scale["zero"])
        self.assertEqual(x_scale["domain"], [local.frequency.min(), local.frequency.max()])


if __name__ == "__main__":
    unittest.main()
