"""Session-owned state for the diagnostic workbench, without a Streamlit dependency."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path
import re
import tempfile

import numpy as np

from data_loader import load_mat_dataset, load_single_sample
from msm_features import analyze_signal


def explicit_sample_number(text):
    """Recognize a standalone sample request; leave general questions to the LLM."""
    match = re.fullmatch(r"\s*(?:第\s*)?(\d+)\s*(?:条|个)?\s*", str(text))
    return int(match.group(1)) if match else None


@dataclass
class WorkbenchState:
    mat_file_path: str | None = None
    file_name: str | None = None
    dataset: np.ndarray | None = None
    file_digest: str | None = None
    selected_index: int = 0
    sample: dict | None = None
    analysis: dict | None = None
    report: dict | None = None
    error: str | None = None
    last_agent_result: dict | None = None
    upload_directory: str | None = None
    messages: list = field(default_factory=list)
    reports: OrderedDict = field(default_factory=OrderedDict)

    @property
    def total_samples(self):
        return len(self.dataset) if self.dataset is not None else 0

    @property
    def sample_number(self):
        return self.selected_index + 1

    def activate_path(self, path, display_name=None, digest=None):
        """Parse once, validate before replacing the active file, reset stale results."""
        path = Path(path).resolve()
        dataset = load_mat_dataset(path)
        if dataset.ndim != 2 or dataset.shape[0] == 0 or dataset.shape[1] < 12:
            raise ValueError("MAT 中的 data 必须是非空、至少 12 列的样本表。")
        self.mat_file_path = str(path)
        self.file_name = display_name or path.name
        self.file_digest = digest
        self.dataset = dataset
        self.selected_index = 0
        self.sample = self.analysis = self.report = self.last_agent_result = None
        self.error = None
        self.reports.clear()
        self.messages.clear()
        self.add_message("assistant", f"已读取 {self.file_name}，共 {len(dataset)} 条数据。请选择样本并开始诊断。")

    def activate_upload(self, name, content, upload_root):
        """Keep identical filenames in different sessions isolated."""
        if Path(name).suffix.lower() != ".mat":
            raise ValueError("请上传 .mat 文件。")
        digest = hashlib.sha256(content).hexdigest()
        if self.dataset is not None and digest == self.file_digest:
            return False
        root = Path(upload_root)
        root.mkdir(parents=True, exist_ok=True)
        if self.upload_directory is None:
            self.upload_directory = tempfile.mkdtemp(prefix="session_", dir=root)
        path = Path(self.upload_directory) / f"{digest}.mat"
        path.write_bytes(content)
        self.activate_path(path, display_name=Path(name).name, digest=digest)
        return True

    def select_sample(self, number):
        number_value = float(number)
        if not math.isfinite(number_value) or not number_value.is_integer():
            raise ValueError("样本编号必须是整数。")
        number = int(number_value)
        if number < 1 or number > self.total_samples:
            raise ValueError(f"样本编号必须位于 1～{self.total_samples}。")
        index = number - 1
        if index != self.selected_index:
            self.selected_index = index
            self.sample = self.analysis = self.last_agent_result = None
            self.report = self.reports.get(index)
            self.error = None

    def ensure_signal(self):
        if self.dataset is None:
            return None
        if self.sample is None:
            self.sample = load_single_sample(
                self.mat_file_path, self.selected_index, dataset=self.dataset
            )
        if self.analysis is None:
            self.analysis = analyze_signal(self.sample["signal"], self.sample["sampling_rate"])
        return self.analysis

    def context(self):
        # These arrays remain local. build_planner_context uses a whitelist and
        # never serializes the private cached dataset into a model prompt.
        return {
            "mat_file_path": self.mat_file_path,
            "mat_file_name": self.file_name,
            "total_samples": self.total_samples,
            "last_sample_number": self.sample_number,
            "last_report": self.report,
            "_mat_dataset": self.dataset,
            "_mat_dataset_path": self.mat_file_path,
            "_conversation_history": [
                {"role": m["role"], "content": m["content"]}
                for m in self.messages
                if m.get("sample_number") == self.sample_number
            ][-10:],
        }

    def remember_report(self, report):
        self.report = report
        self.reports[self.selected_index] = report
        self.reports.move_to_end(self.selected_index)
        while len(self.reports) > 8:
            self.reports.popitem(last=False)
        self.error = None

    def diagnose(self):
        if self.dataset is None:
            raise ValueError("请先上传 MAT 文件。")
        if self.report is not None:
            return self.report
        from agent_tools import diagnose_sample
        result = diagnose_sample(self.context(), self.sample_number, top_n=5)
        self.remember_report(result["_full_report"])
        return self.report

    def add_message(self, role, content):
        self.messages.append({
            "role": role, "content": str(content),
            "file_name": self.file_name, "sample_number": self.sample_number,
        })
        self.messages[:] = self.messages[-60:]

    def ask(self, question):
        from agent_runtime import run_agent_task
        if self.dataset is None:
            raise ValueError("请先上传 MAT 文件。")
        self.add_message("user", question)
        result = run_agent_task(question, self.context(), max_steps=5)
        context = result.get("context") or {}
        diagnosed = any(
            item.get("success") and item.get("tool") == "diagnose_sample"
            for item in result.get("observations", [])
        )
        if diagnosed and context.get("last_report") is not None:
            self.select_sample(context["last_sample_number"])
            self.remember_report(context["last_report"])
        self.last_agent_result = result
        answer = result.get("answer") or "Agent 未返回回答。"
        if result.get("status") in ("failed", "max_steps_reached"):
            answer = "本次 Agent 未完成请求：" + answer
        self.add_message("assistant", answer)
        return result
