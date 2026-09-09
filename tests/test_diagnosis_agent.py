import sys
from pathlib import Path


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# 2. 导入
# ============================================================

from data_loader import load_single_sample

from diagnosis_agent import (
    run_diagnosis_agent,
    print_agent_report,
    save_agent_report,
)


# ============================================================
# 3. 文件
# ============================================================

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "agent"
    / "sample_001_agent_report.json"
)


# ============================================================
# 4. 读取样本
# ============================================================

sample = load_single_sample(
    MAT_FILE,
    sample_index=0
)


# ============================================================
# 5. Agent诊断
# ============================================================

report = run_diagnosis_agent(
    signal_data=sample["signal"],
    sampling_rate=sample["sampling_rate"],
    top_n=5
)


# ============================================================
# 6. 原始信息
# ============================================================

print("\n")

print("=" * 80)

print("真实样本信息")

print("=" * 80)

print(
    f"设备编号: "
    f"{sample['equipment_id']}"
)

print(
    f"设备名称: "
    f"{sample['equipment_name']}"
)

print(
    f"测点: "
    f"{sample['measurement_point']}"
)

print(
    f"真实标签: "
    f"{sample['label']}"
)

print(
    f"采样频率: "
    f"{sample['sampling_rate']} Hz"
)


# ============================================================
# 7. Agent输出
# ============================================================

print_agent_report(
    report
)


# ============================================================
# 8. 保存
# ============================================================

save_agent_report(
    report,
    OUTPUT_FILE
)

print("\nAgent报告已保存:")

print(
    OUTPUT_FILE
)