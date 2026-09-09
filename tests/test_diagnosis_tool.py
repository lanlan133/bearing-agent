import sys
from pathlib import Path


# ============================================================
# 1. 路径
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

from diagnosis_tool import (
    diagnose_signal,
    save_diagnosis_json,
    print_diagnosis,
)


# ============================================================
# 3. 数据
# ============================================================

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)

REPORT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "diagnosis"
    / "sample_001_diagnosis.json"
)


# ============================================================
# 4. 读取样本
# ============================================================

sample = load_single_sample(
    MAT_FILE,
    sample_index=0
)


signal_data = sample[
    "signal"
]

sampling_rate = sample[
    "sampling_rate"
]


# ============================================================
# 5. 诊断
# ============================================================

result = diagnose_signal(
    signal_data,
    sampling_rate,
    top_n=5
)


# ============================================================
# 6. 打印
# ============================================================

print("\n真实样本信息")

print("-" * 75)

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


print_diagnosis(
    result
)


# ============================================================
# 7. 保存
# ============================================================

save_diagnosis_json(
    result,
    REPORT_FILE
)


print("\n诊断结果已保存:")

print(
    REPORT_FILE
)