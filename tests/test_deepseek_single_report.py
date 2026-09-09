import sys
from pathlib import Path

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


from data_loader import load_single_sample

from diagnosis_agent import (
    run_diagnosis_agent,
)

from llm_reporter import (
    generate_single_report,
    save_text_report,
)


MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "reports"
    / "deepseek"
    / "sample_1463_deepseek_report.md"
)


# ============================================================
# 1. 读取一个高风险样本
# ============================================================

sample_index = 1463

sample = load_single_sample(
    MAT_FILE,
    sample_index=sample_index
)


# ============================================================
# 2. 本地 Diagnosis Agent
# ============================================================

report = run_diagnosis_agent(
    signal_data=sample["signal"],
    sampling_rate=sample["sampling_rate"],
    top_n=5
)


# ============================================================
# 3. 给 DeepSeek 的样本信息
# ============================================================

sample_info = {

    "sample_index":
        sample_index,

    "equipment_id":
        sample["equipment_id"],

    "equipment_name":
        sample["equipment_name"],

    "measurement_point":
        sample["measurement_point"],

    "sampling_rate":
        sample["sampling_rate"],

    "speed":
        sample["speed"],

    "timestamp":
        sample["timestamp"],

    "true_label":
        sample["label"],
}


# ============================================================
# 4. DeepSeek 报告
# ============================================================

llm_result = generate_single_report(
    report=report,
    sample_info=sample_info
)


print("\n")
print("=" * 80)
print("DeepSeek 单样本诊断报告测试")
print("=" * 80)

print(
    f"调用成功: "
    f"{llm_result['success']}"
)

print(
    f"报告来源: "
    f"{llm_result['source']}"
)

print(
    f"模型: "
    f"{llm_result['model']}"
)

if llm_result["error"]:

    print(
        f"API错误: "
        f"{llm_result['error']}"
    )


print("\n")
print(
    llm_result[
        "report"
    ]
)


# ============================================================
# 5. 保存 Markdown
# ============================================================

save_text_report(
    llm_result["report"],
    OUTPUT_FILE
)

print("\n报告已保存:")
print(
    OUTPUT_FILE
)