import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:
    sys.path.insert(
        0,
        str(SRC_DIR),
    )


from agent_tools import (
    execute_tool,
)


MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)


context = {
    "mat_file_path":
        str(MAT_FILE),
}


print(
    "=" * 80
)

print(
    "测试1：inspect_dataset"
)

dataset_result = execute_tool(
    tool_name="inspect_dataset",
    arguments={},
    context=context,
)

print(
    dataset_result
)


print(
    "\n测试2：get_sample_info"
)

sample_result = execute_tool(
    tool_name="get_sample_info",
    arguments={
        "sample_number": 1,
    },
    context=context,
)

print(
    sample_result
)


print(
    "\n测试3：diagnose_sample"
)

diagnosis_result = execute_tool(
    tool_name="diagnose_sample",
    arguments={
        "sample_number": 1,
        "top_n": 5,
    },
    context=context,
)

diagnosis = (
    diagnosis_result[
        "data"
    ][
        "diagnosis"
    ]
)

print(
    "Agent最终决策：",
    diagnosis[
        "decision"
    ]
)

print(
    "Stacking最终概率：",
    diagnosis[
        "stacking_probability"
    ]
)

print(
    "是否需要复核：",
    diagnosis[
        "needs_review"
    ]
)

print(
    "Level-0概率：",
    diagnosis[
        "level_0_probabilities"
    ]
)