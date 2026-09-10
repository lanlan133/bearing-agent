import json
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


from agent_runtime import (
    run_agent_task,
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

    "mat_file_name":
        MAT_FILE.name,

    "total_samples":
        1753,

    "last_sample_number":
        None,

    "last_report":
        None,
}


result = run_agent_task(
    user_goal=
        "请诊断第1条数据，"
        "并说明是否需要人工复核。",

    context=
        context,

    max_steps=
        5,
)


print(
    "=" * 80
)

print(
    "Agent执行结果"
)

print(
    "=" * 80
)

print(
    "状态：",
    result["status"]
)

print(
    "执行步骤：",
    result["steps"]
)

print(
    "\n最终回答："
)

print(
    result["answer"]
)

print(
    "\n工具观察记录："
)

print(
    json.dumps(
        result["observations"],
        ensure_ascii=False,
        indent=2,
    )
)


# ============================================================
# 追问测试：应直接使用上一条报告，不重复运行诊断工具
# ============================================================

follow_up_result = run_agent_task(
    user_goal=
        "为什么这个结果不需要人工复核？",

    context=
        result["context"],

    max_steps=
        3,
)


print(
    "\n" + "=" * 80
)

print(
    "上一条诊断结果追问测试"
)

print(
    "=" * 80
)

print(
    "追问状态：",
    follow_up_result["status"]
)

print(
    "追问步骤：",
    follow_up_result["steps"]
)

print(
    "\n追问回答："
)

print(
    follow_up_result["answer"]
)

print(
    "\n追问工具记录："
)

print(
    json.dumps(
        follow_up_result["observations"],
        ensure_ascii=False,
        indent=2,
    )
)
