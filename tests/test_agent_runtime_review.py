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
    user_goal=(
        "请诊断第5条数据，"
        "并说明是否需要人工复核。"
    ),
    context=context,
    max_steps=5,
)


print(
    "=" * 80
)

print(
    "人工复核样本测试"
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


assert result["status"] == "completed"
assert "复核" in result["answer"]
