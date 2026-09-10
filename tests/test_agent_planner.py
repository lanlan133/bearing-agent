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


from agent_planner import (
    plan_next_action,
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


action = plan_next_action(
    user_goal=
        "请诊断第1条数据",

    context=
        context,

    observations=
        [],
)


print(
    "=" * 80
)

print(
    "LLM规划结果"
)

print(
    "=" * 80
)

print(
    json.dumps(
        action,
        ensure_ascii=False,
        indent=2,
    )
)