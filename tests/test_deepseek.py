import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from llm_reporter import call_deepseek


result = call_deepseek(
    system_prompt="你是工业设备状态监测助手。",
    user_prompt="请只回复：DeepSeek连接成功"
)

print(result)