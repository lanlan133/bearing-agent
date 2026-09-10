import json
from typing import Any

from agent_tools import (
    TOOL_DEFINITIONS,
    TOOL_REGISTRY,
)
from llm_reporter import call_deepseek


# ============================================================
# 1. 规划器异常
# ============================================================

class AgentPlannerError(RuntimeError):
    """LLM规划结果无效。"""


# ============================================================
# 2. 规划器系统提示词
# ============================================================

PLANNER_SYSTEM_PROMPT = """
你是轴承润滑诊断AI Agent的任务规划器。

你的职责是：

1. 理解用户目标；
2. 检查当前任务状态；
3. 从允许的工具中选择下一步行动；
4. 如果信息不足，向用户提问；
5. 如果任务完成，给出最终回答。

重要约束：

1. 你不能自行计算或修改任何模型概率。
2. 你不能修改MSM-ADASYN-Stacking的诊断结果。
3. 你不能覆盖Review Policy的判断。
4. 进入复核区间时，不能自动判定正常或故障。
5. 你不能访问工具列表之外的文件或程序。
6. 每次只能选择一个行动。
7. 必须只返回一个合法JSON对象。
8. 不要使用Markdown代码块。
9. 不要在JSON前后添加解释文字。

action_type只能是：

- tool：调用一个工具
- ask_user：向用户补充提问
- finish：任务已经完成

调用工具时返回：

{
  "action_type": "tool",
  "tool_name": "工具名称",
  "arguments": {},
  "reason": "选择该工具的原因"
}

向用户提问时返回：

{
  "action_type": "ask_user",
  "question": "需要用户补充的问题",
  "reason": "为什么需要该信息"
}

任务完成时返回：

{
  "action_type": "finish",
  "answer": "最终回答",
  "reason": "为什么任务已经完成"
}
"""


# ============================================================
# 3. 提取JSON
# ============================================================

def _extract_json_object(
        text: str
) -> dict[str, Any]:

    if not text:
        raise AgentPlannerError(
            "LLM没有返回任何内容。"
        )

    cleaned = text.strip()

    # 兼容LLM偶尔返回Markdown代码块
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()

        if lines:
            lines = lines[1:]

        if (
            lines
            and lines[-1].strip().startswith("```")
        ):
            lines = lines[:-1]

        cleaned = "\n".join(
            lines
        ).strip()

    start_position = cleaned.find(
        "{"
    )

    if start_position < 0:
        raise AgentPlannerError(
            "LLM返回内容中没有JSON对象。"
        )

    try:
        decoder = json.JSONDecoder()

        result, _ = decoder.raw_decode(
            cleaned[start_position:]
        )

    except json.JSONDecodeError as exc:
        raise AgentPlannerError(
            "LLM返回的JSON无法解析："
            f"{exc}"
        ) from exc

    if not isinstance(
        result,
        dict,
    ):
        raise AgentPlannerError(
            "LLM返回结果必须是JSON对象。"
        )

    return result


# ============================================================
# 4. 验证规划结果
# ============================================================

def validate_action(
        action: dict[str, Any]
) -> dict[str, Any]:

    action_type = action.get(
        "action_type"
    )

    if action_type not in {
        "tool",
        "ask_user",
        "finish",
    }:
        raise AgentPlannerError(
            "action_type必须是tool、"
            "ask_user或finish。"
        )

    reason = str(
        action.get(
            "reason",
            ""
        )
    ).strip()

    if not reason:
        raise AgentPlannerError(
            "规划结果缺少reason。"
        )

    # --------------------------------------------------------
    # 调用工具
    # --------------------------------------------------------

    if action_type == "tool":

        tool_name = action.get(
            "tool_name"
        )

        if tool_name not in TOOL_REGISTRY:
            raise AgentPlannerError(
                f"LLM选择了不允许的工具："
                f"{tool_name}"
            )

        arguments = action.get(
            "arguments",
            {}
        )

        if not isinstance(
            arguments,
            dict,
        ):
            raise AgentPlannerError(
                "arguments必须是JSON对象。"
            )

        return {
            "action_type":
                "tool",

            "tool_name":
                tool_name,

            "arguments":
                arguments,

            "reason":
                reason,
        }

    # --------------------------------------------------------
    # 询问用户
    # --------------------------------------------------------

    if action_type == "ask_user":

        question = str(
            action.get(
                "question",
                ""
            )
        ).strip()

        if not question:
            raise AgentPlannerError(
                "ask_user行动缺少question。"
            )

        return {
            "action_type":
                "ask_user",

            "question":
                question,

            "reason":
                reason,
        }

    # --------------------------------------------------------
    # 完成任务
    # --------------------------------------------------------

    answer = str(
        action.get(
            "answer",
            ""
        )
    ).strip()

    if not answer:
        raise AgentPlannerError(
            "finish行动缺少answer。"
        )

    return {
        "action_type":
            "finish",

        "answer":
            answer,

        "reason":
            reason,
    }


# ============================================================
# 5. 构建安全的任务状态
# ============================================================

def build_planner_context(
        context: dict[str, Any]
) -> dict[str, Any]:
    """
    只把规划所需信息发送给LLM。

    不发送本地文件绝对路径，
    不发送原始振动信号。
    """

    last_report = (
        context.get(
            "last_report"
        )
        or {}
    )

    return {
        "dataset_loaded":
            bool(
                context.get(
                    "mat_file_path"
                )
            ),

        "file_name":
            context.get(
                "mat_file_name"
            ),

        "total_samples":
            context.get(
                "total_samples"
            ),

        "last_sample_number":
            context.get(
                "last_sample_number"
            ),

        "last_diagnosis": {
            "decision":
                last_report.get(
                    "decision"
                ),

            "decision_status":
                last_report.get(
                    "decision_status"
                ),

            "needs_review":
                last_report.get(
                    "needs_review"
                ),

            "model_diagnosis":
                last_report.get(
                    "model_diagnosis"
                ),

            "stacking_probability":
                last_report.get(
                    "stacking_probability"
                ),

            "risk_level":
                last_report.get(
                    "risk_level"
                ),
        } if last_report else None,
    }


# ============================================================
# 6. 压缩工具观察结果
# ============================================================

def compact_observation(
        tool_result: dict[str, Any]
) -> dict[str, Any]:
    """
    删除不应该重新发送给LLM的内部完整报告。
    """

    return {
        key: value
        for key, value in tool_result.items()
        if key != "_full_report"
    }


# ============================================================
# 7. 构建规划提示词
# ============================================================

def build_planner_prompt(
        user_goal: str,
        context: dict[str, Any],
        observations: list[dict[str, Any]] | None = None,
) -> str:

    user_goal = str(
        user_goal
    ).strip()

    if not user_goal:
        raise AgentPlannerError(
            "用户目标不能为空。"
        )

    planner_context = (
        build_planner_context(
            context
        )
    )

    observation_list = (
        observations
        or []
    )

    # 防止上下文无限增长
    recent_observations = (
        observation_list[-6:]
    )

    payload = {
        "user_goal":
            user_goal,

        "current_context":
            planner_context,

        "available_tools":
            TOOL_DEFINITIONS,

        "recent_observations":
            recent_observations,
    }

    return f"""
下面是当前Agent任务：

{json.dumps(
    payload,
    ensure_ascii=False,
    indent=2,
)}

请选择最直接、最必要的下一步行动。

规则：

1. 尚未上传数据集时，应使用ask_user要求上传MAT文件。
2. 用户询问数据集样本数量时，选择inspect_dataset。
3. 用户只想查看样本信息时，选择get_sample_info。
4. 用户要求诊断一条样本时，选择diagnose_sample。
5. 用户要求分析连续样本或相邻趋势时，选择diagnose_range。
6. 已经获得充分结果时，选择finish。
7. 不要重复执行recent_observations中已经成功完成的相同行动。
8. 只输出JSON。
"""


# ============================================================
# 8. LLM规划入口
# ============================================================

def plan_next_action(
        user_goal: str,
        context: dict[str, Any],
        observations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:

    user_prompt = build_planner_prompt(
        user_goal=user_goal,
        context=context,
        observations=observations,
    )

    response_text = call_deepseek(
        system_prompt=
            PLANNER_SYSTEM_PROMPT,

        user_prompt=
            user_prompt,
    )

    action = _extract_json_object(
        response_text
    )

    return validate_action(
        action
    )