import json
from typing import Any

from agent_planner import (
    AgentPlannerError,
    compact_observation,
    plan_next_action,
)
from agent_tools import (
    AgentToolError,
    execute_tool,
)


# ============================================================
# 1. 判断用户目标是否需要模型诊断
# ============================================================

def _requires_diagnosis(
        user_goal: str
) -> bool:

    keywords = [
        "诊断",
        "判断",
        "故障",
        "润滑",
        "风险",
        "概率",
        "复核",
    ]

    return any(
        keyword in user_goal
        for keyword in keywords
    )


# ============================================================
# 2. 检查是否已经执行诊断工具
# ============================================================

def _has_diagnosis_observation(
        observations: list[dict[str, Any]]
) -> bool:

    diagnosis_tools = {
        "diagnose_sample",
        "diagnose_range",
    }

    for observation in observations:

        if not observation.get(
                "success",
                False,
        ):
            continue

        if observation.get(
                "tool"
        ) in diagnosis_tools:
            return True

    return False


# ============================================================
# 3. 检查LLM是否可以结束任务
# ============================================================

def _validate_finish(
        user_goal: str,
        answer: str,
        context: dict[str, Any],
        observations: list[dict[str, Any]],
) -> tuple[bool, str | None]:

    # --------------------------------------------------------
    # 需要诊断的任务必须先运行诊断工具
    # --------------------------------------------------------

    has_previous_report = bool(
        context.get(
            "last_report"
        )
    )

    explicit_new_diagnosis = (
        user_goal.strip().isdigit()
        or any(
            keyword in user_goal
            for keyword in [
                "诊断第",
                "诊断样本",
                "重新诊断",
                "分析第",
                "检查第",
                "判断第",
            ]
        )
    )

    requires_tool = (
        explicit_new_diagnosis
        or (
            _requires_diagnosis(
                user_goal
            )
            and not has_previous_report
        )
    )

    if (
        requires_tool
        and not _has_diagnosis_observation(
            observations
        )
    ):
        return (
            False,
            "用户要求进行新的诊断，"
            "但尚未成功执行诊断工具。",
        )

    # --------------------------------------------------------
    # Review样本的回答必须明确包含复核提示
    # --------------------------------------------------------

    last_report = (
        context.get(
            "last_report"
        )
        or {}
    )

    if last_report.get(
            "needs_review",
            False,
    ):
        if "复核" not in answer:
            return (
                False,
                "当前样本需要人工复核，"
                "最终回答必须明确说明复核要求。",
            )

    return True, None


# ============================================================
# 4. 根据工具结果更新任务上下文
# ============================================================

def _update_context(
        context: dict[str, Any],
        tool_result: dict[str, Any],
) -> None:

    tool_name = tool_result.get(
        "tool"
    )

    data = (
        tool_result.get(
            "data"
        )
        or {}
    )

    # --------------------------------------------------------
    # 数据集信息
    # --------------------------------------------------------

    if tool_name == "inspect_dataset":

        context["total_samples"] = (
            data.get(
                "total_samples"
            )
        )

    # --------------------------------------------------------
    # 单样本诊断
    # --------------------------------------------------------

    if tool_name == "diagnose_sample":

        full_report = tool_result.get(
            "_full_report"
        )

        if full_report is not None:
            context["last_report"] = (
                full_report
            )

        sample_info = (
            data.get(
                "sample"
            )
            or {}
        )

        context["last_sample_number"] = (
            sample_info.get(
                "sample_number"
            )
        )


# ============================================================
# 5. 生成工具调用唯一标识
# ============================================================

def _action_signature(
        action: dict[str, Any]
) -> str:

    payload = {
        "tool_name":
            action.get(
                "tool_name"
            ),

        "arguments":
            action.get(
                "arguments",
                {}
            ),
    }

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
    )


# ============================================================
# 6. Agent运行入口
# ============================================================

def run_agent_task(
        user_goal: str,
        context: dict[str, Any],
        max_steps: int = 5,
) -> dict[str, Any]:
    """
    执行：

    LLM规划
    -> 工具执行
    -> 观察结果
    -> LLM再次规划
    -> 完成或询问用户
    """

    user_goal = str(
        user_goal
    ).strip()

    if not user_goal:
        return {
            "status": "failed",
            "answer": "用户目标不能为空。",
            "steps": 0,
            "observations": [],
            "context": dict(context),
        }

    max_steps = int(
        max_steps
    )

    if max_steps < 1 or max_steps > 10:
        raise ValueError(
            "max_steps必须位于1～10。"
        )

    task_context = dict(
        context
    )

    observations = []

    executed_actions = set()

    # ========================================================
    # 规划与执行循环
    # ========================================================

    for step_number in range(
            1,
            max_steps + 1,
    ):

        # ----------------------------------------------------
        # A. LLM规划
        # ----------------------------------------------------

        try:
            action = plan_next_action(
                user_goal=user_goal,
                context=task_context,
                observations=observations,
            )

        except AgentPlannerError as exc:

            return {
                "status": "failed",
                "answer": (
                    "LLM规划失败："
                    f"{exc}"
                ),
                "steps": step_number,
                "observations": observations,
                "context": task_context,
            }

        except Exception as exc:

            return {
                "status": "failed",
                "answer": (
                    "调用规划模型失败："
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
                "steps": step_number,
                "observations": observations,
                "context": task_context,
            }

        action_type = action[
            "action_type"
        ]

        # ----------------------------------------------------
        # B. LLM需要询问用户
        # ----------------------------------------------------

        if action_type == "ask_user":

            return {
                "status": "waiting_user",

                "answer":
                    action["question"],

                "reason":
                    action["reason"],

                "steps":
                    step_number,

                "observations":
                    observations,

                "context":
                    task_context,
            }

        # ----------------------------------------------------
        # C. LLM认为任务已经完成
        # ----------------------------------------------------

        if action_type == "finish":

            allowed, rejection_reason = (
                _validate_finish(
                    user_goal=user_goal,
                    answer=action["answer"],
                    context=task_context,
                    observations=observations,
                )
            )

            if allowed:
                return {
                    "status": "completed",

                    "answer":
                        action["answer"],

                    "reason":
                        action["reason"],

                    "steps":
                        step_number,

                    "observations":
                        observations,

                    "context":
                        task_context,
                }

            # LLM过早结束，记录反馈后重新规划
            observations.append({
                "success": False,
                "tool": "completion_guard",
                "error": rejection_reason,
            })

            continue

        # ----------------------------------------------------
        # D. 工具调用
        # ----------------------------------------------------

        signature = _action_signature(
            action
        )

        # 防止反复调用相同工具和参数
        if signature in executed_actions:

            observations.append({
                "success": False,

                "tool":
                    action["tool_name"],

                "error":
                    "该工具和参数已经执行过，"
                    "请根据已有结果继续规划。",
            })

            continue

        executed_actions.add(
            signature
        )

        try:
            tool_result = execute_tool(
                tool_name=
                    action["tool_name"],

                arguments=
                    action["arguments"],

                context=
                    task_context,
            )

            # 完整报告留在程序上下文中
            _update_context(
                task_context,
                tool_result,
            )

            # 只把压缩结果交给LLM
            observation = (
                compact_observation(
                    tool_result
                )
            )

            observations.append(
                observation
            )

        except AgentToolError as exc:

            observations.append({
                "success": False,

                "tool":
                    action["tool_name"],

                "error":
                    str(exc),
            })

        except Exception as exc:

            observations.append({
                "success": False,

                "tool":
                    action["tool_name"],

                "error": (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
            })

    # ========================================================
    # 达到最大步骤数
    # ========================================================

    return {
        "status": "max_steps_reached",

        "answer": (
            "任务未在规定步骤内完成。"
            "请缩小诊断范围或补充更明确的要求。"
        ),

        "steps":
            max_steps,

        "observations":
            observations,

        "context":
            task_context,
    }
