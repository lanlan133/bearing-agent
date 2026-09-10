from pathlib import Path
from typing import Any

import numpy as np

from data_loader import (
    load_mat_dataset,
    load_single_sample,
)
from diagnosis_agent import run_diagnosis_agent


# ============================================================
# 1. 工具异常
# ============================================================

class AgentToolError(RuntimeError):
    """Agent工具执行失败。"""


# ============================================================
# 2. 工具定义
#
# 后续会把该列表提供给LLM，让LLM知道有哪些工具可用。
# ============================================================

TOOL_DEFINITIONS = [
    {
        "name": "inspect_dataset",
        "description": (
            "读取当前已经上传的MAT数据集，"
            "返回数据集样本数量。"
        ),
        "arguments": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_sample_info",
        "description": (
            "读取指定样本的设备编号、测点、"
            "转速、采样频率和时间等基本信息，"
            "但不运行诊断模型。"
        ),
        "arguments": {
            "type": "object",
            "properties": {
                "sample_number": {
                    "type": "integer",
                    "description": (
                        "用户看到的样本编号，"
                        "从1开始。"
                    ),
                },
            },
            "required": [
                "sample_number",
            ],
        },
    },
    {
        "name": "diagnose_sample",
        "description": (
            "使用MSM-ADASYN-Stacking"
            "诊断指定样本，返回Stacking最终概率、"
            "三个Level-0概率、Review Policy结果"
            "和判断依据。"
        ),
        "arguments": {
            "type": "object",
            "properties": {
                "sample_number": {
                    "type": "integer",
                    "description": (
                        "用户看到的样本编号，"
                        "从1开始。"
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "返回多少条主要判断依据，"
                        "允许范围1～10。"
                    ),
                    "default": 5,
                },
            },
            "required": [
                "sample_number",
            ],
        },
    },
    {
        "name": "diagnose_range",
        "description": (
            "依次诊断一段连续样本，"
            "用于检查相邻样本或短期概率变化。"
            "起止编号均包含在诊断范围内。"
        ),
        "arguments": {
            "type": "object",
            "properties": {
                "start_number": {
                    "type": "integer",
                    "description": "起始样本编号，从1开始。",
                },
                "end_number": {
                    "type": "integer",
                    "description": (
                        "结束样本编号，"
                        "该编号也会被诊断。"
                    ),
                },
                "top_n": {
                    "type": "integer",
                    "description": (
                        "每条样本返回的判断依据数量。"
                    ),
                    "default": 3,
                },
            },
            "required": [
                "start_number",
                "end_number",
            ],
        },
    },
]


# ============================================================
# 3. 基础参数检查
# ============================================================

def _get_mat_path(
        context: dict[str, Any]
) -> Path:
    """
    MAT路径只从程序上下文中读取。

    不允许LLM自行提供文件路径，
    避免LLM访问非授权文件。
    """

    mat_file_path = context.get(
        "mat_file_path"
    )

    if not mat_file_path:
        raise AgentToolError(
            "当前没有已上传的MAT文件。"
        )

    path = Path(
        mat_file_path
    ).resolve()

    if not path.exists():
        raise AgentToolError(
            f"MAT文件不存在：{path}"
        )

    if path.suffix.lower() != ".mat":
        raise AgentToolError(
            "当前文件不是MAT文件。"
        )

    return path


def _validate_top_n(
        top_n: int
) -> int:

    try:
        top_n = int(
            top_n
        )
    except (TypeError, ValueError) as exc:
        raise AgentToolError(
            "top_n必须是整数。"
        ) from exc

    if top_n < 1 or top_n > 10:
        raise AgentToolError(
            "top_n必须位于1～10。"
        )

    return top_n


def _validate_sample_number(
        sample_number: int,
        total_samples: int,
) -> int:
    """
    将用户使用的1起始编号转换成Python的0起始索引。
    """

    try:
        sample_number = int(
            sample_number
        )
    except (TypeError, ValueError) as exc:
        raise AgentToolError(
            "样本编号必须是整数。"
        ) from exc

    if sample_number < 1:
        raise AgentToolError(
            "样本编号必须从1开始。"
        )

    if sample_number > total_samples:
        raise AgentToolError(
            f"当前数据集共有{total_samples}条数据，"
            f"样本编号{sample_number}超出范围。"
        )

    return sample_number - 1


# ============================================================
# 4. 返回适合LLM读取的样本信息
# ============================================================

def _build_sample_info(
        sample: dict[str, Any],
        sample_number: int,
) -> dict[str, Any]:

    signal = np.asarray(
        sample["signal"]
    ).reshape(-1)

    return {
        "sample_number":
            int(sample_number),

        "equipment_id":
            sample.get("equipment_id"),

        "equipment_name":
            sample.get("equipment_name"),

        "measurement_point":
            sample.get("measurement_point"),

        "speed":
            float(sample.get("speed", 0.0)),

        "sampling_rate":
            float(
                sample.get(
                    "sampling_rate",
                    0.0,
                )
            ),

        "signal_length":
            int(signal.size),

        "timestamp":
            sample.get("timestamp"),

        "source":
            sample.get("source"),

    }


# ============================================================
# 5. 压缩诊断结果
#
# 不把大数组或全部中间数据发给LLM。
# ============================================================
def _compact_report(
        report: dict[str, Any]
) -> dict[str, Any]:

    review_policy = (
        report.get(
            "review_policy"
        )
        or {}
    )

    return {
        "decision":
            report.get("decision"),

        "decision_status":
            report.get(
                "decision_status"
            ),

        "needs_review":
            bool(
                report.get(
                    "needs_review",
                    False,
                )
            ),

        "review_reason":
            report.get(
                "review_reason"
            ),

        "model_name":
            report.get("model_name"),

        "model_diagnosis":
            report.get(
                "model_diagnosis"
            ),

        "stacking_probability":
            float(
                report.get(
                    "stacking_probability",
                    0.0,
                )
            ),

        "classification_threshold":
            float(
                report.get(
                    "classification_threshold",
                    0.5,
                )
            ),

        "level_0_probabilities": {
            "rf":
                float(
                    report.get(
                        "rf_probability",
                        0.0,
                    )
                ),

            "xgboost":
                float(
                    report.get(
                        "xgb_probability",
                        0.0,
                    )
                ),

            "svm":
                float(
                    report.get(
                        "svm_probability",
                        0.0,
                    )
                ),
        },

        "risk_level":
            report.get("risk_level"),

        "review_policy": {
            "lower_threshold":
                review_policy.get(
                    "lower_threshold",
                    0.01,
                ),

            "upper_threshold":
                review_policy.get(
                    "upper_threshold",
                    0.96,
                ),

            "version":
                review_policy.get(
                    "version"
                ),

            "method":
                review_policy.get(
                    "method"
                ),
        },

        "domain":
            report.get(
                "domain",
                {}
            ),

        "top_evidence":
            report.get(
                "top_evidence",
                []
            ),

        "mechanism_explanation":
            report.get(
                "mechanism_explanation"
            ),

        "maintenance_advice":
            report.get(
                "maintenance_advice",
                []
            ),

        "roi":
            report.get(
                "roi",
                {}
            ),

        "gaussian":
            report.get(
                "gaussian",
                {}
            ),
    }


# ============================================================
# 6. 工具一：查看数据集
# ============================================================

def inspect_dataset(
        context: dict[str, Any]
) -> dict[str, Any]:

    mat_path = _get_mat_path(
        context
    )

    data = load_mat_dataset(
        mat_path
    )

    return {
        "success": True,
        "tool": "inspect_dataset",
        "data": {
            "file_name":
                mat_path.name,

            "total_samples":
                int(data.shape[0]),

            "column_count":
                int(data.shape[1]),

            "sample_number_range": {
                "minimum": 1,
                "maximum":
                    int(data.shape[0]),
            },
        },
    }


# ============================================================
# 7. 工具二：读取样本基本信息
# ============================================================

def get_sample_info(
        context: dict[str, Any],
        sample_number: int,
) -> dict[str, Any]:

    mat_path = _get_mat_path(
        context
    )

    data = load_mat_dataset(
        mat_path
    )

    sample_index = (
        _validate_sample_number(
            sample_number,
            int(data.shape[0]),
        )
    )

    sample = load_single_sample(
        mat_path,
        sample_index=sample_index,
    )

    return {
        "success": True,
        "tool": "get_sample_info",
        "data": _build_sample_info(
            sample,
            sample_number,
        ),
    }


# ============================================================
# 8. 工具三：诊断单条样本
# ============================================================

def diagnose_sample(
        context: dict[str, Any],
        sample_number: int,
        top_n: int = 5,
) -> dict[str, Any]:

    mat_path = _get_mat_path(
        context
    )

    top_n = _validate_top_n(
        top_n
    )

    data = load_mat_dataset(
        mat_path
    )

    sample_index = (
        _validate_sample_number(
            sample_number,
            int(data.shape[0]),
        )
    )

    sample = load_single_sample(
        mat_path,
        sample_index=sample_index,
    )

    report = run_diagnosis_agent(
        signal_data=
            sample["signal"],

        sampling_rate=
            sample["sampling_rate"],

        top_n=top_n,
    )

    return {
        "success": True,
        "tool": "diagnose_sample",
        "data": {
            "sample":
                _build_sample_info(
                    sample,
                    sample_number,
                ),

            "diagnosis":
                _compact_report(
                    report
                ),
        },

        # 该字段供网页保存完整结果。
        # 后续发送给LLM时可以删除。
        "_full_report":
            report,
    }


# ============================================================
# 9. 工具四：诊断连续样本
# ============================================================

def diagnose_range(
        context: dict[str, Any],
        start_number: int,
        end_number: int,
        top_n: int = 3,
) -> dict[str, Any]:

    mat_path = _get_mat_path(
        context
    )

    top_n = _validate_top_n(
        top_n
    )

    data = load_mat_dataset(
        mat_path
    )

    total_samples = int(
        data.shape[0]
    )

    start_index = (
        _validate_sample_number(
            start_number,
            total_samples,
        )
    )

    end_index = (
        _validate_sample_number(
            end_number,
            total_samples,
        )
    )

    if start_index > end_index:
        raise AgentToolError(
            "起始样本编号不能大于结束样本编号。"
        )

    sample_count = (
        end_index
        - start_index
        + 1
    )

    # 防止LLM一次调用大量模型预测
    if sample_count > 20:
        raise AgentToolError(
            "单次最多诊断20条数据，"
            f"当前请求诊断{sample_count}条。"
        )

    results = []
    errors = []

    for sample_index in range(
            start_index,
            end_index + 1,
    ):

        sample_number = (
            sample_index + 1
        )

        try:
            sample = load_single_sample(
                mat_path,
                sample_index=sample_index,
            )

            report = run_diagnosis_agent(
                signal_data=
                    sample["signal"],

                sampling_rate=
                    sample["sampling_rate"],

                top_n=top_n,
            )

            results.append({
                "sample_number":
                    sample_number,

                "equipment_id":
                    sample.get(
                        "equipment_id"
                    ),

                "measurement_point":
                    sample.get(
                        "measurement_point"
                    ),

                "timestamp":
                    sample.get(
                        "timestamp"
                    ),

                "decision":
                    report.get(
                        "decision"
                    ),

                "decision_status":
                    report.get(
                        "decision_status"
                    ),

                "needs_review":
                    bool(
                        report.get(
                            "needs_review",
                            False,
                        )
                    ),

                "model_diagnosis":
                    report.get(
                        "model_diagnosis"
                    ),

                "stacking_probability":
                    float(
                        report.get(
                            "stacking_probability",
                            0.0,
                        )
                    ),

                "rf_probability":
                    float(
                        report.get(
                            "rf_probability",
                            0.0,
                        )
                    ),

                "xgb_probability":
                    float(
                        report.get(
                            "xgb_probability",
                            0.0,
                        )
                    ),

                "svm_probability":
                    float(
                        report.get(
                            "svm_probability",
                            0.0,
                        )
                    ),
            })

        except Exception as exc:
            errors.append({
                "sample_number":
                    sample_number,

                "error_type":
                    type(exc).__name__,

                "error_message":
                    str(exc),
            })

    if not results:
        raise AgentToolError(
            "所成功诊断任何样本。"
        )

    probabilities = [
        item["stacking_probability"]
        for item in results
    ]

    highest_result = max(
        results,
        key=lambda item:
            item["stacking_probability"],
    )

    summary = {
        "requested_count":
            sample_count,

        "successful_count":
            len(results),

        "failed_count":
            len(errors),

        "auto_normal_count":
            sum(
                item["decision_status"]
                == "auto_normal"
                for item in results
            ),

        "review_count":
            sum(
                item["decision_status"]
                == "review"
                for item in results
            ),

        "auto_fault_count":
            sum(
                item["decision_status"]
                == "auto_fault"
                for item in results
            ),

        "minimum_probability":
            float(min(probabilities)),

        "maximum_probability":
            float(max(probabilities)),

        "mean_probability":
            float(
                np.mean(
                    probabilities
                )
            ),

        "highest_probability_sample":
            int(
                highest_result[
                    "sample_number"
                ]
            ),
    }

    return {
        "success": True,
        "tool": "diagnose_range",
        "data": {
            "start_number":
                int(start_number),

            "end_number":
                int(end_number),

            "summary":
                summary,

            "results":
                results,

            "errors":
                errors,
        },
    }


# ============================================================
# 10. 工具注册表
# ============================================================

TOOL_REGISTRY = {
    "inspect_dataset":
        inspect_dataset,

    "get_sample_info":
        get_sample_info,

    "diagnose_sample":
        diagnose_sample,

    "diagnose_range":
        diagnose_range,
}


# ============================================================
# 11. 统一工具执行入口
# ============================================================

def execute_tool(
        tool_name: str,
        arguments: dict[str, Any],
        context: dict[str, Any],
) -> dict[str, Any]:
    """
    只允许执行注册表中的工具。
    """

    if tool_name not in TOOL_REGISTRY:
        raise AgentToolError(
            f"不允许执行的工具：{tool_name}"
        )

    if not isinstance(
        arguments,
        dict,
    ):
        raise AgentToolError(
            "工具参数必须是字典。"
        )

    tool_function = (
        TOOL_REGISTRY[
            tool_name
        ]
    )

    try:
        return tool_function(
            context=context,
            **arguments,
        )

    except AgentToolError:
        raise

    except TypeError as exc:
        raise AgentToolError(
            f"工具参数错误：{exc}"
        ) from exc

    except Exception as exc:
        raise AgentToolError(
            f"{tool_name}执行失败："
            f"{type(exc).__name__}: {exc}"
        ) from exc
