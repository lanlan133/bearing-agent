from pathlib import Path
import json

import numpy as np

from diagnosis_tool import (
    diagnose_signal,
    save_diagnosis_json,
)


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]
# ============================================================
# Review Policy
# ============================================================

REVIEW_POLICY_FILE = (
    PROJECT_ROOT
    / "models"
    / "review_policy.json"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "agent"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 当前模型适用域
# ============================================================

MODEL_VALIDATED_SAMPLING_RATE = 10240.0
# ============================================================
# Review Policy 加载
# ============================================================

def load_review_policy(
        policy_file=REVIEW_POLICY_FILE
):
    """
    加载基于 Paper Stacking 多随机种子留出验证冻结的 Review Policy。

    该 Policy 不是新的分类模型，
    而是 MSM-ADASYN-Stacking 输出之后的 Agent 决策层。
    """

    policy_file = Path(
        policy_file
    )

    if not policy_file.exists():

        raise FileNotFoundError(
            f"找不到 Review Policy: "
            f"{policy_file}"
        )

    with open(
        policy_file,
        "r",
        encoding="utf-8"
    ) as f:

        policy = json.load(
            f
        )

    required_keys = [
        "lower_threshold",
        "upper_threshold",
    ]

    for key in required_keys:

        if key not in policy:

            raise KeyError(
                f"Review Policy 缺少字段: {key}"
            )

    return policy


# ============================================================
# 应用 Review Policy
# ============================================================

def apply_review_policy(
        probability,
        policy
):
    """
    三态决策：

    p < lower
        -> 自动判定非轴承润滑不良

    lower <= p <= upper
        -> 建议复核

    p > upper
        -> 自动判定轴承润滑不良
    """

    probability = float(
        probability
    )

    lower = float(
        policy[
            "lower_threshold"
        ]
    )

    upper = float(
        policy[
            "upper_threshold"
        ]
    )

    if probability < lower:

        return {
            "decision":
                "非轴承润滑不良",

            "decision_status":
                "auto_normal",

            "needs_review":
                False,

            "review_reason":
                None,

            "lower_threshold":
                lower,

            "upper_threshold":
                upper,
        }

    if probability > upper:

        return {
            "decision":
                "轴承润滑不良",

            "decision_status":
                "auto_fault",

            "needs_review":
                False,

            "review_reason":
                None,

            "lower_threshold":
                lower,

            "upper_threshold":
                upper,
        }

    return {
        "decision":
            "建议复核",

        "decision_status":
            "review",

        "needs_review":
            True,

        "review_reason":
            (
                f"Stacking 润滑失效概率 "
                f"{probability:.4f} "
                f"位于冻结的复核区间 "
                f"[{lower:.2f}, {upper:.2f}] 内。"
            ),

        "lower_threshold":
            lower,

        "upper_threshold":
            upper,
    }


# ============================================================
# 3. 特征组映射
# ============================================================

FEATURE_GROUPS = {

    "频谱能量迁移": {
        "f1",
        "f2",
        "f3",
        "f4",
        "f5",
    },

    "宽带扩展与统计分散": {
        "f6",
        "f7",
        "f8",
        "f9",
    },

    "冲击-摩擦包络": {
        "f10",
        "f11",
        "f12",
        "f13",
        "f14",
    },

    "高频频谱轮廓": {
        "f15",
        "f16",
        "f17",
    },
}


# ============================================================
# 4. 找特征属于哪个机理组
# ============================================================

def get_feature_group(
        feature_name
):

    for (
        group_name,
        features
    ) in FEATURE_GROUPS.items():

        if feature_name in features:

            return group_name

    return "其他"


# ============================================================
# 5. 模型适用域检查
# ============================================================

def check_model_domain(
        sampling_rate
):
    """
    当前主模型是在 10240 Hz 数据上完成设备级验证的。

    因此：
    - 10240 Hz -> validated
    - 其他采样率 -> out_of_domain
    """

    sampling_rate = float(
        sampling_rate
    )

    if np.isclose(
        sampling_rate,
        MODEL_VALIDATED_SAMPLING_RATE
    ):

        return {
            "validated":
                True,

            "status":
                "validated",

            "message":
                (
                    "当前样本采样频率位于模型主要验证域内。"
                ),
        }

    return {
        "validated":
            False,

        "status":
            "out_of_domain",

        "message":
            (
                f"当前样本采样频率为 "
                f"{sampling_rate:.2f} Hz，"
                f"而当前主模型主要基于 "
                f"{MODEL_VALIDATED_SAMPLING_RATE:.0f} Hz "
                f"数据完成设备级验证。"
                f"本次结果可作为辅助参考，"
                f"不应与验证域内结果等同解释。"
            ),
    }


# ============================================================
# 6. 风险概率描述
# ============================================================

def probability_description(
        probability
):

    probability = float(
        probability
    )

    if probability >= 0.95:

        return (
            "模型对轴承润滑不良的判定非常强"
        )

    elif probability >= 0.85:

        return (
            "模型对轴承润滑不良的判定较强"
        )

    elif probability >= 0.60:

        return (
            "模型存在中等程度的轴承润滑不良倾向"
        )

    elif probability >= 0.40:

        return (
            "模型处于相对不确定区域"
        )

    else:

        return (
            "模型更倾向于非轴承润滑不良"
        )


# ============================================================
# 7. 提取正向 / 反向证据
# ============================================================

def split_evidence(
        evidence
):

    positive = []

    negative = []

    for item in evidence:

        if item[
            "shap_value"
        ] > 0:

            positive.append(
                item
            )

        elif item[
            "shap_value"
        ] < 0:

            negative.append(
                item
            )

    return (
        positive,
        negative
    )


# ============================================================
# 8. 汇总主要机理组
# ============================================================

def summarize_feature_groups(
        evidence
):

    group_scores = {}

    for item in evidence:

        group = get_feature_group(
            item[
                "feature"
            ]
        )

        group_scores.setdefault(
            group,
            0.0
        )

        group_scores[
            group
        ] += abs(
            float(
                item[
                    "shap_value"
                ]
            )
        )

    ranked = sorted(
        group_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return ranked


# ============================================================
# 9. 生成机理解释
# ============================================================

def build_mechanism_explanation(
        diagnosis_result
):

    evidence = diagnosis_result[
        "top_evidence"
    ]

    positive, negative = split_evidence(
        evidence
    )

    group_ranking = summarize_feature_groups(
        evidence
    )

    lines = []

    # --------------------------------------------------------
    # 主导特征组
    # --------------------------------------------------------

    if group_ranking:

        main_group = group_ranking[0][0]

        lines.append(
            f"本次模型判定主要受到"
            f"“{main_group}”相关特征影响。"
        )

    # --------------------------------------------------------
    # 正向证据
    # --------------------------------------------------------

    if positive:

        lines.append(
            "主要支持轴承润滑不良判定的证据包括："
        )

        for item in positive[:3]:

            lines.append(
                (
                    f"{item['feature']} "
                    f"{item['feature_name']} "
                    f"(SHAP={item['shap_value']:+.4f})，"
                    f"{item['mechanism']}。"
                )
            )

    # --------------------------------------------------------
    # 反向证据
    # --------------------------------------------------------

    if negative:

        lines.append(
            "同时存在部分反向证据："
        )

        for item in negative[:2]:

            lines.append(
                (
                    f"{item['feature']} "
                    f"{item['feature_name']} "
                    f"(SHAP={item['shap_value']:+.4f})。"
                )
            )

    # --------------------------------------------------------
    # ROI
    # --------------------------------------------------------

    roi = diagnosis_result[
        "roi"
    ]

    if roi[
        "active"
    ]:

        lines.append(
            (
                f"动态高频ROI已激活，"
                f"主要频带约为 "
                f"{roi['start']:.1f}"
                f"–"
                f"{roi['end']:.1f} Hz，"
                f"主峰约 "
                f"{roi['peak_frequency']:.1f} Hz。"
            )
        )

    else:

        lines.append(
            "当前样本动态高频ROI未激活。"
        )

    # --------------------------------------------------------
    # Gaussian
    # --------------------------------------------------------

    gaussian = diagnosis_result[
        "gaussian"
    ]

    if gaussian[
        "success"
    ]:

        lines.append(
            (
                f"高频轮廓拟合成功，"
                f"sigma="
                f"{gaussian['sigma']:.1f} Hz，"
                f"RMSE="
                f"{gaussian['rmse']:.4f}。"
            )
        )

    else:

        lines.append(
            "当前样本高斯频谱轮廓拟合未成功。"
        )

    lines.append(
        (
            "上述内容反映的是模型判别证据与"
            "频谱统计特征之间的对应关系，"
            "不应单独视为润滑状态的物理因果证明。"
        )
    )

    return "\n".join(
        lines
    )


# ============================================================
# 10. 维护建议
# ============================================================

def build_maintenance_advice(
        diagnosis_result,
        domain_status
):

    probability = float(
        diagnosis_result[
            "stacking_probability"
        ]
    )

    prediction = int(
        diagnosis_result[
            "prediction"
        ]
    )

    # --------------------------------------------------------
    # 适用域外
    # --------------------------------------------------------

    if not domain_status[
        "validated"
    ]:

        return [
            (
                "当前样本不在模型主要验证采样率范围内，"
                "建议优先复核采集参数或使用匹配采样率的模型。"
            ),
            (
                "可结合原始时域波形、频谱和设备运行记录"
                "进行人工复核。"
            ),
        ]

    # --------------------------------------------------------
    # 高风险
    # --------------------------------------------------------

    if (
        prediction == 1
        and probability >= 0.85
    ):

        return [
            (
                "建议尽快复核轴承润滑状态，"
                "检查润滑脂/润滑油供给是否充足。"
            ),
            (
                "检查润滑系统供油、油路、润滑周期"
                "及润滑剂状态。"
            ),
            (
                "结合温度、噪声、振动趋势及历史记录"
                "进行二次确认。"
            ),
            (
                "若风险连续多次保持较高水平，"
                "建议安排维护检查。"
            ),
        ]

    # --------------------------------------------------------
    # 中风险
    # --------------------------------------------------------

    elif probability >= 0.60:

        return [
            (
                "建议增加该设备近期振动监测频率。"
            ),
            (
                "关注高频能量、谱底及频谱宽带化特征"
                "是否持续增强。"
            ),
            (
                "结合温度和润滑维护记录进行复核。"
            ),
        ]

    # --------------------------------------------------------
    # 低风险
    # --------------------------------------------------------

    else:

        return [
            (
                "当前模型未显示强烈的轴承润滑不良风险。"
            ),
            (
                "建议继续按照既定周期进行趋势监测。"
            ),
            (
                "如果振动、温度或噪声明显变化，"
                "仍应重新诊断。"
            ),
        ]


# ============================================================
# 11. 生成完整 Agent 报告
# ============================================================

def build_agent_report(
        diagnosis_result
):

    # ========================================================
    # 1. 模型适用域
    # ========================================================

    domain_status = check_model_domain(
        diagnosis_result[
            "sampling_rate"
        ]
    )

    # ========================================================
    # 2. Stacking 最终概率
    # ========================================================

    probability = float(
        diagnosis_result[
            "stacking_probability"
        ]
    )

    # ========================================================
    # 3. 原始二分类模型输出
    # ========================================================

    model_prediction = int(
        diagnosis_result[
            "prediction"
        ]
    )

    model_diagnosis = diagnosis_result[
        "diagnosis"
    ]

    # ========================================================
    # 4. Review Policy
    # ========================================================

    policy = load_review_policy()

    review_decision = apply_review_policy(
        probability,
        policy
    )

    # ========================================================
    # 5. 机理解释
    # ========================================================

    mechanism = build_mechanism_explanation(
        diagnosis_result
    )

    # ========================================================
    # 6. 维护建议
    #
    # 注意：
    # 这里仍使用原始模型诊断结果生成基础建议，
    # 后面如果进入 review 再附加复核建议。
    # ========================================================

    maintenance = build_maintenance_advice(
        diagnosis_result,
        domain_status
    )

    if review_decision[
        "needs_review"
    ]:

        maintenance = [
            (
                "当前结果处于模型复核区间，"
                "不建议直接依据单次自动诊断执行维护决策。"
            ),
            (
                "建议结合原始时域波形、频谱、"
                "动态ROI及近期振动趋势进行二次复核。"
            ),
            (
                "建议结合温度、噪声、润滑维护记录"
                "和设备运行工况进行交叉确认。"
            ),
        ] + maintenance

    # ========================================================
    # 7. 最终Agent结果
    # ========================================================

    report = {

        # ----------------------------------------------------
        # Agent 最终三态决策
        # ----------------------------------------------------

        "diagnosis":
            review_decision[
                "decision"
            ],

        "decision":
            review_decision[
                "decision"
            ],

        "decision_status":
            review_decision[
                "decision_status"
            ],

        "needs_review":
            review_decision[
                "needs_review"
            ],

        "review_reason":
            review_decision[
                "review_reason"
            ],

        # ----------------------------------------------------
        # 保留 Stacking 与 Level-0 输出，避免丢失模型事实
        # ----------------------------------------------------

        "model_diagnosis":
            model_diagnosis,

        "model_prediction":
            model_prediction,

        "risk_probability":
            probability,

        "stacking_probability":
            probability,

        "model_name":
            diagnosis_result.get(
                "model_name",
                "MSM-ADASYN-Stacking"
            ),

        "model_version":
            diagnosis_result.get(
                "model_version",
                "unknown"
            ),

        "classification_threshold":
            diagnosis_result.get(
                "classification_threshold",
                0.5
            ),

        "base_probabilities":
            diagnosis_result[
                "base_probabilities"
            ],

        "rf_probability":
            diagnosis_result[
                "rf_probability"
            ],

        "xgb_probability":
            diagnosis_result[
                "xgb_probability"
            ],

        "svm_probability":
            diagnosis_result[
                "svm_probability"
            ],

        "base_probability_mean":
            diagnosis_result.get(
                "base_probability_mean"
            ),

        "base_probability_std":
            diagnosis_result.get(
                "base_probability_std"
            ),

        "model_summary":
            diagnosis_result.get(
                "summary"
            ),

        "risk_level":
            diagnosis_result[
                "risk_level"
            ],

        "probability_interpretation":
            probability_description(
                probability
            ),

        # ----------------------------------------------------
        # Policy信息
        # ----------------------------------------------------

        "review_policy": {

            "lower_threshold":
                review_decision[
                    "lower_threshold"
                ],

            "upper_threshold":
                review_decision[
                    "upper_threshold"
                ],

            "version":
                policy.get(
                    "version",
                    "unknown"
                ),

            "method":
                policy.get(
                    "method",
                    "unknown"
                ),

            "probability_field":
                policy.get(
                    "probability_field",
                    "stacking_probability"
                ),

            "validation_metrics":
                policy.get(
                    "validation_metrics",
                    {}
                ),
        },

        # ----------------------------------------------------
        # 适用域
        # ----------------------------------------------------

        "domain":
            domain_status,

        # ----------------------------------------------------
        # 可解释诊断
        # ----------------------------------------------------

        "mechanism_explanation":
            mechanism,

        "maintenance_advice":
            maintenance,

        "top_evidence":
            diagnosis_result[
                "top_evidence"
            ],

        "roi":
            diagnosis_result[
                "roi"
            ],

        "gaussian":
            diagnosis_result[
                "gaussian"
            ],

        "features":
            diagnosis_result[
                "features"
            ],
    }

    return report


# ============================================================
# 12. Agent 主入口
# ============================================================

def run_diagnosis_agent(
        signal_data,
        sampling_rate,
        top_n=5
):
    """
    Agent的核心入口。

    后续无论接命令行、Web API、LLM、Streamlit，
    都可以直接调用这里。
    """

    # --------------------------------------------------------
    # Tool 层
    # --------------------------------------------------------

    diagnosis_result = diagnose_signal(
        signal_data,
        sampling_rate,
        top_n=top_n
    )

    # --------------------------------------------------------
    # Agent解释层
    # --------------------------------------------------------

    agent_report = build_agent_report(
        diagnosis_result
    )

    return agent_report


# ============================================================
# 13. 保存 Agent JSON
# ============================================================

def save_agent_report(
        report,
        output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# 14. 打印 Agent 报告
# ============================================================

def print_agent_report(
        report
):

    print("\n")

    print(
        "=" * 80
    )

    print(
        "轴承润滑诊断 Agent"
    )

    print(
        "=" * 80
    )

    print(
        f"Agent最终决策: "
        f"{report['diagnosis']}"
    )

    print(
        f"决策状态: "
        f"{report['decision_status']}"
    )

    print(
        f"Stacking原始结论: "
        f"{report['model_diagnosis']}"
    )

    print(
        f"Stacking最终概率: "
        f"{report['stacking_probability']:.4f}"
    )

    print(
        "Level-0概率: "
        f"RF={report['rf_probability']:.4f}, "
        f"XGBoost={report['xgb_probability']:.4f}, "
        f"SVM={report['svm_probability']:.4f}"
    )

    print(
        f"风险等级: "
        f"{report['risk_level']}"
    )

    print(
        f"模型判断: "
        f"{report['probability_interpretation']}"
    )

    print(
        f"是否需要复核: "
        f"{report['needs_review']}"
    )

    print(
        f"Review Policy: "
        f"{report['review_policy']['lower_threshold']:.2f}"
        f" ~ "
        f"{report['review_policy']['upper_threshold']:.2f}"
    )

    if report["needs_review"]:
        print(
            f"复核原因: "
            f"{report['review_reason']}"
        )

    print("\n模型适用域:")

    print(
        report[
            "domain"
        ][
            "message"
        ]
    )

    print("\n关键证据:")

    print(
        "-" * 80
    )

    for item in report[
        "top_evidence"
    ]:

        print(
            (
                f"{item['rank']}. "
                f"{item['feature']} "
                f"{item['feature_name']} "
                f"| value="
                f"{item['feature_value']:.6f} "
                f"| SHAP="
                f"{item['shap_value']:+.6f}"
            )
        )

    print("\n机理解释:")

    print(
        report[
            "mechanism_explanation"
        ]
    )

    print("\n维护建议:")

    for index, advice in enumerate(
        report[
            "maintenance_advice"
        ],
        start=1
    ):

        print(
            f"{index}. {advice}"
        )

    print(
        "=" * 80
    )
