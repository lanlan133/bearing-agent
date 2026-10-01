from pathlib import Path
import json
import os
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(
    PROJECT_ROOT / ".env"
)
DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)

from openai import OpenAI


# ============================================================
# 1. 基础配置
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "deepseek"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DEEPSEEK_BASE_URL = "https://api.deepseek.com"

# 如果你希望速度优先，可改成 deepseek-v4-flash
DEEPSEEK_MODEL = "deepseek-v4-pro"


# ============================================================
# 2. 创建 DeepSeek Client
# ============================================================

def get_deepseek_client():

    api_key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    if not api_key:

        raise RuntimeError(
            "没有检测到 DEEPSEEK_API_KEY。\n"
            "请先在 PowerShell 中执行：\n"
            '$env:DEEPSEEK_API_KEY="你的API_KEY"'
        )

    client = OpenAI(
        api_key=api_key,
        base_url=DEEPSEEK_BASE_URL,
    )

    return client


# ============================================================
# 3. DeepSeek 通用调用
# ============================================================

def call_deepseek(
        system_prompt,
        user_prompt,
        model=DEEPSEEK_MODEL
):

    client = get_deepseek_client()

    response = (
        client.chat.completions.create(
            model=model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            stream=False,

            # 报告生成不需要很强的推理模式，
            # 先关闭 thinking，速度更快、输出更稳定
            extra_body={
                "thinking": {
                    "type": "disabled"
                }
            },
        )
    )

    content = (
        response
        .choices[0]
        .message
        .content
    )

    if not content:

        raise RuntimeError(
            "DeepSeek 返回了空内容"
        )

    return content.strip()


# ============================================================
# 4. 单样本系统提示词
# ============================================================

SINGLE_SYSTEM_PROMPT = """
你是一个工业旋转机械状态监测报告助手。

你的任务不是重新诊断，也不是修改模型结果。
Python程序已经完成：
1. 振动信号处理
2. MSM-17D特征提取
3. MSM-ADASYN-Stacking综合诊断
4. RF、XGBoost、SVM三个Level-0基学习器概率计算
5. 完整MSM-ADASYN-Stacking最终概率的SHAP解释
6. Review Policy决策

你只能依据输入JSON中的事实生成诊断报告。

必须遵守：

1. 不得修改风险概率。
2. 不得修改Agent最终决策。
3. 不得修改MSM-ADASYN-Stacking原始结论。
4. 不得修改Stacking最终概率或Level-0基学习器概率。
5. 不得修改完整Stacking SHAP数值。
6. 如果Agent决定“建议复核”，不能擅自改成正常或故障。
7. SHAP表示17维MSM特征对完整Stacking最终概率的贡献，
   但不等于物理因果证明。
8. 不要凭空添加传感器测量值、温度、油液检测结果或维修历史。
9. 维护建议应使用“建议检查、建议复核”等工程措辞，
   不得把模型结果描述成绝对物理事实。

输出中文报告。
"""


# ============================================================
# 5. 单样本提示词
# ============================================================

def build_single_prompt(
        report,
        sample_info=None
):

    payload = {
        "sample_info":
            sample_info or {},

        "agent_result":
            {
                "final_decision":
                    report.get(
                        "decision"
                    ),

                "decision_status":
                    report.get(
                        "decision_status"
                    ),

                "needs_review":
                    report.get(
                        "needs_review"
                    ),

                "review_reason":
                    report.get(
                        "review_reason"
                    ),

                "model_diagnosis":
                    report.get(
                        "model_diagnosis"
                    ),

                "model_name":
                    report.get(
                        "model_name",
                        "MSM-ADASYN-Stacking"
                    ),

                "stacking_probability":
                    report.get(
                        "stacking_probability",
                        report.get(
                            "risk_probability"
                        )
                    ),

                "level_0_probabilities":
                    report.get(
                        "base_probabilities",
                        {
                            "rf": report.get(
                                "rf_probability"
                            ),
                            "xgb": report.get(
                                "xgb_probability"
                            ),
                            "svm": report.get(
                                "svm_probability"
                            ),
                        }
                    ),

                "risk_level":
                    report.get(
                        "risk_level"
                    ),

                "review_policy":
                    report.get(
                        "review_policy"
                    ),

                "domain":
                    report.get(
                        "domain"
                    ),

                "explanation_model":
                    "XGBoost base learner",
            },

        "top_evidence":
            report.get(
                "top_evidence",
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

    json_text = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2
    )

    return f"""
下面是轴承振动诊断程序生成的结构化证据：

{json_text}

请严格依据上述数据生成一份工程诊断报告。

请按下面结构输出：

# 轴承润滑状态诊断报告

## 1. 诊断结论
说明Agent最终决策、MSM-ADASYN-Stacking综合诊断、
Stacking最终概率和风险等级。

## 2. 模型判别依据
列出RF、XGBoost、SVM三个Level-0基学习器概率，
并解释Stacking最终概率与Review Policy的关系。
当前冻结的Review Policy复核区间为0.01～0.96；
同时应以输入JSON中的实际阈值为准。
如果进入复核区，必须明确说明系统没有自动放行。

## 3. 完整Stacking关键SHAP证据
重点解释完整Stacking最终概率的Top SHAP特征。
必须说明正SHAP推动润滑不良，负SHAP推动非润滑不良。
应明确这些SHAP值对应完整Stacking最终概率。

## 4. 频谱与MSM机理解释
结合ROI、Gaussian和MSM特征含义进行解释。
只能描述“与某种现象一致”“支持某种判断”，
不能把相关性描述为已证明的物理因果。

## 5. 风险与不确定性
说明适用域和模型不确定性。

## 6. 维护建议
根据最终决策给出分层建议：
自动正常：继续趋势监测；
建议复核：优先人工复核和多源信息确认；
自动故障：建议尽快检查润滑状态及相关系统。

报告语言简洁、专业、适合工程人员阅读。
"""


# ============================================================
# 6. 单样本 DeepSeek 报告
# ============================================================

def generate_single_report(
        report,
        sample_info=None
):
    """
    DeepSeek 单样本智能报告。

    API失败时自动退回本地规则报告。
    """

    try:

        prompt = build_single_prompt(
            report,
            sample_info
        )

        text = call_deepseek(
            SINGLE_SYSTEM_PROMPT,
            prompt
        )

        return {
            "success":
                True,

            "source":
                "deepseek",

            "model":
                DEEPSEEK_MODEL,

            "report":
                text,

            "error":
                None,
        }

    except Exception as exc:

        # -----------------------------------------------
        # fallback
        # -----------------------------------------------

        local_text = build_local_single_report(
            report,
            sample_info
        )

        return {
            "success":
                False,

            "source":
                "local_fallback",

            "model":
                None,

            "report":
                local_text,

            "error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }


# ============================================================
# 7. 本地单样本备用报告
# ============================================================

def build_local_single_report(
        report,
        sample_info=None
):

    sample_info = (
        sample_info
        or {}
    )

    lines = []

    lines.append(
        "# 轴承润滑状态诊断报告"
    )

    lines.append(
        ""
    )

    lines.append(
        "## 1. 诊断结论"
    )

    lines.append(
        (
            f"Agent最终决策："
            f"{report.get('decision')}。"
        )
    )

    lines.append(
        (
            f"MSM-ADASYN-Stacking综合诊断："
            f"{report.get('model_diagnosis')}。"
        )
    )

    stacking_probability = report.get(
        "stacking_probability",
        report.get("risk_probability", 0)
    )

    lines.append(
        (
            f"Stacking最终概率："
            f"{float(stacking_probability):.2%}。"
        )
    )

    base_probabilities = report.get(
        "base_probabilities",
        {}
    )

    rf_probability = base_probabilities.get(
        "rf",
        report.get("rf_probability", 0)
    )
    xgb_probability = base_probabilities.get(
        "xgb",
        report.get("xgb_probability", 0)
    )
    svm_probability = base_probabilities.get(
        "svm",
        report.get("svm_probability", 0)
    )

    lines.append(
        (
            "Level-0基学习器概率："
            f"RF={float(rf_probability):.2%}，"
            f"XGBoost={float(xgb_probability):.2%}，"
            f"SVM={float(svm_probability):.2%}。"
        )
    )

    lines.append(
        (
            f"风险等级："
            f"{report.get('risk_level')}。"
        )
    )

    lines.append(
        ""
    )

    lines.append(
        "## 2. Review Policy"
    )

    review_policy = report.get(
        "review_policy",
        {}
    )
    lower_threshold = float(
        review_policy.get("lower_threshold", 0.01)
    )
    upper_threshold = float(
        review_policy.get("upper_threshold", 0.96)
    )

    lines.append(
        (
            "当前冻结复核区间："
            f"{lower_threshold:.2f}～{upper_threshold:.2f}。"
        )
    )

    if report.get(
        "needs_review"
    ):

        lines.append(
            (
                "该样本已进入人工复核区间。"
                f"{report.get('review_reason', '')}"
            )
        )

    else:

        lines.append(
            "当前样本位于自动诊断区间。"
        )

    lines.append(
        ""
    )

    lines.append(
        "## 3. 完整Stacking SHAP证据"
    )

    lines.append(
        "以下SHAP值解释17维MSM特征对完整Stacking最终概率的贡献。"
    )

    for item in report.get(
        "top_evidence",
        []
    )[:5]:

        lines.append(
            (
                f"- {item.get('feature')} "
                f"{item.get('feature_name')}："
                f"SHAP="
                f"{float(item.get('shap_value', 0)):+.4f}；"
                f"{item.get('mechanism', '')}"
            )
        )

    lines.append(
        ""
    )

    lines.append(
        "## 4. 维护建议"
    )

    for advice in report.get(
        "maintenance_advice",
        []
    ):

        lines.append(
            f"- {advice}"
        )

    return "\n".join(
        lines
    )


# ============================================================
# 8. 批量报告系统提示词
# ============================================================

BATCH_SYSTEM_PROMPT = """
你是工业旋转机械批量状态监测报告助手。

输入数据来自已经完成验证的本地诊断程序。
你只负责总结和组织报告，不重新计算，不修改任何数值。

必须遵守：

1. 不得修改自动正常、建议复核、自动故障数量。
2. 不得修改Stacking最终概率或RF、XGBoost、SVM的Level-0概率。
3. 不得虚构设备异常。
4. 不得把SHAP解释成物理因果证明。
5. 对高风险设备应表述为“建议优先检查”。
6. 对Review设备应强调需要人工或多源信息复核。
7. 输出应适合设备维护和状态监测汇报。
"""


# ============================================================
# 9. 批量 Prompt
# ============================================================

def build_batch_prompt(
        summary
):

    json_text = json.dumps(
        summary,
        ensure_ascii=False,
        indent=2
    )

    return f"""
以下是批量轴承润滑巡检程序生成的统计摘要：

{json_text}

请生成一份中文《轴承润滑状态批量巡检报告》。

要求包含：

# 轴承润滑状态批量巡检报告

## 1. 巡检概况
说明本次处理样本数、程序失败数。

## 2. 三态诊断分布
总结自动正常、建议复核、自动故障数量，
以及自动诊断率、复核率。
说明三态决策使用Stacking最终概率和冻结的Review Policy，
当前复核区间为0.01～0.96。

## 3. 重点风险设备
根据 high_risk_top10 和 equipment_top10
指出建议优先检查的设备。
不得自行增加设备。

## 4. 主要模型证据
根据 top_global_features
总结本批数据中常见的完整Stacking SHAP特征。

## 5. 人工复核建议
重点说明Review样本需要进一步检查的原因。

## 6. 维护优先级建议
形成：
高优先级
中优先级
常规监测
三层建议。

## 7. 使用限制
明确说明：
MSM-ADASYN-Stacking、完整Stacking SHAP和Agent用于状态监测辅助，
不能代替现场检查和专业维护判断。
"""


# ============================================================
# 10. 批量 DeepSeek 报告
# ============================================================

def generate_batch_report(
        summary
):

    try:

        prompt = build_batch_prompt(
            summary
        )

        text = call_deepseek(
            BATCH_SYSTEM_PROMPT,
            prompt
        )

        return {
            "success":
                True,

            "source":
                "deepseek",

            "model":
                DEEPSEEK_MODEL,

            "report":
                text,

            "error":
                None,
        }

    except Exception as exc:

        return {
            "success":
                False,

            "source":
                "local_fallback",

            "model":
                None,

            "report":
                build_local_batch_report(
                    summary
                ),

            "error":
                (
                    f"{type(exc).__name__}: "
                    f"{exc}"
                ),
        }


# ============================================================
# 11. 本地批量备用报告
# ============================================================

def build_local_batch_report(
        summary
):

    scope = summary.get(
        "data_scope",
        {}
    )

    distribution = summary.get(
        "decision_distribution",
        {}
    )

    lines = [
        "# 轴承润滑状态批量巡检报告",
        "",
        "## 1. 巡检概况",
        (
            f"成功处理 "
            f"{scope.get('total_processed', 0)} "
            f"个样本。"
        ),
        (
            f"程序失败 "
            f"{scope.get('processing_errors', 0)} "
            f"个样本。"
        ),
        "",
        "## 2. 三态诊断分布",
        (
            f"- 自动正常："
            f"{distribution.get('auto_normal', 0)}"
        ),
        (
            f"- 建议复核："
            f"{distribution.get('review', 0)}"
        ),
        (
            f"- 自动故障："
            f"{distribution.get('auto_fault', 0)}"
        ),
        (
            f"- 自动诊断率："
            f"{float(distribution.get('auto_rate', 0)):.2%}"
        ),
        (
            f"- 复核率："
            f"{float(distribution.get('review_rate', 0)):.2%}"
        ),
        "",
        "## 3. 高风险样本",
    ]

    for item in summary.get(
        "high_risk_top10",
        []
    ):

        stacking_probability = item.get(
            "stacking_probability",
            item.get("risk_probability", 0)
        )

        lines.append(
            (
                f"- sample "
                f"{item.get('sample_index')}，"
                f"设备 "
                f"{item.get('equipment_id')}，"
                f"Stacking最终概率 "
                f"{float(stacking_probability):.2%}，"
                f"Agent决策："
                f"{item.get('agent_decision')}"
            )
        )

    lines.extend(
        [
            "",
            "## 4. 建议",
            (
                "建议优先复核高风险及Review样本，"
                "并结合现场工况、温度、噪声和维护记录进行确认。"
            ),
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# 12. 保存报告
# ============================================================

def save_text_report(
        text,
        output_path
):

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path.write_text(
        text,
        encoding="utf-8"
    )

    return output_path
def generate_batch_report(batch_result):
    """
    使用 DeepSeek 对批量诊断结果生成总结报告。
    """

    import os
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有检测到 DEEPSEEK_API_KEY。"
        )

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    # --------------------------------------------------
    # 兼容 DataFrame / dict / list
    # --------------------------------------------------

    if hasattr(batch_result, "to_dict"):
        records = batch_result.to_dict(
            orient="records"
        )
    elif isinstance(batch_result, dict):
        records = batch_result
    else:
        records = list(batch_result)

    # --------------------------------------------------
    # 防止一次发送过多样本给大模型
    # --------------------------------------------------

    if isinstance(records, list):
        records_for_llm = records[:100]
    else:
        records_for_llm = records

    prompt = f"""
你是一名旋转机械状态监测与轴承故障诊断专家。

下面是轴承润滑诊断 Agent 的批量诊断结果。

诊断系统采用：

MSM-17D 特征
→ MSM-ADASYN数据平衡
→ RF、XGBoost、SVM Level-0基学习器
→ Stacking综合诊断与最终概率
→ 完整Stacking最终概率SHAP解释
→ Review Policy（复核区间0.01～0.96）
→ Agent 三态决策

三态决策包括：

1. 非轴承润滑不良（自动正常）
2. 建议复核
3. 轴承润滑不良（自动故障）

请根据下面的批量诊断数据生成专业的设备状态诊断报告。

批量诊断数据：

{records_for_llm}

请严格按照以下结构输出 Markdown：

# 轴承润滑批量智能诊断报告

## 1. 批量诊断概况

说明：
- 本次分析样本数量
- 自动正常数量
- 建议复核数量
- 自动故障数量
- 整体风险情况

## 2. 高风险设备分析

识别风险概率较高的样本。

说明：
- 设备编号
- 测点
- Stacking最终概率
- RF、XGBoost、SVM Level-0概率（数据中存在时）
- Agent 决策
- 风险原因

不要虚构数据中不存在的设备。

## 3. 建议人工复核样本

重点分析 Agent 判定为“建议复核”的样本。

说明为什么这些样本需要人工确认。

## 4. 设备风险排序

按照风险程度给出：

高优先级
中优先级
低优先级

的设备或样本。

## 5. 维护建议

结合批量诊断结果给出维护建议，包括：

- 润滑状态检查
- 润滑油/脂检查
- 振动趋势复测
- 温度检查
- 异常设备持续监测

## 6. 总体结论

总结本批次设备运行状态。

重要要求：

1. 不得修改MSM-ADASYN-Stacking诊断结果或任何概率。
2. SHAP解释17维MSM特征对完整Stacking最终概率的贡献。
3. 不得虚构不存在的数据。
4. DeepSeek 只负责解释和总结。
5. Agent 的诊断结果是最终机器决策依据。
6. 对“建议复核”样本必须明确说明需要人工复核。
"""

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "你是一名工业旋转机械状态监测、"
                    "轴承润滑分析和预测性维护专家。"
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=0.2,
    )

    report_text = response.choices[0].message.content

    return {
        "success": True,
        "report": report_text,
    }
def answer_diagnosis_question(
    question: str,
    report: dict,
) -> dict:
    """
    基于当前诊断结果回答用户追问。
    """

    import os
    import json

    from openai import OpenAI


    # ========================================================
    # 1. API KEY
    # ========================================================

    api_key = os.getenv(
        "DEEPSEEK_API_KEY"
    )

    if not api_key:

        try:
            import streamlit as st

            api_key = st.secrets[
                "DEEPSEEK_API_KEY"
            ]

        except Exception:
            api_key = None


    if not api_key:

        return {
            "success": False,
            "answer": (
                "未检测到 DEEPSEEK_API_KEY，"
                "暂时无法调用 DeepSeek。"
            ),
        }


    # ========================================================
    # 2. DeepSeek Client
    # ========================================================

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )


    # ========================================================
    # 3. 从诊断报告提取关键信息
    # ========================================================

    diagnosis = report.get(
        "diagnosis",
        "未知",
    )

    stacking_probability = report.get(
        "stacking_probability",
        report.get(
            "risk_probability",
            None,
        ),
    )

    rf_probability = report.get(
        "rf_probability",
        None,
    )

    xgb_probability = report.get(
        "xgb_probability",
        None,
    )

    svm_probability = report.get(
        "svm_probability",
        None,
    )

    risk_level = report.get(
        "risk_level",
        "未知",
    )

    summary = report.get(
        "summary",
        "",
    )

    evidence = report.get(
        "top_evidence",
        [],
    )


    # ========================================================
    # 4. 构建上下文
    # ========================================================

    diagnosis_context = {
        "final_model": (
            "MSM-ADASYN-Stacking"
        ),

        "diagnosis":
            diagnosis,

        "stacking_probability":
            stacking_probability,

        "level0_probabilities": {
            "RF":
                rf_probability,

            "XGBoost":
                xgb_probability,

            "SVM":
                svm_probability,
        },

        "risk_level":
            risk_level,

        "summary":
            summary,

        "shap_evidence":
            evidence,
    }


    # ========================================================
    # 5. System Prompt
    # ========================================================

    system_prompt = """
你是一名旋转机械轴承润滑状态诊断助手。

当前系统采用 MSM-ADASYN-Stacking 进行最终诊断：

MSM-17D
→ Random Forest
→ XGBoost
→ SVM
→ Logistic Regression Meta Learner
→ 最终 Stacking 概率

注意：

1. 最终故障概率来自 Stacking，不是单独的 XGBoost。
2. RF、XGBoost、SVM 的概率属于 Level-0 基学习器结果。
3. SHAP证据解释17维MSM特征对完整Stacking最终概率的贡献。
4. 回答必须基于提供的诊断数据。
5. 不要编造没有出现在诊断上下文中的数据。
6. 如果用户询问判断原因，应结合：
   - Stacking 最终概率
   - 三个基学习器概率
   - SHAP 特征证据
   进行解释。
7. 如果样本位于 Review Policy 复核区，
   应明确说明需要人工复核。
8. 使用中文回答。
9. 优先使用清晰、工程化、容易理解的表达。
"""


    # ========================================================
    # 6. User Prompt
    # ========================================================

    user_prompt = f"""
以下是当前样本的诊断结果：

{json.dumps(
    diagnosis_context,
    ensure_ascii=False,
    indent=2,
)}

用户的问题：

{question}

请只围绕当前样本回答。
"""


    # ========================================================
    # 7. 调用 DeepSeek
    # ========================================================

    try:

        response = client.chat.completions.create(

            model="deepseek-chat",

            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],

            temperature=0.2,
        )


        answer = (
            response
            .choices[0]
            .message
            .content
        )


        return {
            "success": True,
            "answer": answer,
        }


    except Exception as exc:

        return {
            "success": False,
            "answer": (
                "DeepSeek 调用失败："
                f"{exc}"
            ),
        }
