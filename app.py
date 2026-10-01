import sys
import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ============================================================
# src 模块路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# ============================================================
# 项目模块
# ============================================================

from data_loader import load_single_sample
from diagnosis_agent import run_diagnosis_agent
from batch_diagnosis import run_batch_diagnosis
from llm_reporter import (
    generate_single_report,
    generate_batch_report,
)


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# 2. 导入项目模块
# ============================================================

from data_loader import (
    load_mat_dataset,
    load_single_sample,
)

from diagnosis_agent import (
    run_diagnosis_agent,
)

from batch_diagnosis import (
    run_batch_diagnosis,
)

from llm_reporter import (
    generate_single_report,
    generate_batch_report,
)

# ============================================================
# 3. 文件路径
# ============================================================

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)


# ============================================================
# 4. 页面设置
# ============================================================

st.set_page_config(
    page_title="轴承润滑诊断 Agent",
    page_icon="⚙️",
    layout="wide",
)


# ============================================================
# 5. 标题
# ============================================================

st.title(
    "轴承润滑诊断 Agent"
)

st.caption(
    "MSM-17D + MSM-ADASYN-Stacking + 完整 Stacking SHAP"
)

st.divider()


# ============================================================
# 6. 数据集信息
# ============================================================

@st.cache_data
def get_dataset_size():

    data = load_mat_dataset(
        MAT_FILE
    )

    return int(
        data.shape[0]
    )


total_samples = get_dataset_size()


# ============================================================
# 7. 侧边栏
# ============================================================

with st.sidebar:

    st.header(
        "诊断设置"
    )

    # ==================================================
    # 诊断模式
    # ==================================================
    diagnosis_mode = st.radio(
        "诊断模式",
        [
            "单样本诊断",
            "批量诊断",
        ],
        index=0,
    )

    st.divider()

    # ==================================================
    # 单样本诊断
    # ==================================================
    if diagnosis_mode == "单样本诊断":

        st.write(
            f"数据集样本数：{total_samples}"
        )

        sample_index = st.number_input(
            label="样本编号",
            min_value=0,
            max_value=total_samples - 1,
            value=0,
            step=1,
        )

        top_n = st.slider(
            label="显示 Top-N SHAP 证据",
            min_value=3,
            max_value=10,
            value=5,
            step=1,
        )

        run_button = st.button(
            label="开始诊断",
            type="primary",
            width="stretch",
        )

    # ==================================================
    # 批量诊断
    # ==================================================
    else:

        st.write(
            f"数据集样本数：{total_samples}"
        )

        batch_start = st.number_input(
            label="起始样本编号",
            min_value=0,
            max_value=total_samples - 1,
            value=0,
            step=1,
        )

        batch_end = st.number_input(
            label="结束样本编号",
            min_value=0,
            max_value=total_samples - 1,
            value=min(19, total_samples - 1),
            step=1,
        )

        batch_button = st.button(
            label="开始批量诊断",
            type="primary",
            width="stretch",
        )


# ============================================================
# 8. 读取当前样本
# ============================================================

if diagnosis_mode == "单样本诊断":

    sample = load_single_sample(
        MAT_FILE,
        sample_index=int(
            sample_index
        )
    )

    # ========================================================
    # 9. 样本信息
    # ========================================================

    st.subheader(
        "样本信息"
    )

    col1, col2, col3, col4 = st.columns(
        4
    )

    with col1:

        st.metric(
            label="设备编号",
            value=str(
                sample[
                    "equipment_id"
                ]
            )
        )

    with col2:

        st.metric(
            label="测点",
            value=str(
                sample[
                    "measurement_point"
                ]
            )
        )

    with col3:

        st.metric(
            label="采样频率",
            value=(
                f"{float(sample['sampling_rate']):.0f} Hz"
            )
        )

    with col4:

        st.metric(
            label="转速",
            value=(
                f"{float(sample['speed']):.0f} rpm"
            )
        )

    st.write(
        f"**设备名称：** "
        f"{sample['equipment_name']}"
    )

    st.write(
        f"**真实标签：** "
        f"{sample['label']}"
    )

    st.write(
        f"**时间：** "
        f"{sample['timestamp']}"
    )


# ============================================================
# 10. 运行 Agent
# ============================================================
if diagnosis_mode == "单样本诊断" and run_button:

    with st.spinner(
        "正在执行 MSM-17D、MSM-ADASYN-Stacking 和完整 Stacking SHAP 诊断..."
    ):

        report = run_diagnosis_agent(
            signal_data=sample["signal"],
            sampling_rate=sample["sampling_rate"],
            top_n=top_n,
        )

    # 保存诊断结果，防止点击 DeepSeek 按钮后结果丢失
    st.session_state["single_report"] = report
    st.session_state["single_sample_index"] = int(sample_index)


# ============================================================
# 从 session_state 恢复诊断结果
# ============================================================

report = None

if (
    diagnosis_mode == "单样本诊断"
    and "single_report" in st.session_state
    and st.session_state.get("single_sample_index") == int(sample_index)
):
    report = st.session_state["single_report"]

    # ========================================================
    # 11. Agent 三态诊断结果
    # ========================================================

    if diagnosis_mode == "单样本诊断" and report is not None:
        st.subheader(
            "Agent 最终决策"
        )

        decision_status = report[
            "decision_status"
        ]

        decision = report[
            "diagnosis"
        ]

        probability = float(
            report.get(
                "stacking_probability",
                report["risk_probability"]
            )
        )

    # --------------------------------------------------------
    # 三态状态框
    # --------------------------------------------------------

    if decision_status == "auto_normal":

        st.success(
            f"🟢 自动诊断：{decision}"
        )

    elif decision_status == "auto_fault":

        st.error(
            f"🔴 自动诊断：{decision}"
        )

    elif decision_status == "review":

        st.warning(
            "🟡 建议人工复核"
        )

    else:

        st.error(
            f"未知决策状态：{decision_status}"
        )

    # --------------------------------------------------------
    # 核心指标
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(
        4
    )

    with c1:

        st.metric(
            "Agent 最终决策",
            decision
        )

    with c2:

        st.metric(
            "Stacking 最终概率",
            f"{probability:.2%}"
        )

    with c3:

        st.metric(
            "Stacking 综合诊断",
            report[
                "model_diagnosis"
            ]
        )

    with c4:

        st.metric(
            "风险等级",
            report[
                "risk_level"
            ]
        )

    st.subheader(
        "Level-0 基学习器概率"
    )

    base_probabilities = report.get(
        "base_probabilities",
        {}
    )

    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric(
            "RF",
            f"{float(base_probabilities.get('rf', report.get('rf_probability', 0))):.2%}"
        )

    with b2:
        st.metric(
            "XGBoost",
            f"{float(base_probabilities.get('xgb', report.get('xgb_probability', 0))):.2%}"
        )

    with b3:
        st.metric(
            "SVM",
            f"{float(base_probabilities.get('svm', report.get('svm_probability', 0))):.2%}"
        )

    # ========================================================
    # Review Policy
    # ========================================================

    st.subheader(
        "Review Policy"
    )

    policy = report[
        "review_policy"
    ]

    p1, p2, p3 = st.columns(3)

    with p1:
        st.metric(
            "正常阈值",
            f"< {policy['lower_threshold']:.2f}"
        )

    with p2:
        st.metric(
            "复核区间",
            (
                f"{policy['lower_threshold']:.2f}"
                f" ~ "
                f"{policy['upper_threshold']:.2f}"
            )
        )

    with p3:
        st.metric(
            "故障阈值",
            f"> {policy['upper_threshold']:.2f}"
        )

    if report["needs_review"]:
        st.warning(
            report["review_reason"]
        )
        st.info(
            "当前样本不会直接作为自动正常或自动故障结果使用，"
            "建议结合频谱、完整Stacking SHAP证据和现场运行信息进行二次复核。"
        )
    else:
        st.success(
            "当前样本位于自动诊断区间，无需触发 Review。"
        )


    # ========================================================
    # 12. 模型适用域
    # ========================================================

    if report[
        "domain"
    ][
        "validated"
    ]:

        st.success(
            report[
                "domain"
            ][
                "message"
            ]
        )

    else:

        st.warning(
            report[
                "domain"
            ][
                "message"
            ]
        )


    # ========================================================
    # 13. 诊断摘要
    # ========================================================

    st.subheader(
        "模型判断"
    )

    st.write(
        report[
            "probability_interpretation"
        ]
    )

    st.write(
        f"**MSM-ADASYN-Stacking 综合诊断：** "
        f"{report['model_diagnosis']}"
    )

    st.write(
        f"**Stacking 最终概率：** {probability:.4f}"
    )

    st.write(
        f"**Agent 三态决策状态：** "
        f"{report['decision_status']}"
    )


    # ========================================================
    # 14. Top SHAP 证据
    # ========================================================

    st.subheader(
        "完整 Stacking SHAP 证据"
    )

    st.caption(
        "以下 SHAP 值解释17维MSM特征对完整Stacking最终概率的贡献。"
    )

    evidence_rows = []

    for item in report[
        "top_evidence"
    ]:

        evidence_rows.append(
            {
                "排名":
                    item[
                        "rank"
                    ],

                "特征":
                    item[
                        "feature"
                    ],

                "名称":
                    item[
                        "feature_name"
                    ],

                "特征值":
                    item[
                        "feature_value"
                    ],

                "SHAP值":
                    item[
                        "shap_value"
                    ],

                "方向":
                    item[
                        "direction"
                    ],
            }
        )

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    st.dataframe(
        evidence_df,
        width="stretch",
        hide_index=True,
    )


    # ========================================================
    # 15. SHAP 条形图
    # ========================================================

    st.subheader(
        "完整 Stacking SHAP 贡献"
    )

    shap_chart_df = (
        evidence_df[
            [
                "名称",
                "SHAP值"
            ]
        ]
        .set_index(
            "名称"
        )
    )

    st.bar_chart(
        shap_chart_df
    )


    # ========================================================
    # 16. 机理解释
    # ========================================================

    st.subheader(
        "机理解释"
    )

    st.text(
        report[
            "mechanism_explanation"
        ]
    )


    # ========================================================
    # 17. ROI / Gaussian
    # ========================================================

    st.subheader(
        "频谱辅助证据"
    )

    roi_col, gaussian_col = st.columns(
        2
    )

    with roi_col:

        st.markdown(
            "### 动态 ROI"
        )

        roi = report[
            "roi"
        ]

        st.write(
            f"ROI 激活："
            f"{roi['active']}"
        )

        if roi[
            "active"
        ]:

            st.write(
                f"主峰频率："
                f"{roi['peak_frequency']:.2f} Hz"
            )

            st.write(
                f"加权质心："
                f"{roi['centroid']:.2f} Hz"
            )

            st.write(
                f"ROI："
                f"{roi['start']:.2f}"
                f" ~ "
                f"{roi['end']:.2f} Hz"
            )

            st.write(
                f"离散度："
                f"{roi['spread']:.2f} Hz"
            )

    with gaussian_col:

        st.markdown(
            "### Gaussian 高频轮廓"
        )

        gaussian = report[
            "gaussian"
        ]

        st.write(
            f"拟合成功："
            f"{gaussian['success']}"
        )

        st.write(
            f"A："
            f"{gaussian['A']:.6f}"
        )

        st.write(
            f"μ："
            f"{gaussian['mu']:.2f} Hz"
        )

        st.write(
            f"σ："
            f"{gaussian['sigma']:.2f} Hz"
        )

        st.write(
            f"RMSE："
            f"{gaussian['rmse']:.6f}"
        )


    # ========================================================
    # 18. MSM-17D
    # ========================================================

    st.subheader(
        "MSM-17D 特征"
    )

    feature_df = pd.DataFrame(
        {
            "特征":
                list(
                    report[
                        "features"
                    ].keys()
                ),

            "数值":
                list(
                    report[
                        "features"
                    ].values()
                ),
        }
    )

    st.dataframe(
        feature_df,
        width="stretch",
        hide_index=True,
    )


    # ========================================================
    # 19. 维护建议
    # ========================================================

    st.subheader(
        "维护建议"
    )

    for i, advice in enumerate(
        report[
            "maintenance_advice"
        ],
        start=1
    ):

        st.write(
            f"{i}. {advice}"
        )


    # ========================================================
    # 20. 真实标签对照
    # ========================================================

    # ========================================================
    # 真实标签对照
    # ========================================================

    st.subheader(
        "结果对照"
    )

    true_label = str(
        sample[
            "label"
        ]
    )

    model_diagnosis = report[
        "model_diagnosis"
    ]

    decision_status = report[
        "decision_status"
    ]

    st.write(
        f"**真实标签：** {true_label}"
    )

    st.write(
        f"**Stacking 综合诊断：** {model_diagnosis}"
    )

    st.write(
        f"**Agent 最终决策：** {report['diagnosis']}"
    )

    # --------------------------------------------------------
    # Review样本
    # --------------------------------------------------------

    if decision_status == "review":

        if model_diagnosis == true_label:

            st.warning(
                "该样本被 Agent 送入复核区间。"
                "Stacking 综合判断与真实标签一致，"
                "但 Agent 为降低高风险误判选择不自动放行。"
            )

        else:

            st.warning(
                "该样本被 Agent 成功送入复核区间，"
                "避免了 Stacking 错误结果被直接自动输出。"
            )

    # --------------------------------------------------------
    # 自动诊断样本
    # --------------------------------------------------------

    else:

        if report[
            "diagnosis"
        ] == true_label:

            st.success(
                "Agent 自动诊断与真实标签一致。"
            )

        else:

            st.error(
                "Agent 自动诊断与真实标签不一致。"
            )


    # ============================================================
    # DeepSeek 智能诊断报告
    # ============================================================
    report_state_key = (
        f"deepseek_single_report_"
        f"{int(sample_index)}"
    )

    generate_ai_report = st.button(
        "生成 DeepSeek 智能诊断报告",
        key=f"deepseek_single_{int(sample_index)}",
        width="stretch",
    )

    if generate_ai_report:
        sample_info = {
            "sample_index":
                int(sample_index),

            "equipment_id":
                sample["equipment_id"],

            "equipment_name":
                sample["equipment_name"],

            "measurement_point":
                sample["measurement_point"],

            "sampling_rate":
                sample["sampling_rate"],

            "speed":
                sample["speed"],

            "timestamp":
                sample["timestamp"],

            "true_label":
                sample["label"],
        }

        with st.spinner(
                "正在调用 DeepSeek 生成智能诊断报告..."
        ):
            llm_result = generate_single_report(
                report=report,
                sample_info=sample_info,
            )

        st.session_state[
            report_state_key
        ] = llm_result

    if report_state_key in st.session_state:

        llm_result = st.session_state[
            report_state_key
        ]

        if llm_result["success"]:

            st.success(
                "DeepSeek 智能诊断报告生成成功"
            )

        else:

            st.warning(
                "DeepSeek 调用失败，已自动使用本地规则报告。"
            )

            if llm_result["error"]:
                with st.expander(
                        "查看调用错误"
                ):
                    st.code(
                        llm_result["error"]
                    )

        st.markdown(
            llm_result["report"]
        )

        st.download_button(
            label="下载智能诊断报告 Markdown",
            data=llm_result["report"],
            file_name=(
                f"deepseek_diagnosis_"
                f"{int(sample_index):04d}.md"
            ),
            mime="text/markdown",
            key=f"download_single_ai_{int(sample_index)}",
            width="stretch",
        )
    # ========================================================
    # 21. 下载 Agent JSON
    # ========================================================

    import json

    json_text = json.dumps(
        report,
        ensure_ascii=False,
        indent=2
    )

    st.download_button(
        label="下载诊断 JSON",
        data=json_text,
        file_name=(
            f"diagnosis_"
            f"{int(sample_index):04d}.json"
        ),
        mime="application/json",
    )
# ============================================================
# 批量诊断模式
# ============================================================

elif diagnosis_mode == "批量诊断":

    st.title(
        "批量轴承润滑诊断"
    )

    st.write(
        f"诊断范围："
        f"{int(batch_start)} ~ {int(batch_end)}"
    )

    if batch_end < batch_start:

        st.error(
            "结束样本编号不能小于起始样本编号。"
        )


if diagnosis_mode == "批量诊断":

    if batch_end < batch_start:

        st.error(
            "结束样本编号不能小于起始样本编号。"
        )

    elif batch_button:

        with st.spinner(
            "正在执行批量诊断，请稍候..."
        ):

            result_df, summary = run_batch_diagnosis(
                start_index=int(batch_start),
                end_index=int(batch_end) + 1,
                top_n=5,
            )

        st.session_state["batch_result_df"] = result_df
        st.session_state["batch_summary"] = summary
        st.session_state["batch_start_index"] = int(batch_start)
        st.session_state["batch_end_index"] = int(batch_end)
    # ============================================================
    # 恢复批量诊断结果
    # ============================================================

    result_df = st.session_state.get(
        "batch_result_df"
    )

    summary = st.session_state.get(
        "batch_summary"
    )

        # ====================================================
        # 结果统计
        # ====================================================

    # ============================================================
    # 显示已经保存的批量诊断结果
    # ============================================================

    if (
            diagnosis_mode == "批量诊断"
            and result_df is not None
            and summary is not None
    ):

        st.success(
            "批量诊断完成"
        )

        distribution = summary[
            "decision_distribution"
        ]

        c1, c2, c3, c4 = st.columns(
            4
        )

        with c1:

            st.metric(
                "🟢 自动正常",
                distribution[
                    "auto_normal"
                ]
            )

        with c2:

            st.metric(
                "🟡 建议复核",
                distribution[
                    "review"
                ]
            )

        with c3:

            st.metric(
                "🔴 自动故障",
                distribution[
                    "auto_fault"
                ]
            )

        with c4:

            st.metric(
                "复核率",
                f"{distribution['review_rate']:.2%}"
            )

        # ====================================================
        # 自动诊断率
        # ====================================================

        st.metric(
            "自动诊断率",
            f"{distribution['auto_rate']:.2%}"
        )

        # ====================================================
        # 高风险 Top-10
        # ====================================================

        st.subheader(
            "高风险样本 Top-10"
        )

        high_risk_df = (
            result_df
            .sort_values(
                "risk_probability",
                ascending=False
            )
            .head(10)
        )

        high_risk_display = high_risk_df[
            [
                "sample_index",
                "equipment_id",
                "equipment_name",
                "measurement_point",
                "risk_probability",
                "agent_decision",
            ]
        ].rename(
            columns={
                "risk_probability": "Stacking最终概率",
                "agent_decision": "Agent最终决策",
            }
        )

        st.dataframe(
            high_risk_display,
            width="stretch",
        )

        # ====================================================
        # Review 样本
        # ====================================================

        st.subheader(
            "建议人工复核样本"
        )

        review_df = result_df[
            result_df[
                "decision_status"
            ] == "review"
            ].copy()

        if len(review_df) > 0:

            review_display = review_df[
                [
                    "sample_index",
                    "equipment_id",
                    "measurement_point",
                    "risk_probability",
                    "agent_decision",
                ]
            ].rename(
                columns={
                    "risk_probability": "Stacking最终概率",
                    "agent_decision": "Agent最终决策",
                }
            )

            st.dataframe(
                review_display,
                width="stretch",
            )

        else:

            st.success(
                "当前批次没有需要人工复核的样本。"
            )

        # ====================================================
        # 自动故障样本
        # ====================================================

        st.subheader(
            "自动判定轴承润滑不良"
        )

        fault_df = result_df[
            result_df[
                "decision_status"
            ] == "auto_fault"
            ].copy()

        if len(fault_df) > 0:

            fault_display = fault_df[
                [
                    "sample_index",
                    "equipment_id",
                    "equipment_name",
                    "measurement_point",
                    "risk_probability",
                ]
            ].rename(
                columns={
                    "risk_probability": "Stacking最终概率",
                }
            )

            st.dataframe(
                fault_display,
                width="stretch",
            )

        else:

            st.info(
                "当前批次没有自动故障样本。"
            )

        # ====================================================
        # CSV 下载
        # ====================================================

        csv_data = result_df.to_csv(
            index=False
        ).encode(
            "utf-8-sig"
        )

        st.download_button(
            label="下载批量诊断 CSV",
            data=csv_data,
            file_name=(
                f"batch_diagnosis_"
                f"{int(batch_start)}_"
                f"{int(batch_end)}.csv"
            ),
            mime="text/csv",
            width="stretch",
        )
        # ============================================================
        # DeepSeek 批量智能报告
        # ============================================================

        st.divider()

        st.subheader(
            "🤖 DeepSeek 批量智能巡检报告"
        )

        st.caption(
            "DeepSeek 根据批量 Diagnosis Agent 的统计结果、"
            "Stacking 高风险样本和完整 Stacking SHAP 特征生成巡检总结，"
            "不会修改本地模型的诊断结果。"
        )

        generate_batch_ai = st.button(
            "生成 DeepSeek 批量智能分析报告",
            key=(
                f"deepseek_batch_"
                f"{int(batch_start)}_"
                f"{int(batch_end)}"
            ),
            width="stretch",
        )

        if generate_batch_ai:
            with st.spinner(
                    "正在调用 DeepSeek 分析批量巡检结果..."
            ):
                llm_batch_result = generate_batch_report(
                    summary
                )

            st.session_state[
                "batch_deepseek_report"
            ] = llm_batch_result
            # ============================================================
            # 显示 DeepSeek 批量报告
            # ============================================================

            if "batch_deepseek_report" in st.session_state:

                llm_batch_result = st.session_state[
                    "batch_deepseek_report"
                ]

                if llm_batch_result.get(
                        "success",
                        False
                ):

                    st.success(
                        "DeepSeek 批量智能巡检报告生成成功"
                    )

                else:

                    st.warning(
                        "DeepSeek 调用失败。"
                    )

                    if llm_batch_result.get(
                            "error"
                    ):
                        with st.expander(
                                "查看调用错误"
                        ):
                            st.code(
                                llm_batch_result[
                                    "error"
                                ]
                            )

                batch_report_text = llm_batch_result.get(
                    "report",
                    ""
                )

                if batch_report_text:
                    st.markdown(
                        batch_report_text
                    )

                    st.download_button(
                        label="下载 DeepSeek 批量巡检报告",
                        data=batch_report_text,
                        file_name=(
                            f"deepseek_batch_report_"
                            f"{int(batch_start):04d}_"
                            f"{int(batch_end):04d}.md"
                        ),
                        mime="text/markdown",
                        key=(
                            f"download_batch_deepseek_"
                            f"{int(batch_start)}_"
                            f"{int(batch_end)}"
                        ),
                        width="stretch",
                    )


if diagnosis_mode == "单样本诊断" and report is None:

    st.info(
        "请在左侧选择样本编号，然后点击“开始诊断”。"
    )
