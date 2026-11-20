"""Five-region local workbench for the three-branch bearing diagnosis Agent."""

import json
from pathlib import Path
import sys

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from diagnosis_evidence import build_key_evidence, build_model_details, build_probability_summary
from signal_charts import signal_frames, waveform_chart, spectrum_chart, roi_chart
from workbench_state import WorkbenchState, explicit_sample_number


st.set_page_config(
    page_title="轴承润滑诊断工作台", page_icon=":material/query_stats:", layout="wide",
)
# Keep the title clear of the fixed Streamlit toolbar.
st.html("""
<style>
    [data-testid="stMain"] [data-testid="stMainBlockContainer"] { padding-top: 3.375rem; }
</style>
""")

if "workbench" not in st.session_state:
    st.session_state.workbench = WorkbenchState()
workbench = st.session_state.workbench
if "_sample_number_sync" in st.session_state:
    st.session_state.sample_number = st.session_state.pop("_sample_number_sync")
st.session_state.setdefault("sample_number", workbench.sample_number)


def on_upload():
    uploaded = st.session_state.workbench_upload
    if uploaded is None:
        return
    try:
        changed = workbench.activate_upload(
            uploaded.name, uploaded.getvalue(), PROJECT_ROOT / "data" / "uploaded"
        )
        if changed:
            st.session_state.sample_number = 1
    except Exception as exc:
        workbench.error = f"文件读取失败：{exc}"


def on_sample_change():
    try:
        workbench.select_sample(st.session_state.sample_number)
    except (TypeError, ValueError) as exc:
        workbench.error = str(exc)


def move_sample(offset):
    workbench.select_sample(workbench.sample_number + offset)
    st.session_state.sample_number = workbench.sample_number


def clear_conversation():
    workbench.messages.clear()


def maintenance_actions(report):
    # The card already shows the diagnosis, so keep only the follow-up actions.
    actions = [
        item for item in (report.get("maintenance_advice") or [])
        if item != "当前模型未显示强烈的轴承润滑不良风险。"
    ]
    return "\n".join(f"- {item}" for item in actions) or "当前报告未提供后续建议。"


def report_markdown():
    sample = workbench.sample or {}
    report = workbench.report
    if report is None:
        return ""
    return (
        f"# 第 {workbench.sample_number} 条数据诊断报告\n\n"
        f"文件：{workbench.file_name}\n\n"
        f"设备：{sample.get('equipment_id', '-')}，测点：{sample.get('measurement_point', '-')}\n\n"
        f"最终诊断：{report['diagnosis']}\n\n"
        f"{build_probability_summary(report)}\n\n"
        f"## 关键依据\n\n{build_key_evidence(report)}\n\n"
        f"## 后续建议\n\n{maintenance_actions(report)}\n\n"
        f"## 模型详情\n\n{build_model_details(report)}"
    )


def render_report():
    if workbench.error:
        st.error(workbench.error)
    if workbench.dataset is None:
        st.info("上传 .mat 振动数据并选择样本，即可开始诊断。", icon=":material/info:")
        return
    sample = workbench.sample or {}
    st.caption(
        f"第 {workbench.sample_number} / {workbench.total_samples} 条 · "
        f"设备 {sample.get('equipment_id', '-')} · 测点 {sample.get('measurement_point', '-')}"
    )
    report = workbench.report
    if report is None:
        st.info("当前样本已就绪。点击「开始诊断」查看结论和关键依据。", icon=":material/info:")
        return
    if report["diagnosis"] == "轴承润滑不良":
        st.warning(f"诊断结果：{report['diagnosis']}", icon=":material/priority_high:")
    else:
        st.success(f"诊断结果：{report['diagnosis']}", icon=":material/check_circle:")
    st.markdown(build_probability_summary(report))
    domain = report.get("domain") or {}
    if domain.get("validated") is False and domain.get("message"):
        st.caption(domain["message"])
    st.markdown("#### 关键依据")
    st.markdown(build_key_evidence(report))
    st.markdown("#### 后续建议")
    st.markdown(maintenance_actions(report))
    with st.expander("展开查看模型详情", expanded=False):
        st.markdown(build_model_details(report))


with st.container(gap="xsmall"):
    st.title("轴承润滑诊断工作台", anchor=False)
    st.caption("振动信号分析 · 三路 Judge 融合诊断")
left, right = st.columns([2, 1], gap="medium", vertical_alignment="top")
diagnose_requested = False

with left:
    # Region 4: all high-frequency actions belong to the current file/sample.
    with st.container(border=True, key="operation_panel", gap="small"):
        st.markdown("#### 数据与样本")
        with st.container(horizontal=True, vertical_alignment="bottom", gap="small"):
            with st.popover("上传 MAT", icon=":material/upload_file:"):
                st.file_uploader(
                    "上传 .mat 数据文件", type=["mat"], key="workbench_upload",
                    on_change=on_upload,
                )
            st.number_input(
                "样本编号", min_value=1, max_value=max(1, workbench.total_samples),
                step=1, key="sample_number", width=120,
                disabled=workbench.dataset is None, on_change=on_sample_change,
            )
            st.button(
                "上一条", key="previous_sample", on_click=move_sample, args=(-1,),
                disabled=workbench.dataset is None or workbench.sample_number <= 1,
            )
            st.button(
                "下一条", key="next_sample", on_click=move_sample, args=(1,),
                disabled=workbench.dataset is None or workbench.sample_number >= workbench.total_samples,
            )
            diagnose_requested = st.button(
                "开始诊断", type="primary", icon=":material/play_arrow:",
                key="diagnose_current", disabled=workbench.dataset is None,
            )
            with st.popover("导出", icon=":material/download:", disabled=workbench.report is None):
                st.download_button(
                    "下载 Markdown 报告", data=report_markdown(), mime="text/markdown",
                    file_name=f"diagnosis_{workbench.sample_number}.md", on_click="ignore",
                )
                payload = {
                    "file_name": workbench.file_name,
                    "sample_number": workbench.sample_number,
                    "report": workbench.report,
                }
                st.download_button(
                    "下载 JSON 结果", data=json.dumps(payload, ensure_ascii=False, indent=2),
                    mime="application/json", file_name=f"diagnosis_{workbench.sample_number}.json",
                    on_click="ignore",
                )
        st.caption(
            f"当前文件：{workbench.file_name} · 共 {workbench.total_samples} 条 · 当前第 {workbench.sample_number} 条"
            if workbench.dataset is not None else "未上传数据 · 支持 MATLAB .mat 振动文件"
        )

    # Region 3: keep the waveform and full-band spectrum in the same panel.
    with st.container(border=True, height=550, key="signal_panel", gap="small"):
        signal_slot = st.empty()

    # Region 5: the automatically identified ROI and its spectral envelopes.
    with st.container(border=True, height=420, key="roi_panel", gap="small"):
        roi_slot = st.empty()

with right, st.container(height="stretch", key="analysis_column", gap="small"):
    # Region 1: a scrollable current-sample report, separate from conversation.
    with st.container(border=True, height=730, key="diagnosis_panel", gap="small"):
        st.markdown("#### 诊断结果与分析")
        report_slot = st.empty()

    # Region 2: inline chat input stays below its own history inside this column.
    with st.container(
        border=True, height="stretch", key="conversation_panel", gap="small",
        vertical_alignment="distribute",
    ):
        with st.container(horizontal=True, horizontal_alignment="distribute"):
            st.markdown("#### DeepSeek 对话")
            st.button("清空", on_click=clear_conversation, key="clear_conversation", type="tertiary")
        st.caption(
            f"围绕当前第 {workbench.sample_number} 条提问，历史消息保留样本标记。"
            if workbench.dataset is not None else "上传数据后，可询问样本的诊断依据或请求批量诊断。"
        )
        with st.container(height=160, border=False, key="conversation_history", autoscroll=True):
            if not workbench.messages:
                st.caption("例如：哪些特征影响最大？三路模型的判断是否一致？")
            for message in workbench.messages:
                avatar = ":material/analytics:" if message["role"] == "assistant" else None
                with st.chat_message(message["role"], avatar=avatar):
                    st.caption(f"{message.get('file_name', '')} · 第 {message.get('sample_number', 1)} 条")
                    st.markdown(message["content"])
        prompt = st.chat_input(
            "输入问题或样本编号…", key="workbench_question",
            disabled=workbench.dataset is None, submit_mode="disable", max_chars=3000,
        )


# Render stable panels first, then compute the selected signal just once per switch.
frames = None
roi_frame = None
analysis = None
if workbench.dataset is not None:
    try:
        with st.spinner("正在准备当前样本图表…"):
            analysis = workbench.ensure_signal()
            frames = signal_frames(workbench.sample, analysis, envelope_view="full")
            roi_frame = signal_frames(workbench.sample, analysis, envelope_view="roi")[2]
    except Exception as exc:
        workbench.error = f"当前样本绘图失败：{exc}"

# Use the same chart layout before and after upload; empty data leaves axes visible.
roi = analysis["roi"] if frames is not None else {}
with signal_slot.container(gap="xxsmall"):
    with st.container(horizontal=True, horizontal_alignment="distribute", vertical_alignment="center", gap="xsmall"):
        st.markdown("#### 振动信号")
        if frames is not None:
            sample = workbench.sample
            st.caption(
                f"采样率 {sample['sampling_rate']:g} Hz · 转速 {sample['speed']:g} rpm · "
                f"{len(sample['signal']):,} 点"
            )
        else:
            st.caption("采样率 — Hz · 转速 — rpm · 点数 —")
    st.markdown("**时域波形**")
    st.altair_chart(
        waveform_chart(frames[0] if frames is not None else None),
        width="stretch", height=200, key="waveform_plot",
    )
    st.markdown("**全频段频谱**")
    st.altair_chart(
        spectrum_chart(frames[1] if frames is not None else None, roi),
        width="stretch", height=210, key="spectrum_plot",
    )

with roi_slot.container(gap="xxsmall"):
    st.markdown("#### 频谱包络与 ROI")
    if frames is None:
        st.caption("关注频带与主峰待识别")
    elif roi.get("active"):
        st.caption(
            f"关注频带 {roi['start'] / 1000:.2f}–{roi['end'] / 1000:.2f} kHz · "
            f"主峰 {roi['peak_frequency'] / 1000:.2f} kHz"
        )
    else:
        st.caption("当前样本未激活 ROI，左右两图均显示完整频段。")
    full_envelope_column, expanded_roi_column = st.columns(
        2, gap="small", vertical_alignment="top", wrap=False,
    )
    with full_envelope_column:
        st.markdown("**完整包络**")
        st.altair_chart(
            roi_chart(frames[2] if frames is not None else None, roi=roi),
            width="stretch", height=280, key="full_envelope_plot",
        )
    with expanded_roi_column:
        st.markdown("**ROI 扩展**")
        st.altair_chart(
            roi_chart(roi_frame, roi=roi),
            width="stretch", height=280, key="expanded_roi_plot",
        )

with report_slot.container():
    render_report()


# Buttons request the deterministic tool; natural-language tasks use the LLM loop.
if diagnose_requested:
    try:
        with report_slot.container():
            with st.spinner(f"正在诊断第 {workbench.sample_number} 条并生成 SHAP 解释…"):
                workbench.diagnose()
    except Exception as exc:
        workbench.error = f"诊断失败：{exc}"
    st.rerun()

if prompt:
    question = str(prompt).strip()
    number = explicit_sample_number(question)
    try:
        with report_slot.container():
            if number is not None:
                workbench.select_sample(number)
                workbench.add_message("user", question)
                with st.spinner(f"正在诊断第 {number} 条…"):
                    workbench.diagnose()
                workbench.add_message("assistant", f"第 {number} 条诊断完成，结果已更新至右上方诊断区。")
            else:
                with st.spinner("DeepSeek 正在分析问题并调用所需工具…"):
                    workbench.ask(question)
    except Exception as exc:
        workbench.add_message("assistant", f"本次请求未完成：{exc}")
    st.session_state._sample_number_sync = workbench.sample_number
    st.rerun()
