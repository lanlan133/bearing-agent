"""Data-grounded descriptions for the diagnosis UI, independent of Streamlit.

Time-domain statistics describe the uploaded signal only. They are not additional
inputs to the trained 17D models. SHAP in these reports explains p_full only.
"""

from __future__ import annotations

import math

import numpy as np

from config import HIGH_FREQ_RATIO


def calculate_signal_statistics(signal_data):
    """Raw-signal RMS/peak factor and centered population Pearson kurtosis.

    Scaling before taking powers avoids overflow for large finite amplitudes.
    Kurtosis is undefined for a constant signal; peak factor is undefined at zero
    RMS. Return None for those cases so the report never invents a value.
    """
    values = np.asarray(signal_data, dtype=float).ravel()
    if not values.size or not np.isfinite(values).all():
        raise ValueError("时域统计要求非空且全部有限的一维信号。")
    peak = float(np.max(np.abs(values)))
    scaled = values / peak if peak else np.zeros_like(values)
    rms_scaled = float(np.sqrt(np.mean(scaled ** 2)))
    centered = scaled - np.mean(scaled)
    variance = float(np.mean(centered ** 2))
    kurtosis = (
        float(np.mean(centered ** 4) / variance ** 2)
        if variance > 0 else None
    )
    return {
        "sample_count": int(values.size),
        "mean": float(np.mean(scaled) * peak),
        "rms": float(rms_scaled * peak),
        "peak_abs": peak,
        "crest_factor": float(1.0 / rms_scaled) if rms_scaled else None,
        "kurtosis_pearson": kurtosis,
        "source": "raw_signal_before_detrending",
    }


def _number(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _format(value):
    number = _number(value)
    if number is None:
        return "未提供"
    if 0 < abs(number) < 0.0001 or abs(number) >= 10000:
        return f"{number:.4g}"
    return f"{number:.4f}"


def _probability(report, *keys):
    for key in keys:
        value = _number(report.get(key))
        if value is not None and 0 <= value <= 1:
            return value
    return None


def _signal_description(report):
    stats = report.get("signal_statistics") or {}
    rms, peak = _number(stats.get("rms")), _number(stats.get("peak_abs"))
    if rms is None or peak is None:
        return "当前报告没有原始时域统计，暂不能描述波形幅值与冲击性；可重新诊断以补充这些指标。"
    text = (
        f"原始振动信号的 RMS 为 {_format(rms)}，绝对峰值为 {_format(peak)}"
        "（沿用上传信号的幅值单位）。"
    )
    crest = _number(stats.get("crest_factor"))
    if crest is not None:
        text += f"峰值因子为 {crest:.2f}，即最大绝对幅值约为 RMS 的 {crest:.2f} 倍。"
    else:
        text += "当前 RMS 为零，峰值因子无定义。"
    kurtosis = _number(stats.get("kurtosis_pearson"))
    if kurtosis is None:
        text += "信号为常量，时域峭度无定义。"
    elif kurtosis > 3:
        text += (
            f"时域 Pearson 峭度为 {kurtosis:.2f}，高于高斯分布参考值 3，"
            "提示该段信号幅值分布的尾部较重、可能存在突出峰值。"
        )
    else:
        text += (
            f"时域 Pearson 峭度为 {kurtosis:.2f}，未超过高斯分布参考值 3；"
            "该统计量本身不能排除局部冲击。"
        )
    return text + "幅值是否异常仍需同设备、同工况的历史基线作比较。"


def _spectrum_description(report):
    features = report.get("features") or {}
    parts = []
    ratio = _number(features.get("f2"))
    fs = _number(report.get("sampling_rate"))
    if ratio is not None:
        band = (
            f" {HIGH_FREQ_RATIO * fs / 2 / 1000:.2f}–{fs / 2 / 1000:.2f} kHz"
            if fs is not None and fs > 0 else ""
        )
        parts.append(
            f"高频能量占比为 {ratio:.2%}，表示所定义的{band}高频范围内"
            "包含相应比例的频谱能量。"
        )
    roi = report.get("roi") or {}
    start, end = _number(roi.get("start")), _number(roi.get("end"))
    if roi.get("active") and start is not None and end is not None:
        text = (
            f"动态 ROI 识别出的局部关注频带为 {start / 1000:.2f}–{end / 1000:.2f} kHz"
        )
        peak, centroid = _number(roi.get("peak_frequency")), _number(roi.get("centroid"))
        if peak is not None:
            text += f"，主峰位于 {peak / 1000:.2f} kHz"
        if centroid is not None:
            text += f"，能量加权质心位于 {centroid / 1000:.2f} kHz"
        parts.append(text + "。该频带描述局部高频结构，不等于全部信号能量的分布范围。")
    elif "active" in roi:
        parts.append("本次未激活动态 ROI，尚未识别出符合当前算法条件的局部高频关注带。")
    flatness = _number(features.get("f7"))
    if flatness is not None:
        parts.append(
            f"高频谱平坦度为 {flatness:.4f}；该指标越接近 0 越偏向尖峰/峰谷结构，"
            "越接近 1 越接近均匀平坦谱。"
        )
    gaussian = report.get("gaussian") or {}
    rmse = _number(gaussian.get("rmse"))
    if gaussian.get("success") and rmse is not None:
        parts.append(
            f"高斯轮廓拟合 RMSE 为 {_format(rmse)}，用于描述峰值归一化频谱"
            "与拟合曲线的形状差异。"
        )
    elif "success" in gaussian:
        parts.append("本次高斯轮廓未拟合成功，不据此推断频谱形态。")
    return "".join(parts) or "当前报告缺少频谱与 ROI 信息。"


def _feature_evidence(items):
    text = "、".join(
        f"{item.get('feature', '')} {item.get('feature_name', '特征')}"
        f"（值 {_format(item.get('feature_value'))}）"
        for item in items[:3]
    )
    return text + ("等特征" if len(items) > 3 else "")


def _feature_context(evidence):
    names = {item.get("feature") for item in evidence}
    descriptions = []
    if names & {"f10", "f11", "f12", "f13"}:
        descriptions.append("下谱包络指标对应持续能量的谱底背景")
    if "f14" in names:
        descriptions.append("上包络 RMS 对应局部峰值轮廓强度")
    if names & {"f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9"}:
        descriptions.append("频谱/ROI 指标对应频带能量分布、峰谷和宽带结构")
    if names & {"f15", "f16", "f17"}:
        descriptions.append("高斯轮廓指标对应高频峰形及拟合差异")
    return "其中，" + "，".join(descriptions) + "。" if descriptions else ""


def _model_description(report):
    evidence = [
        item for item in (report.get("top_evidence") or [])
        if _number(item.get("shap_value")) is not None
    ]
    full = _probability(report, "full_stacking_probability", "stacking_probability")
    text = ""
    if full is not None:
        label = "轴承润滑不良" if full >= 0.5 else "非轴承润滑不良"
        text += f"完整 17D Stacking 分支概率为 {full:.2%}，按 50% 阈值倾向于“{label}”。"
    if not evidence:
        return text + "当前未返回有效 SHAP 证据，无法归纳特征贡献。"
    positive = [item for item in evidence if float(item["shap_value"]) > 0]
    negative = [item for item in evidence if float(item["shap_value"]) < 0]
    positive_sum = sum(float(item["shap_value"]) for item in positive)
    negative_sum = -sum(float(item["shap_value"]) for item in negative)
    if positive:
        text += f"所展示的特征中，{_feature_evidence(positive)}提高该分支的润滑不良概率。"
    if negative:
        text += f"{_feature_evidence(negative)}降低该分支的润滑不良概率。"
    text += _feature_context(evidence)
    if positive_sum or negative_sum:
        direction = (
            "提高润滑不良概率" if positive_sum > negative_sum
            else "降低润滑不良概率" if negative_sum > positive_sum
            else "相互抵消"
        )
        text += (
            f"这 {len(evidence)} 项证据的正向贡献合计 {positive_sum:.4f}、"
            f"负向贡献绝对值合计 {negative_sum:.4f}，合并作用为{direction}。"
            "这只归纳已展示特征；完整分支概率还包含 SHAP 背景值和其余特征的贡献。"
        )
    return text + "这些 SHAP 值解释完整 Stacking 分支，不直接解释三路 Judge 的最终概率。"


def build_judge_conclusion(report):
    """Describe branch agreement and the actual Judge threshold decision."""
    judge = _probability(report, "judge_probability", "risk_probability")
    threshold = _number(report.get("classification_threshold"))
    threshold = threshold if threshold is not None else 0.5
    branches = [
        ("5D 谱包络", _probability(report, "envelope_probability")),
        ("12D 频谱/ROI", _probability(report, "spectrum_probability")),
        ("完整 17D Stacking", _probability(report, "full_stacking_probability", "stacking_probability")),
    ]
    available = [(name, p) for name, p in branches if p is not None]
    parts = []
    if available:
        parts.append("三路分支输出为：" + "、".join(f"{name} {p:.2%}" for name, p in available) + "。")
    if len(available) == 3:
        positive_count = sum(p >= 0.5 for _, p in available)
        if positive_count in (0, 3):
            label = "轴承润滑不良" if positive_count else "非轴承润滑不良"
            parts.append(f"以 50% 为参照，三路均倾向于“{label}”。")
        else:
            parts.append("以 50% 为参照，三路判断存在分歧，Judge 使用训练得到的逻辑回归融合关系综合这些概率。")
    if judge is not None:
        label = report.get("diagnosis") or ("轴承润滑不良" if judge >= threshold else "非轴承润滑不良")
        comparison = "不低于" if judge >= threshold else "低于"
        parts.append(
            f"Judge 最终概率为 {judge:.2%}，{comparison} {threshold:.0%} 分类阈值，"
            f"最终输出“{label}”。"
        )
    else:
        parts.append("当前报告未提供 Judge 概率，无法补充其阈值判断。")
    parts.append("以上为当前样本的信号描述与模型证据，不单独构成润滑故障的物理因果证明。")
    return "".join(parts)


def build_evidence_narrative(report):
    sections = (
        ("信号状态", _signal_description(report)),
        ("频谱与 ROI 状态", _spectrum_description(report)),
        ("模型证据", _model_description(report)),
        ("综合结论", build_judge_conclusion(report)),
    )
    return "\n\n".join(f"**{title}：** {text}" for title, text in sections)


def build_evidence_details(report):
    """Small Markdown tables for the native collapsible detail panel."""
    stats = report.get("signal_statistics") or {}
    lines = [
        "时域统计来自原始信号；不作为新增特征输入诊断模型。",
        "", "| 时域指标 | 数值 |", "|---|---:|",
    ]
    for name, key in (
        ("样本数", "sample_count"), ("均值", "mean"), ("RMS", "rms"),
        ("绝对峰值", "peak_abs"), ("峰值因子", "crest_factor"),
        ("时域 Pearson 峭度（常量时无定义）", "kurtosis_pearson"),
    ):
        lines.append(f"| {name} | {_format(stats.get(key))} |")
    lines += ["", "SHAP 解释目标：完整 17D Stacking 分支概率 p_full。", "",
              "| 特征 | 特征值 | SHAP | 对 p_full 的作用 |", "|---|---:|---:|---|"]
    for item in report.get("top_evidence") or []:
        shap = _number(item.get("shap_value"))
        direction = "提高" if shap is not None and shap > 0 else "降低" if shap is not None and shap < 0 else "接近零"
        name = f"{item.get('feature', '')} {item.get('feature_name', '')}".replace("|", "\\|")
        lines.append(f"| {name} | {_format(item.get('feature_value'))} | {shap:+.4f} | {direction} |" if shap is not None
                     else f"| {name} | {_format(item.get('feature_value'))} | 未提供 | 未提供 |")
    return "\n".join(lines)
