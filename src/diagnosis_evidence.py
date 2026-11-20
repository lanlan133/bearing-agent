"""Spectral descriptions for the diagnosis UI; SHAP explains p_full only."""

from __future__ import annotations

import math

from config import HIGH_FREQ_RATIO


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


def build_probability_summary(report):
    """Show what the Judge probability measures and how it meets the threshold."""
    judge = _probability(report, "judge_probability", "risk_probability")
    if judge is None:
        return "**润滑不良概率**未提供。"
    threshold = _probability(report, "classification_threshold")
    if threshold is None:
        threshold = 0.5
    comparison = "达到" if judge >= threshold else "低于"
    return f"**润滑不良概率 {judge:.2%}**，{comparison} **{threshold * 100:g}%** 的分类阈值。"


def _roi_description(report, *, detailed=False):
    roi = report.get("roi") or {}
    if not roi.get("active"):
        if "active" in roi:
            return "本次未激活动态 ROI，尚未识别出符合当前算法条件的局部高频关注带。"
        return ""
    start, end = _number(roi.get("start")), _number(roi.get("end"))
    if start is None or end is None:
        return ""
    text = f"动态 ROI 识别出的局部关注频带为 {start / 1000:.2f}–{end / 1000:.2f} kHz"
    peak = _number(roi.get("peak_frequency"))
    if peak is not None:
        text += f"，主峰位于 {peak / 1000:.2f} kHz"
    centroid = _number(roi.get("centroid"))
    if detailed and centroid is not None:
        text += f"，能量加权质心位于 {centroid / 1000:.2f} kHz"
    text += "。"
    if detailed:
        text += "该频带描述局部高频结构，不等于全部信号能量的分布范围。"
    return text


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
    parts.append(_roi_description(report, detailed=True))
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


def build_key_evidence(report):
    roi = _roi_description(report) or "当前报告未提供 ROI 频带。"
    contributions = {"提高": [], "降低": []}
    for item in report.get("top_evidence") or []:
        shap = _number(item.get("shap_value"))
        name = item.get("feature_name") or item.get("feature")
        if shap and name:
            direction = "提高" if shap > 0 else "降低"
            contributions[direction].append((abs(shap), name))

    directions = []
    for direction, items in contributions.items():
        if items:
            strongest = sorted(items, reverse=True)[:2]
            names = "、".join(name for _, name in strongest)
            directions.append(f"{names}{direction}该分支的润滑不良概率")
    model = "当前报告未提供可归纳的特征贡献。"
    if directions:
        model = "在完整 17D Stacking 分支中，" + "，".join(directions) + "。"
    return f"**频谱与 ROI**　{roi}\n\n**模型证据**　{model}"


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


def _branch_probabilities(report):
    return (
        ("5D 谱包络", _probability(report, "envelope_probability")),
        ("12D 频谱/ROI", _probability(report, "spectrum_probability")),
        ("完整 17D Stacking", _probability(report, "full_stacking_probability", "stacking_probability")),
    )


def build_judge_conclusion(report):
    """Describe branch agreement and the actual Judge threshold decision."""
    judge = _probability(report, "judge_probability", "risk_probability")
    threshold = _number(report.get("classification_threshold"))
    threshold = threshold if threshold is not None else 0.5
    available = [(name, p) for name, p in _branch_probabilities(report) if p is not None]
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
        ("频谱与 ROI 状态", _spectrum_description(report)),
        ("模型证据", _model_description(report)),
        ("综合结论", build_judge_conclusion(report)),
    )
    return "\n\n".join(f"**{title}：** {text}" for title, text in sections)


def build_evidence_details(report, *, include_all_features=False):
    """Feature values and SHAP in the native collapsible detail panel."""
    lines = [
        "SHAP 解释目标：完整 17D Stacking 分支概率 p_full。", "",
        "| 特征 | 特征值 | SHAP | 对 p_full 的作用 |", "|---|---:|---:|---|",
    ]
    evidence = report.get("top_evidence") or []
    for item in evidence:
        shap = _number(item.get("shap_value"))
        direction = "提高" if shap is not None and shap > 0 else "降低" if shap is not None and shap < 0 else "接近零"
        name = f"{item.get('feature', '')} {item.get('feature_name', '')}".replace("|", "\\|")
        lines.append(f"| {name} | {_format(item.get('feature_value'))} | {shap:+.4f} | {direction} |" if shap is not None
                     else f"| {name} | {_format(item.get('feature_value'))} | 未提供 | 未提供 |")
    if include_all_features:
        displayed = {item.get("feature") for item in evidence}
        for feature, value in (report.get("features") or {}).items():
            if feature not in displayed:
                lines.append(f"| {feature} | {_format(value)} | — | — |")
    return "\n".join(lines)


def build_model_details(report):
    branches = " · ".join(
        f"{name} {value:.2%}" if value is not None else f"{name} 未提供"
        for name, value in _branch_probabilities(report)
    )
    sections = [
        f"**三路分支的润滑不良概率**\n\n{branches}",
        f"**频谱指标**\n\n{_spectrum_description(report)}",
        "**特征值与 SHAP**\n\n“—”表示该特征的 SHAP 值未随报告返回。\n\n"
        + build_evidence_details(report, include_all_features=True),
    ]
    domain = report.get("domain") or {}
    if domain.get("message"):
        sections.append(f"**适用范围**\n\n{domain['message']}")
    return "\n\n".join(sections)
