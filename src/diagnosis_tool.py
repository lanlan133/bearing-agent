from pathlib import Path
import json
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import shap

from msm_features import (
    analyze_signal,
)
from diagnosis_evidence import calculate_signal_statistics

# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "MultiView_Judge_Bundle.joblib"
)

DEFAULT_REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "diagnosis"
)

DEFAULT_REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


_SHAP_EXPLAINER_CACHE = {}


# ============================================================
# 2. MSM-17D 特征
# ============================================================

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. 特征物理含义
# ============================================================

FEATURE_MEANINGS = {

    "f1": (
        "宽带能量占比",
        "反映动态ROI局部高频共振能量的聚集程度"
    ),

    "f2": (
        "高频能量占比",
        "反映频谱能量由低频向高频迁移的程度"
    ),

    "f3": (
        "宽带质心",
        "反映局部高频共振中心位置"
    ),

    "f4": (
        "宽带扩展度",
        "反映高频共振频带横向扩展程度"
    ),

    "f5": (
        "宽带谱熵",
        "反映ROI内部频谱能量弥散程度"
    ),

    "f6": (
        "高频分段谱熵",
        "反映高频能量跨多个子频段扩散的程度"
    ),

    "f7": (
        "谱平坦度",
        "反映频谱由离散尖峰向宽带平坦背景演化的程度"
    ),

    "f8": (
        "频谱峭度",
        "反映频谱中突出尖峰、重尾和冲击性成分"
    ),

    "f9": (
        "频谱偏度",
        "反映频谱幅值统计分布的不对称程度"
    ),

    "f10": (
        "下包络面积比",
        "反映持续摩擦背景对应的谱底抬升程度"
    ),

    "f11": (
        "下包络均值",
        "反映持续摩擦背景的平均水平"
    ),

    "f12": (
        "下包络中位数",
        "反映高频谱底的稳健中心水平"
    ),

    "f13": (
        "下包络RMS",
        "反映谱底持续能量强度"
    ),

    "f14": (
        "上包络RMS",
        "反映瞬态冲击和局部峰值轮廓强度"
    ),

    "f15": (
        "高斯峰值A",
        "反映主要高频轮廓的峰值强度"
    ),

    "f16": (
        "高斯带宽sigma",
        "反映主要高频轮廓的横向展宽"
    ),

    "f17": (
        "高斯拟合RMSE",
        "反映实际频谱相对平滑高斯轮廓的不规则程度"
    ),
}


# ============================================================
# 4. 模型加载
# ============================================================

@lru_cache(maxsize=4)
def load_diagnosis_model(
        model_file=DEFAULT_MODEL_FILE
):
    """
    加载训练好的三路 Judge 部署模型包。
    """

    model_file = Path(
        model_file
    )

    if not model_file.exists():

        raise FileNotFoundError(
            f"找不到模型文件: {model_file}"
        )

    model = joblib.load(
        model_file
    )

    return model


# ============================================================
# 5. 完整 Stacking 概率函数
# ============================================================

def predict_complete_stacking_probability(
        model_bundle,
        feature_matrix
):
    """
    从原始17维MSM特征计算完整Stacking的正类概率。

    该函数覆盖：标准化、三个Level-0概率、17维特征直通和
    Level-1逻辑回归，是完整Stacking SHAP实际解释的函数。
    """

    X = np.asarray(
        feature_matrix,
        dtype=float,
    )

    if X.ndim == 1:
        X = X.reshape(1, -1)

    if X.ndim != 2 or X.shape[1] != 17:
        raise ValueError(
            "完整Stacking解释要求输入形状为 (n_samples, 17)，"
            f"当前为 {X.shape}。"
        )

    if not np.isfinite(X).all():
        raise ValueError(
            "完整Stacking解释输入存在 NaN 或 Inf。"
        )

    X_scaled = model_bundle[
        "scaler"
    ].transform(X)

    probability_features = np.column_stack(
        [
            model_bundle["rf"].predict_proba(
                X_scaled
            )[:, 1],
            model_bundle["xgb"].predict_proba(
                X_scaled
            )[:, 1],
            model_bundle["svm"].predict_proba(
                X_scaled
            )[:, 1],
        ]
    )

    X_meta = np.hstack(
        [
            probability_features,
            X_scaled,
        ]
    )

    return model_bundle[
        "meta"
    ].predict_proba(X_meta)[:, 1]


def predict_three_branch_judge(
        model_bundle,
        feature_array,
        sampling_rate,
):
    """Return all three branch probabilities and the Judge's final decision.

    The final class is determined exclusively by ``p_judge_three``.  There is
    deliberately no positive-guard OR rule in this production path.
    """

    if not isinstance(model_bundle, dict) or "model" not in model_bundle:
        raise ValueError(
            "三路 Judge 模型包无效；请使用 build_multiview_deployment.py 生成模型。"
        )

    model = model_bundle["model"]
    if not all(hasattr(model, name) for name in (
        "predict_frame", "full_", "envelope_", "spectrum_", "judge_three_"
    )):
        raise ValueError("三路 Judge 模型包缺少必要分支。")

    values = np.asarray(feature_array, dtype=float).reshape(1, -1)
    if values.shape != (1, 17) or not np.isfinite(values).all():
        raise ValueError("三路 Judge 输入必须是有限的 17 维 MSM 特征。")

    row = pd.DataFrame(values, columns=FEATURE_COLUMNS)
    row["sampling_rate"] = float(sampling_rate)
    prediction_row = model.predict_frame(row).iloc[0]

    base_probabilities = {
        name: float(pipeline.predict_proba(values)[:, 1][0])
        for name, pipeline in model.full_.models_.items()
    }
    judge_probability = float(prediction_row["p_judge_three"])
    threshold = float(model.config.threshold)
    prediction = int(judge_probability >= threshold)

    return {
        "p_envelope": float(prediction_row["p_envelope"]),
        "p_spectrum": float(prediction_row["p_spectrum"]),
        "p_full": float(prediction_row["p_full"]),
        "p_judge_three": judge_probability,
        "prediction": prediction,
        "diagnosis": "轴承润滑不良" if prediction else "非轴承润滑不良",
        "threshold": threshold,
        "base_probabilities": base_probabilities,
    }


# ============================================================
# 6. 特征字典 -> 17维数组
# ============================================================

def features_to_array(
        features
):
    """
    将 f1~f17 按固定顺序转换成模型输入。
    """

    values = np.array(
        [
            features[
                name
            ]
            for name in FEATURE_COLUMNS
        ],
        dtype=float
    )

    if values.shape != (17,):

        raise RuntimeError(
            f"MSM特征维度异常: {values.shape}"
        )

    if not np.all(
        np.isfinite(
            values
        )
    ):

        raise ValueError(
            "MSM-17D 存在 NaN 或 Inf"
        )

    return values


# ============================================================
# 7. 单样本 SHAP
# ============================================================

def explain_prediction(
        model_bundle,
        feature_array,
        top_n=5
):
    """
    对完整 17D Stacking 分支概率做局部 Kernel SHAP 解释。
    """

    if "shap_background_raw" not in model_bundle:
        raise KeyError(
            "模型包缺少 shap_background_raw。"
            "请重新运行 train_paper_stacking.py。"
        )

    background = np.asarray(
        model_bundle["shap_background_raw"],
        dtype=float,
    )

    if (
        background.ndim != 2
        or background.shape[1] != 17
        or len(background) == 0
    ):
        raise ValueError(
            "模型包中的SHAP背景数据无效："
            f"{background.shape}"
        )

    cache_key = id(model_bundle)
    explainer = _SHAP_EXPLAINER_CACHE.get(
        cache_key
    )

    if explainer is None:
        if "model" in model_bundle:
            probability_function = lambda values: (
                model_bundle["model"].full_.predict_proba(
                    np.asarray(values, dtype=float)
                )[:, 1]
            )
        else:
            probability_function = lambda values: (
                predict_complete_stacking_probability(
                    model_bundle,
                    values,
                )
            )

        explainer = shap.KernelExplainer(
            probability_function,
            background,
            link="identity",
            feature_names=FEATURE_COLUMNS,
        )

        _SHAP_EXPLAINER_CACHE[
            cache_key
        ] = explainer

    X = feature_array.reshape(
        1,
        -1
    )

    shap_config = model_bundle.get(
        "shap_config",
        {},
    )

    shap_values = explainer.shap_values(
        X,
        nsamples=int(
            shap_config.get(
                "nsamples",
                256,
            )
        ),
        l1_reg=0.0,
        silent=True,
    )

    # --------------------------------------------------------
    # 兼容不同 SHAP 版本
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            shap_values = shap_values[1]

        else:

            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values,
        dtype=float
    )

    if shap_values.ndim == 2:

        sample_shap = shap_values[0]

    else:

        sample_shap = shap_values

    if len(sample_shap) != 17:

        raise RuntimeError(
            f"SHAP维度异常: {sample_shap.shape}"
        )

    order = np.argsort(
        np.abs(
            sample_shap
        )
    )[::-1]

    evidence = []

    for rank, feature_index in enumerate(
        order[:top_n],
        start=1
    ):

        feature = FEATURE_COLUMNS[
            feature_index
        ]

        feature_name, mechanism = (
            FEATURE_MEANINGS[
                feature
            ]
        )

        shap_value = float(
            sample_shap[
                feature_index
            ]
        )

        feature_value = float(
            feature_array[
                feature_index
            ]
        )

        if shap_value > 0:

            direction = (
                "推动模型更倾向于判定为轴承润滑不良"
            )

        elif shap_value < 0:

            direction = (
                "推动模型更倾向于判定为非轴承润滑不良"
            )

        else:

            direction = (
                "对本次预测影响接近0"
            )

        evidence.append(
            {
                "rank":
                    int(rank),

                "feature":
                    feature,

                "feature_name":
                    feature_name,

                "feature_value":
                    feature_value,

                "shap_value":
                    shap_value,

                "direction":
                    direction,

                "mechanism":
                    mechanism,
            }
        )

    return evidence


# ============================================================
# 8. 风险等级
# ============================================================

def get_risk_level(
        probability
):
    """
    当前只是工程显示层的风险分级，
    不参与模型训练。

    后续可以依据设备维护策略重新调整。
    """

    probability = float(
        probability
    )

    if probability >= 0.85:

        return "高"

    elif probability >= 0.60:

        return "中"

    else:

        return "低"


# ============================================================
# 9. 生成简要诊断描述
# ============================================================

def build_diagnosis_summary(
    probability,
    prediction,
    evidence,
    rf_probability=None,
    xgb_probability=None,
    svm_probability=None,
    envelope_probability=None,
    spectrum_probability=None,
    full_stacking_probability=None,
):
    """
    生成不依赖LLM的基础诊断摘要。
    """

    if prediction == 1:

        diagnosis = (
            "模型判定为轴承润滑不良"
        )

    else:

        diagnosis = (
            "模型判定为非轴承润滑不良"
        )

    positive_features = [
        item
        for item in evidence
        if item[
            "shap_value"
        ] > 0
    ]

    if len(
        positive_features
    ) > 0:

        mechanism_text = "；".join(
            [
                (
                    f"{item['feature']} "
                    f"{item['feature_name']}："
                    f"{item['mechanism']}"
                )
                for item in positive_features[:3]
            ]
        )

    else:

        mechanism_text = (
            "当前Top证据主要推动模型判定为非轴承润滑不良"
        )

    summary = (
        f"三路 Judge 综合诊断结果为：{diagnosis}。"
        f"Judge 最终概率为 {probability:.4f}。"
    )

    if all(value is not None for value in (
        envelope_probability, spectrum_probability, full_stacking_probability
    )):
        summary += (
            " 三路输入概率分别为："
            f"5D包络={envelope_probability:.4f}，"
            f"12D频谱/ROI={spectrum_probability:.4f}，"
            f"完整17D Stacking={full_stacking_probability:.4f}。"
        )

    if (
            rf_probability is not None
            and xgb_probability is not None
            and svm_probability is not None
    ):
        summary += (
            f" Level-0 基学习器概率分别为："
            f"RF={rf_probability:.4f}，"
            f"XGBoost={xgb_probability:.4f}，"
            f"SVM={svm_probability:.4f}。"
        )

    summary += (
        f" 当前特征解释基于完整 17D Stacking 分支概率的 SHAP 证据。"
        f"{mechanism_text}。"
    )

    return summary


# ============================================================
# 10. 核心诊断函数
# ============================================================

def diagnose_signal(
        signal_data,
        sampling_rate,
        model_file=DEFAULT_MODEL_FILE,
        top_n=5
):
    """
    一条振动信号的完整诊断。

    输入
    ----------
    signal_data:
        一维振动信号

    sampling_rate:
        采样频率 Hz

    返回
    ----------
    dict
        完整结构化诊断结果
    """

    # ========================================================
    # 1) 输入检查
    # ========================================================

    signal_data = np.asarray(
        signal_data,
        dtype=float
    ).ravel()

    sampling_rate = float(
        sampling_rate
    )

    if len(
        signal_data
    ) == 0:

        raise ValueError(
            "输入振动信号为空"
        )

    if not np.all(
        np.isfinite(
            signal_data
        )
    ):

        raise ValueError(
            "输入振动信号存在 NaN 或 Inf"
        )

    if sampling_rate <= 0:

        raise ValueError(
            "采样频率必须大于0"
        )

    # 原始时域描述仅用于报告，不改变已训练模型的 17 维输入。
    signal_statistics = calculate_signal_statistics(signal_data)

    # ========================================================
    # 2) MSM分析
    # ========================================================

    msm_result = analyze_signal(
        signal_data,
        sampling_rate
    )

    features = msm_result[
        "features"
    ]

    roi = msm_result[
        "roi"
    ]

    gaussian_result = msm_result[
        "gaussian"
    ]

    # ========================================================
    # 3) 17维输入
    # ========================================================

    feature_array = features_to_array(
        features
    )

    X = feature_array.reshape(
        1,
        -1
    )

    # ========================================================
    # 4) 加载三路 Judge 模型包
    # ========================================================

    model = load_diagnosis_model(
        model_file
    )

    # ============================================================
    # 5) 三路 Judge 最终诊断
    #
    # 5D 谱包络、12D 频谱/ROI、完整 17D Stacking 分支分别出概率；
    # Judge 只以 p_judge_three 与固定 0.5 阈值输出最终类别。
    # 不使用完整 Stacking 阳性保底或任何 OR 改判规则。
    # ============================================================

    judge_result = predict_three_branch_judge(
        model,
        feature_array,
        sampling_rate,
    )

    probability = judge_result["p_judge_three"]
    prediction = judge_result["prediction"]
    diagnosis_label = judge_result["diagnosis"]
    envelope_probability = judge_result["p_envelope"]
    spectrum_probability = judge_result["p_spectrum"]
    full_stacking_probability = judge_result["p_full"]
    rf_probability = judge_result["base_probabilities"]["rf"]
    xgb_probability = judge_result["base_probabilities"]["xgb"]
    svm_probability = judge_result["base_probabilities"]["svm"]

    # ------------------------------------------------------------
    # 风险等级
    #
    # 风险等级仅用于辅助展示；最终诊断由 Stacking 的 0.5
    # 二分类阈值直接给出，风险等级不参与最终类别选择。
    # ------------------------------------------------------------

    risk_level = get_risk_level(
        probability
    )

    # ============================================================
    # 6) 完整 17D Stacking 分支 SHAP
    # ============================================================

    evidence = explain_prediction(
        model,
        feature_array,
        top_n=top_n
    )

    # ========================================================
    # 7) 摘要
    # ========================================================

    summary = build_diagnosis_summary(
        probability=probability,
        prediction=prediction,
        evidence=evidence,
        rf_probability=rf_probability,
        xgb_probability=xgb_probability,
        svm_probability=svm_probability,
        envelope_probability=envelope_probability,
        spectrum_probability=spectrum_probability,
        full_stacking_probability=full_stacking_probability,
    )

    # ========================================================
    # 8) ROI信息
    # ========================================================

    roi_info = {
        "active":
            bool(
                roi[
                    "active"
                ]
            ),

        "peak_frequency":
            (
                float(
                    roi[
                        "peak_frequency"
                    ]
                )
                if roi[
                    "peak_frequency"
                ] is not None
                else None
            ),

        "centroid":
            (
                float(
                    roi[
                        "centroid"
                    ]
                )
                if roi[
                    "centroid"
                ] is not None
                else None
            ),

        "spread":
            (
                float(
                    roi[
                        "spread"
                    ]
                )
                if roi[
                    "spread"
                ] is not None
                else None
            ),

        "start":
            (
                float(
                    roi[
                        "start"
                    ]
                )
                if roi[
                    "start"
                ] is not None
                else None
            ),

        "end":
            (
                float(
                    roi[
                        "end"
                    ]
                )
                if roi[
                    "end"
                ] is not None
                else None
            ),
    }

    # ========================================================
    # 9) Gaussian信息
    # ========================================================

    gaussian_info = {
        "success":
            bool(
                gaussian_result[
                    "success"
                ]
            ),

        "A":
            float(
                gaussian_result[
                    "A"
                ]
            ),

        "mu":
            float(
                gaussian_result[
                    "mu"
                ]
            ),

        "sigma":
            float(
                gaussian_result[
                    "sigma"
                ]
            ),

        "rmse":
            float(
                gaussian_result[
                    "rmse"
                ]
            ),
    }

    # ========================================================
    # 10) 17维特征转成原生float
    # ========================================================

    feature_dict = {
        feature:
            float(
                features[
                    feature
                ]
            )
        for feature in FEATURE_COLUMNS
    }

    # ========================================================
    # 最终结果
    # ========================================================

    result = {

        # ========================================================
        # 模型信息
        # ========================================================

        "model_name":
            "三路 Judge（5D 包络 + 12D 频谱/ROI + 完整 17D Stacking）",

        "model_version":
            f"multiview_judge_v{model.get('format_version', 1)}",

        # ========================================================
        # 最终诊断结果
        # ========================================================

        "diagnosis":
            diagnosis_label,

        # 保留旧字段，避免后面的 Agent / app 报错
        "prediction":
            prediction,

        # 新增一个更清楚的字段
        "predicted_class":
            prediction,

        # ========================================================
        # 最终决策概率和三路分支概率
        #
        # risk_probability 是 Judge 的最终概率；stacking_probability 保留为
        # 完整 17D Stacking 分支概率，以兼容已有调用方。
        # ========================================================

        "risk_probability":
            probability,

        "judge_probability":
            probability,

        "stacking_probability":
            full_stacking_probability,

        "full_stacking_probability":
            full_stacking_probability,

        "envelope_probability":
            envelope_probability,

        "spectrum_probability":
            spectrum_probability,

        # ========================================================
        # Level-0 三个基学习器概率
        # ========================================================

        "base_probabilities": {

            "rf":
                rf_probability,

            "xgb":
                xgb_probability,

            "svm":
                svm_probability,
        },

        "rf_probability":
            rf_probability,

        "xgb_probability":
            xgb_probability,

        "svm_probability":
            svm_probability,

        # ========================================================
        # 基模型一致性
        # ========================================================

        "base_probability_mean":
            float(np.mean([rf_probability, xgb_probability, svm_probability])),

        "base_probability_std":
            float(np.std([rf_probability, xgb_probability, svm_probability])),

        # ========================================================
        # 分类阈值
        # ========================================================

        "classification_threshold":
            judge_result["threshold"],

        # ========================================================
        # 风险等级
        # ========================================================

        "risk_level":
            risk_level,

        # ========================================================
        # 摘要
        # ========================================================

        "summary":
            summary,

        # ========================================================
        # 信号信息
        # ========================================================

        "sampling_rate":
            sampling_rate,

        "signal_statistics":
            signal_statistics,

        "signal_length":
            int(
                len(
                    signal_data
                )
            ),

        # ========================================================
        # MSM-17D
        # ========================================================

        "features":
            feature_dict,

        # ========================================================
        # 完整Stacking最终概率的SHAP证据
        # ========================================================

        "explanation_model":
            "完整 17D Stacking 分支概率（p_full）",

        "explanation_method":
            model.get(
                "shap_config",
                {}
            ).get(
                "algorithm",
                "kernel_shap",
            ),

        "explanation_background":
            model.get(
                "shap_config",
                {}
            ).get(
                "background_source",
                "original_real_samples_before_adasyn",
            ),

        "top_evidence":
            evidence,

        "roi":
            roi_info,

        "gaussian":
            gaussian_info,
    }

    return result


# ============================================================
# 11. 保存诊断JSON
# ============================================================

def save_diagnosis_json(
        result,
        output_path
):
    """
    保存结构化诊断结果。
    """

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
            result,
            f,
            ensure_ascii=False,
            indent=2
        )

    return output_path


# ============================================================
# 12. 打印诊断结果
# ============================================================

def print_diagnosis(
        result
):

    print("\n")

    print(
        "=" * 75
    )

    print(
        "轴承润滑诊断结果"
    )

    print(
        "=" * 75
    )

    print(
        f"诊断结果: "
        f"{result['diagnosis']}"
    )

    print(
        f"风险概率: "
        f"{result['risk_probability']:.4f}"
    )

    print(
        f"风险等级: "
        f"{result['risk_level']}"
    )

    print(
        f"采样频率: "
        f"{result['sampling_rate']:.2f} Hz"
    )

    print(
        f"信号长度: "
        f"{result['signal_length']}"
    )

    print("\n动态ROI:")

    print(
        f"  active: "
        f"{result['roi']['active']}"
    )

    if result[
        "roi"
    ][
        "active"
    ]:

        print(
            f"  peak: "
            f"{result['roi']['peak_frequency']:.2f} Hz"
        )

        print(
            f"  centroid: "
            f"{result['roi']['centroid']:.2f} Hz"
        )

        print(
            f"  ROI: "
            f"{result['roi']['start']:.2f}"
            f" ~ "
            f"{result['roi']['end']:.2f} Hz"
        )

    print("\nGaussian:")

    print(
        f"  success: "
        f"{result['gaussian']['success']}"
    )

    print(
        f"  sigma: "
        f"{result['gaussian']['sigma']:.2f} Hz"
    )

    print(
        f"  RMSE: "
        f"{result['gaussian']['rmse']:.6f}"
    )

    print("\nTop诊断证据:")

    print(
        "-" * 75
    )

    for item in result[
        "top_evidence"
    ]:

        print(
            f"{item['rank']}. "
            f"{item['feature']} "
            f"{item['feature_name']} "
            f"| value="
            f"{item['feature_value']:.6f} "
            f"| SHAP="
            f"{item['shap_value']:+.6f}"
        )

        print(
            f"   {item['direction']}"
        )

        print(
            f"   {item['mechanism']}"
        )

    print("\n诊断摘要:")

    print(
        result[
            "summary"
        ]
    )

    print(
        "=" * 75
    )
