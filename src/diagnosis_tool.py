from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import shap

from msm_features import (
    analyze_signal,
)

from stacking_predictor import (
    predict_paper_stacking,
)

# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "XGBoost.joblib"
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

def load_diagnosis_model(
        model_file=DEFAULT_MODEL_FILE
):
    """
    加载训练好的 XGBoost Pipeline。
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
# 5. 获取 Pipeline 中真正的 XGBoost
# ============================================================

def get_xgb_estimator(
        pipeline
):
    """
    从 imblearn Pipeline 中取出最终 XGBoost 模型，
    用于 SHAP。
    """

    if hasattr(
        pipeline,
        "named_steps"
    ):

        if "model" not in pipeline.named_steps:

            raise KeyError(
                "Pipeline 中没有找到 model"
            )

        return pipeline.named_steps[
            "model"
        ]

    return pipeline


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
        pipeline,
        feature_array,
        top_n=5
):
    """
    对单条17维MSM特征做局部SHAP解释。
    """

    xgb_model = get_xgb_estimator(
        pipeline
    )

    explainer = shap.TreeExplainer(
        xgb_model
    )

    X = feature_array.reshape(
        1,
        -1
    )

    shap_values = explainer.shap_values(
        X
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
        f"MSM-ADASYN-Stacking 综合诊断结果为：{diagnosis}。"
        f"最终润滑失效概率为 {probability:.4f}。"
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
        f" 当前特征解释主要基于 XGBoost 基学习器的 SHAP 证据。"
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
    # 4) 加载模型
    # ========================================================

    model = load_diagnosis_model(
        model_file
    )

    # ============================================================
    # 5) Paper MSM-ADASYN-Stacking 最终诊断
    # ============================================================

    stacking_result = predict_paper_stacking(
        feature_array
    )

    # ------------------------------------------------------------
    # Stacking 最终概率
    # ------------------------------------------------------------

    probability = float(
        stacking_result[
            "final_probability"
        ]
    )

    prediction = int(
        stacking_result[
            "predicted_class"
        ]
    )

    diagnosis_label = (
        stacking_result[
            "diagnosis"
        ]
    )

    # ------------------------------------------------------------
    # Level-0 三个基学习器概率
    # ------------------------------------------------------------

    rf_probability = float(
        stacking_result[
            "rf_probability"
        ]
    )

    xgb_probability = float(
        stacking_result[
            "xgb_probability"
        ]
    )

    svm_probability = float(
        stacking_result[
            "svm_probability"
        ]
    )

    # ------------------------------------------------------------
    # 风险等级
    #
    # 目前仍然沿用现有风险等级函数；
    # 后面我们会重新针对 Stacking 概率校准 Review Policy
    # ------------------------------------------------------------

    risk_level = get_risk_level(
        probability
    )

    # ============================================================
    # 6) SHAP
    #
    # 注意：
    # 当前 SHAP 仍解释 XGBoost 基学习器，
    # 不代表整个 Stacking 最终模型的 SHAP。
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
            "MSM-ADASYN-Stacking",

        "model_version":
            stacking_result.get(
                "model_version",
                "paper_v1",
            ),

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
        # Stacking 最终概率
        #
        # 这里从现在开始不再是“XGBoost概率”
        # 而是最终 Stacking 润滑失效概率
        # ========================================================

        "risk_probability":
            probability,

        "stacking_probability":
            probability,

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
            stacking_result.get(
                "base_probability_mean"
            ),

        "base_probability_std":
            stacking_result.get(
                "base_probability_std"
            ),

        # ========================================================
        # 分类阈值
        # ========================================================

        "classification_threshold":
            stacking_result.get(
                "threshold",
                0.5,
            ),

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
        # SHAP证据
        #
        # 注意：
        # 当前这里解释的是 XGBoost 基学习器，
        # 不是完整 Stacking 模型。
        # ========================================================

        "explanation_model":
            "XGBoost base learner",

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