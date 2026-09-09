from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "XGBoost.joblib"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "test.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "explain"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. MSM 特征
# ============================================================

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. MSM 特征物理含义
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
# 4. 加载数据
# ============================================================

def load_data():

    print("=" * 75)
    print("XGBoost + SHAP 可解释诊断")
    print("=" * 75)

    print("\n模型:")
    print(MODEL_FILE)

    print("\n测试集:")
    print(TEST_FILE)

    model = joblib.load(
        MODEL_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    X_test = test_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    y_test = test_df[
        "target"
    ].to_numpy(
        dtype=int
    )

    print(
        f"\n测试样本数: {len(test_df)}"
    )

    print(
        f"特征维度: {X_test.shape[1]}"
    )

    return (
        model,
        test_df,
        X_test,
        y_test,
    )


# ============================================================
# 5. 取出 Pipeline 中真正的 XGBoost
# ============================================================

def get_xgboost_model(
        pipeline
):
    """
    当前保存的 XGBoost.joblib 是 imblearn Pipeline。

    推理阶段：
    ADASYN不会执行 fit_resample，
    真正解释的是 pipeline 中的 model。
    """

    if hasattr(
        pipeline,
        "named_steps"
    ):

        if "model" not in pipeline.named_steps:

            raise KeyError(
                "Pipeline 中没有找到 'model'"
            )

        model = pipeline.named_steps[
            "model"
        ]

        return model

    return pipeline


# ============================================================
# 6. 计算 SHAP
# ============================================================

def calculate_shap(
        model,
        X_test
):

    print("\n正在计算 SHAP...")

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_test
    )

    # --------------------------------------------------------
    # 兼容不同 SHAP 版本
    # --------------------------------------------------------

    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            shap_values = (
                shap_values[1]
            )

        else:

            shap_values = (
                shap_values[0]
            )

    shap_values = np.asarray(
        shap_values,
        dtype=float
    )

    print(
        f"SHAP shape: "
        f"{shap_values.shape}"
    )

    return (
        explainer,
        shap_values
    )


# ============================================================
# 7. 全局重要性
# ============================================================

def global_importance(
        shap_values
):

    mean_abs_shap = np.mean(
        np.abs(
            shap_values
        ),
        axis=0
    )

    importance_df = pd.DataFrame(
        {
            "feature":
                FEATURE_COLUMNS,

            "mean_abs_shap":
                mean_abs_shap,
        }
    )

    importance_df = (
        importance_df
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(
            drop=True
        )
    )

    importance_df[
        "feature_name"
    ] = importance_df[
        "feature"
    ].map(
        lambda x:
        FEATURE_MEANINGS[x][0]
    )

    importance_df[
        "mechanism"
    ] = importance_df[
        "feature"
    ].map(
        lambda x:
        FEATURE_MEANINGS[x][1]
    )

    importance_df.to_csv(
        REPORT_DIR
        / "global_shap_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 75)
    print("全局 SHAP Top-10")
    print("=" * 75)

    print(
        importance_df[
            [
                "feature",
                "feature_name",
                "mean_abs_shap",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    return importance_df


# ============================================================
# 8. SHAP summary 图
# ============================================================

def save_summary_plot(
        shap_values,
        X_test
):

    X_df = pd.DataFrame(
        X_test,
        columns=FEATURE_COLUMNS
    )

    plt.figure()

    shap.summary_plot(
        shap_values,
        X_df,
        show=False
    )

    plt.tight_layout()

    output_path = (
        REPORT_DIR
        / "shap_summary.png"
    )

    plt.savefig(
        output_path,
        dpi=160,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\nSHAP summary图:"
    )

    print(
        output_path
    )


# ============================================================
# 9. 自动挑选一个故障样本
# ============================================================

def choose_positive_sample(
        model_pipeline,
        test_df,
        X_test,
        y_test
):

    probabilities = (
        model_pipeline
        .predict_proba(
            X_test
        )[:, 1]
    )

    positive_indices = np.where(
        y_test == 1
    )[0]

    if len(
        positive_indices
    ) == 0:

        raise RuntimeError(
            "测试集中没有正类样本"
        )

    # --------------------------------------------------------
    # 选择概率最高的真实正类样本
    # --------------------------------------------------------

    selected_index = (
        positive_indices[
            np.argmax(
                probabilities[
                    positive_indices
                ]
            )
        ]
    )

    row = test_df.iloc[
        selected_index
    ]

    print("\n")
    print("=" * 75)
    print("单样本解释对象")
    print("=" * 75)

    print(
        f"测试集行号: "
        f"{selected_index}"
    )

    print(
        f"sample_index: "
        f"{row['sample_index']}"
    )

    print(
        f"equipment_id: "
        f"{row['equipment_id']}"
    )

    print(
        f"测点: "
        f"{row['measurement_point']}"
    )

    print(
        f"真实标签: "
        f"{row['label']}"
    )

    print(
        f"模型风险概率: "
        f"{probabilities[selected_index]:.4f}"
    )

    return (
        selected_index,
        probabilities
    )


# ============================================================
# 10. 单样本 Top-5 SHAP
# ============================================================

def explain_single_sample(
        selected_index,
        test_df,
        X_test,
        shap_values,
        probabilities
):

    sample_shap = shap_values[
        selected_index
    ]

    sample_features = X_test[
        selected_index
    ]

    order = np.argsort(
        np.abs(
            sample_shap
        )
    )[::-1]

    rows = []

    for rank, feature_idx in enumerate(
        order,
        start=1
    ):

        feature = FEATURE_COLUMNS[
            feature_idx
        ]

        feature_name, mechanism = (
            FEATURE_MEANINGS[
                feature
            ]
        )

        shap_value = float(
            sample_shap[
                feature_idx
            ]
        )

        feature_value = float(
            sample_features[
                feature_idx
            ]
        )

        # ----------------------------------------------------
        # SHAP 正负含义
        # ----------------------------------------------------

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

        rows.append(
            {
                "rank":
                    rank,

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

    explanation_df = pd.DataFrame(
        rows
    )

    explanation_df.to_csv(
        REPORT_DIR
        / "single_sample_shap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 75)
    print("单样本 SHAP Top-5")
    print("=" * 75)

    print(
        explanation_df[
            [
                "rank",
                "feature",
                "feature_name",
                "feature_value",
                "shap_value",
            ]
        ]
        .head(5)
        .to_string(
            index=False
        )
    )

    return explanation_df


# ============================================================
# 11. 生成简单 evidence 表
# ============================================================

def generate_evidence(
        selected_index,
        test_df,
        probabilities,
        explanation_df
):
    """
    生成单样本结构化诊断证据 JSON。

    注意：
    pandas / numpy 中的 int64、float64
    需要显式转换为 Python 原生 int / float，
    否则 json.dump 可能无法序列化。
    """

    sample = test_df.iloc[
        selected_index
    ]

    top5 = explanation_df.head(
        5
    )

    evidence_rows = []

    for _, row in top5.iterrows():

        evidence_rows.append(
            {
                "feature":
                    str(
                        row["feature"]
                    ),

                "feature_name":
                    str(
                        row["feature_name"]
                    ),

                "value":
                    float(
                        row["feature_value"]
                    ),

                "shap":
                    float(
                        row["shap_value"]
                    ),

                "direction":
                    str(
                        row["direction"]
                    ),

                "mechanism":
                    str(
                        row["mechanism"]
                    ),
            }
        )

    # ========================================================
    # 注意这里全部转成 Python 原生类型
    # ========================================================

    evidence = {

        "sample_index":
            int(
                sample["sample_index"]
            ),

        "equipment_id":
            str(
                sample["equipment_id"]
            ),

        "equipment_name":
            str(
                sample["equipment_name"]
            ),

        "measurement_point":
            str(
                sample["measurement_point"]
            ),

        "true_label":
            str(
                sample["label"]
            ),

        "true_target":
            int(
                sample["target"]
            ),

        "risk_probability":
            float(
                probabilities[
                    selected_index
                ]
            ),

        "top_features":
            evidence_rows,
    }

    # ========================================================
    # 保存 JSON
    # ========================================================

    import json

    output_path = (
        REPORT_DIR
        / "single_sample_evidence.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evidence,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nEvidence JSON:")

    print(
        output_path
    )

    return evidence

    # --------------------------------------------------------
    # JSON保存
    # --------------------------------------------------------

    import json

    output_path = (
        REPORT_DIR
        / "single_sample_evidence.json"
    )

    with open(
        output_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            evidence,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("\nEvidence JSON:")

    print(
        output_path
    )

    return evidence


# ============================================================
# 12. 主程序
# ============================================================

def main():

    (
        pipeline,
        test_df,
        X_test,
        y_test,
    ) = load_data()

    # --------------------------------------------------------
    # 真正的 XGBoost
    # --------------------------------------------------------

    xgb_model = get_xgboost_model(
        pipeline
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    (
        explainer,
        shap_values
    ) = calculate_shap(
        xgb_model,
        X_test
    )

    # --------------------------------------------------------
    # 全局
    # --------------------------------------------------------

    global_importance(
        shap_values
    )

    save_summary_plot(
        shap_values,
        X_test
    )

    # --------------------------------------------------------
    # 单样本
    # --------------------------------------------------------

    (
        selected_index,
        probabilities
    ) = choose_positive_sample(
        pipeline,
        test_df,
        X_test,
        y_test
    )

    explanation_df = explain_single_sample(
        selected_index,
        test_df,
        X_test,
        shap_values,
        probabilities
    )

    generate_evidence(
        selected_index,
        test_df,
        probabilities,
        explanation_df
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("SHAP解释分析完成")
    print("=" * 75)

    print(
        "\n输出目录:"
    )

    print(
        REPORT_DIR
    )


if __name__ == "__main__":
    main()