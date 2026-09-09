from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

VALIDATION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "agent_validation"
)

VALIDATION_FILE = (
    VALIDATION_DIR
    / "agent_validation.csv"
)

FN_FILE = (
    VALIDATION_DIR
    / "false_negative_samples.csv"
)

FP_FILE = (
    VALIDATION_DIR
    / "false_positive_samples.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "agent_error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. Top特征列
# ============================================================

TOP_FEATURE_COLUMNS = [
    f"top{i}_feature"
    for i in range(1, 6)
]

TOP_NAME_COLUMNS = [
    f"top{i}_name"
    for i in range(1, 6)
]

TOP_SHAP_COLUMNS = [
    f"top{i}_shap"
    for i in range(1, 6)
]


# ============================================================
# 3. 读取
# ============================================================

def load_data():

    validation_df = pd.read_csv(
        VALIDATION_FILE
    )

    fn_df = pd.read_csv(
        FN_FILE
    )

    fp_df = pd.read_csv(
        FP_FILE
    )

    print("=" * 80)
    print("Diagnosis Agent 错误案例分析")
    print("=" * 80)

    print(
        f"\n全部测试样本: "
        f"{len(validation_df)}"
    )

    print(
        f"FN数量: "
        f"{len(fn_df)}"
    )

    print(
        f"FP数量: "
        f"{len(fp_df)}"
    )

    return (
        validation_df,
        fn_df,
        fp_df
    )


# ============================================================
# 4. 基础统计
# ============================================================

def basic_error_statistics(
        fn_df,
        fp_df
):

    print("\n")
    print("=" * 80)
    print("1. 错误概率统计")
    print("=" * 80)

    if len(fn_df) > 0:

        print("\nFN 风险概率:")

        print(
            fn_df[
                "risk_probability"
            ].describe()
        )

        print(
            f"\n最低FN概率: "
            f"{fn_df['risk_probability'].min():.4f}"
        )

        print(
            f"最高FN概率: "
            f"{fn_df['risk_probability'].max():.4f}"
        )

    if len(fp_df) > 0:

        print("\nFP 风险概率:")

        print(
            fp_df[
                "risk_probability"
            ].describe()
        )

        print(
            f"\n最低FP概率: "
            f"{fp_df['risk_probability'].min():.4f}"
        )

        print(
            f"最高FP概率: "
            f"{fp_df['risk_probability'].max():.4f}"
        )


# ============================================================
# 5. 统计Top-5特征出现频次
# ============================================================

def feature_frequency(
        df,
        error_name
):

    rows = []

    for rank in range(
        1,
        6
    ):

        feature_col = (
            f"top{rank}_feature"
        )

        name_col = (
            f"top{rank}_name"
        )

        shap_col = (
            f"top{rank}_shap"
        )

        for _, row in df.iterrows():

            feature = row.get(
                feature_col
            )

            feature_name = row.get(
                name_col
            )

            shap_value = row.get(
                shap_col
            )

            if pd.isna(
                feature
            ):
                continue

            rows.append(
                {
                    "rank":
                        rank,

                    "feature":
                        str(feature),

                    "feature_name":
                        str(feature_name),

                    "shap_value":
                        float(shap_value),
                }
            )

    long_df = pd.DataFrame(
        rows
    )

    if len(long_df) == 0:

        return pd.DataFrame()

    # --------------------------------------------------------
    # 按特征聚合
    # --------------------------------------------------------

    summary = (
        long_df
        .groupby(
            [
                "feature",
                "feature_name"
            ]
        )
        .agg(
            count=(
                "feature",
                "size"
            ),

            mean_shap=(
                "shap_value",
                "mean"
            ),

            mean_abs_shap=(
                "shap_value",
                lambda x:
                np.mean(
                    np.abs(x)
                )
            ),

            positive_count=(
                "shap_value",
                lambda x:
                int(
                    np.sum(
                        x > 0
                    )
                )
            ),

            negative_count=(
                "shap_value",
                lambda x:
                int(
                    np.sum(
                        x < 0
                    )
                )
            ),
        )
        .reset_index()
    )

    summary = summary.sort_values(
        [
            "count",
            "mean_abs_shap"
        ],
        ascending=[
            False,
            False
        ]
    )

    summary.to_csv(
        OUTPUT_DIR
        / f"{error_name}_feature_frequency.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print(
        f"{error_name} Top特征出现频次"
    )

    print("-" * 80)

    print(
        summary.head(
            10
        ).to_string(
            index=False
        )
    )

    return summary


# ============================================================
# 6. FN详细分析
# ============================================================

def analyze_false_negatives(
        fn_df
):

    print("\n")
    print("=" * 80)
    print("2. False Negative 漏诊分析")
    print("=" * 80)

    if len(fn_df) == 0:

        print(
            "没有FN样本"
        )

        return

    columns = [
        "sample_index",
        "equipment_id",
        "measurement_point",
        "risk_probability",
        "roi_active",
        "gaussian_success",
    ]

    for rank in range(
        1,
        6
    ):

        columns.extend(
            [
                f"top{rank}_feature",
                f"top{rank}_name",
                f"top{rank}_shap",
            ]
        )

    columns = [
        col
        for col in columns
        if col in fn_df.columns
    ]

    # --------------------------------------------------------
    # 最危险的漏诊：
    # 概率最低 = 模型最自信地认为正常
    # --------------------------------------------------------

    fn_sorted = fn_df.sort_values(
        "risk_probability",
        ascending=True
    )

    print(
        "\nFN详细列表 "
        "（按风险概率从低到高）:"
    )

    print(
        fn_sorted[
            [
                "sample_index",
                "equipment_id",
                "measurement_point",
                "risk_probability",
            ]
        ].to_string(
            index=False
        )
    )

    fn_sorted[
        columns
    ].to_csv(
        OUTPUT_DIR
        / "false_negative_detailed.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 最典型FN
    # --------------------------------------------------------

    worst = fn_sorted.iloc[
        0
    ]

    print("\n")
    print("最典型 FN")
    print("-" * 80)

    print(
        f"sample_index: "
        f"{int(worst['sample_index'])}"
    )

    print(
        f"equipment_id: "
        f"{worst['equipment_id']}"
    )

    print(
        f"risk_probability: "
        f"{worst['risk_probability']:.4f}"
    )

    print("\nTop-5 SHAP:")

    for rank in range(
        1,
        6
    ):

        feature = worst[
            f"top{rank}_feature"
        ]

        name = worst[
            f"top{rank}_name"
        ]

        shap = worst[
            f"top{rank}_shap"
        ]

        print(
            f"{rank}. "
            f"{feature} "
            f"{name} "
            f"| SHAP="
            f"{float(shap):+.4f}"
        )


# ============================================================
# 7. FP详细分析
# ============================================================

def analyze_false_positives(
        fp_df
):

    print("\n")
    print("=" * 80)
    print("3. False Positive 误报分析")
    print("=" * 80)

    if len(fp_df) == 0:

        print(
            "没有FP样本"
        )

        return

    # --------------------------------------------------------
    # 最典型误报：
    # 概率最高 = 模型最自信地认为故障
    # --------------------------------------------------------

    fp_sorted = fp_df.sort_values(
        "risk_probability",
        ascending=False
    )

    print(
        "\nFP详细列表 "
        "（按风险概率从高到低）:"
    )

    print(
        fp_sorted[
            [
                "sample_index",
                "equipment_id",
                "measurement_point",
                "risk_probability",
            ]
        ].to_string(
            index=False
        )
    )

    fp_sorted.to_csv(
        OUTPUT_DIR
        / "false_positive_detailed.csv",
        index=False,
        encoding="utf-8-sig"
    )

    worst = fp_sorted.iloc[
        0
    ]

    print("\n")
    print("最典型 FP")
    print("-" * 80)

    print(
        f"sample_index: "
        f"{int(worst['sample_index'])}"
    )

    print(
        f"equipment_id: "
        f"{worst['equipment_id']}"
    )

    print(
        f"risk_probability: "
        f"{worst['risk_probability']:.4f}"
    )

    print("\nTop-5 SHAP:")

    for rank in range(
        1,
        6
    ):

        print(
            f"{rank}. "
            f"{worst[f'top{rank}_feature']} "
            f"{worst[f'top{rank}_name']} "
            f"| SHAP="
            f"{float(worst[f'top{rank}_shap']):+.4f}"
        )


# ============================================================
# 8. 边界样本
# ============================================================

def analyze_uncertain_samples(
        validation_df
):
    """
    统计接近0.5决策阈值的样本。

    uncertainty = |p - 0.5|
    越小越不确定。
    """

    print("\n")
    print("=" * 80)
    print("4. 模型不确定样本")
    print("=" * 80)

    df = validation_df.copy()

    df[
        "distance_to_threshold"
    ] = np.abs(
        df[
            "risk_probability"
        ]
        - 0.5
    )

    uncertain = (
        df
        .sort_values(
            "distance_to_threshold"
        )
        .head(20)
    )

    uncertain.to_csv(
        OUTPUT_DIR
        / "most_uncertain_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        uncertain[
            [
                "sample_index",
                "equipment_id",
                "y_true",
                "y_pred",
                "risk_probability",
                "error_type",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# 9. 候选复核区间
# ============================================================

def evaluate_review_band(
        validation_df
):
    """
    这里只做分析，不改变最终分类模型。

    假设：
    0.30 <= p <= 0.70
    时进入人工/Agent二次复核。

    看能覆盖多少错误样本。
    """

    print("\n")
    print("=" * 80)
    print("5. 二次复核区间分析")
    print("=" * 80)

    lower = 0.30
    upper = 0.70

    review_mask = (
        (
            validation_df[
                "risk_probability"
            ] >= lower
        )
        &
        (
            validation_df[
                "risk_probability"
            ] <= upper
        )
    )

    review_df = validation_df[
        review_mask
    ]

    error_mask = (
        validation_df[
            "correct"
        ] == False
    )

    error_df = validation_df[
        error_mask
    ]

    covered_errors = error_df[
        (
            error_df[
                "risk_probability"
            ] >= lower
        )
        &
        (
            error_df[
                "risk_probability"
            ] <= upper
        )
    ]

    print(
        f"复核区间: "
        f"{lower:.2f} ~ {upper:.2f}"
    )

    print(
        f"进入复核的总样本数: "
        f"{len(review_df)}"
    )

    print(
        f"全部错误样本数: "
        f"{len(error_df)}"
    )

    print(
        f"被复核区间覆盖的错误数: "
        f"{len(covered_errors)}"
    )

    coverage = (
        len(
            covered_errors
        )
        /
        len(
            error_df
        )
        if len(
            error_df
        ) > 0
        else 0
    )

    print(
        f"错误覆盖率: "
        f"{coverage:.2%}"
    )

    review_df.to_csv(
        OUTPUT_DIR
        / "review_band_samples.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 10. 主程序
# ============================================================

def main():

    (
        validation_df,
        fn_df,
        fp_df
    ) = load_data()

    basic_error_statistics(
        fn_df,
        fp_df
    )

    # --------------------------------------------------------
    # FN特征
    # --------------------------------------------------------

    feature_frequency(
        fn_df,
        "FN"
    )

    # --------------------------------------------------------
    # FP特征
    # --------------------------------------------------------

    feature_frequency(
        fp_df,
        "FP"
    )

    analyze_false_negatives(
        fn_df
    )

    analyze_false_positives(
        fp_df
    )

    analyze_uncertain_samples(
        validation_df
    )

    evaluate_review_band(
        validation_df
    )

    print("\n")
    print("=" * 80)
    print("Agent错误案例分析完成")
    print("=" * 80)

    print("\n结果目录:")

    print(
        OUTPUT_DIR
    )


if __name__ == "__main__":

    main()