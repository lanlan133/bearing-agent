from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Matplotlib 中文显示
# ============================================================

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Arial Unicode MS"
]

plt.rcParams["axes.unicode_minus"] = False


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "features.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "feature_analysis"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 2. 读取数据
# ============================================================

def load_features():

    print("=" * 70)
    print("MSM-17D 特征质量分析")
    print("=" * 70)

    print("\n读取文件:")
    print(FEATURE_FILE)

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"\n数据形状: {df.shape}"
    )

    print(
        f"样本数量: {len(df)}"
    )

    return df


# ============================================================
# 3. 基础质量检查
# ============================================================

def basic_quality_check(df):

    print("\n")
    print("=" * 70)
    print("1. 基础质量检查")
    print("=" * 70)

    X = df[
        FEATURE_COLUMNS
    ].to_numpy(dtype=float)

    nan_count = int(
        np.isnan(X).sum()
    )

    inf_count = int(
        np.isinf(X).sum()
    )

    print(
        f"特征 NaN 总数: {nan_count}"
    )

    print(
        f"特征 Inf 总数: {inf_count}"
    )

    if "feature_valid" in df.columns:

        print(
            "feature_valid=False 数量:",
            int(
                (~df["feature_valid"].astype(bool)).sum()
            )
        )

    print("\n各特征基本统计:")

    stats = (
        df[FEATURE_COLUMNS]
        .describe()
        .T
    )

    print(
        stats[
            [
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max"
            ]
        ]
    )

    stats.to_csv(
        REPORT_DIR
        / "feature_statistics.csv",
        encoding="utf-8-sig"
    )


# ============================================================
# 4. 标签分布
# ============================================================

def analyze_labels(df):

    print("\n")
    print("=" * 70)
    print("2. 标签分布")
    print("=" * 70)

    counts = df[
        "label"
    ].value_counts(
        dropna=False
    )

    ratios = df[
        "label"
    ].value_counts(
        normalize=True,
        dropna=False
    )

    result = pd.DataFrame(
        {
            "count": counts,
            "ratio": ratios
        }
    )

    print(result)

    result.to_csv(
        REPORT_DIR
        / "label_distribution.csv",
        encoding="utf-8-sig"
    )

    plt.figure(
        figsize=(8, 5)
    )

    counts.plot(
        kind="bar"
    )

    plt.title(
        "Class Distribution"
    )

    plt.xlabel(
        "Label"
    )

    plt.ylabel(
        "Sample Count"
    )

    plt.xticks(
        rotation=20
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "label_distribution.png",
        dpi=150
    )

    plt.close()


# ============================================================
# 5. ROI 检查
# ============================================================

def analyze_roi(df):

    print("\n")
    print("=" * 70)
    print("3. ROI 状态分析")
    print("=" * 70)

    counts = df[
        "roi_active"
    ].value_counts(
        dropna=False
    )

    print(counts)

    inactive = df[
        ~df[
            "roi_active"
        ].astype(bool)
    ].copy()

    print(
        f"\nROI未激活数量: {len(inactive)}"
    )

    print(
        f"ROI未激活比例: "
        f"{len(inactive) / len(df):.4%}"
    )

    if len(inactive) > 0:

        columns = [
            "sample_index",
            "equipment_id",
            "measurement_point",
            "sampling_rate",
            "label"
        ]

        columns = [
            c for c in columns
            if c in inactive.columns
        ]

        inactive[
            columns
        ].to_csv(
            REPORT_DIR
            / "roi_inactive_samples.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nROI未激活样本的采样率分布:")

        print(
            inactive[
                "sampling_rate"
            ].value_counts()
        )

        print("\nROI未激活样本的标签分布:")

        print(
            inactive[
                "label"
            ].value_counts()
        )


# ============================================================
# 6. Gaussian 检查
# ============================================================

def analyze_gaussian(df):

    print("\n")
    print("=" * 70)
    print("4. Gaussian 拟合状态")
    print("=" * 70)

    counts = df[
        "gaussian_success"
    ].value_counts(
        dropna=False
    )

    print(counts)

    failed = df[
        ~df[
            "gaussian_success"
        ].astype(bool)
    ].copy()

    print(
        f"\nGaussian失败数量: {len(failed)}"
    )

    print(
        f"Gaussian失败比例: "
        f"{len(failed) / len(df):.4%}"
    )

    if len(failed) > 0:

        columns = [
            "sample_index",
            "equipment_id",
            "measurement_point",
            "sampling_rate",
            "label"
        ]

        columns = [
            c for c in columns
            if c in failed.columns
        ]

        failed[
            columns
        ].to_csv(
            REPORT_DIR
            / "gaussian_failed_samples.csv",
            index=False,
            encoding="utf-8-sig"
        )

        print("\nGaussian失败样本采样率分布:")

        print(
            failed[
                "sampling_rate"
            ].value_counts()
        )

        print("\nGaussian失败样本标签分布:")

        print(
            failed[
                "label"
            ].value_counts()
        )


# ============================================================
# 7. 采样率分析
# ============================================================

def analyze_sampling_rate(df):

    print("\n")
    print("=" * 70)
    print("5. 采样率分析")
    print("=" * 70)

    sampling_counts = (
        df[
            "sampling_rate"
        ]
        .value_counts()
        .sort_index()
    )

    print(
        "\n采样率样本数量:"
    )

    print(
        sampling_counts
    )

    sampling_counts.to_csv(
        REPORT_DIR
        / "sampling_rate_distribution.csv",
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 每个采样率下的特征中位数
    # 中位数比均值更不容易被极端值影响
    # --------------------------------------------------------

    grouped_median = (
        df
        .groupby(
            "sampling_rate"
        )[FEATURE_COLUMNS]
        .median()
    )

    grouped_median.to_csv(
        REPORT_DIR
        / "features_by_sampling_rate_median.csv",
        encoding="utf-8-sig"
    )

    print(
        "\n不同采样率下的特征中位数:"
    )

    print(
        grouped_median
    )

    # --------------------------------------------------------
    # 采样率 × 标签
    # --------------------------------------------------------

    cross = pd.crosstab(
        df[
            "sampling_rate"
        ],
        df[
            "label"
        ]
    )

    print(
        "\n采样率 × 标签:"
    )

    print(
        cross
    )

    cross.to_csv(
        REPORT_DIR
        / "sampling_rate_label_crosstab.csv",
        encoding="utf-8-sig"
    )


# ============================================================
# 8. 设备级检查
# ============================================================

def analyze_equipment(df):

    print("\n")
    print("=" * 70)
    print("6. 设备级分析")
    print("=" * 70)

    equipment_count = df[
        "equipment_id"
    ].nunique()

    print(
        f"设备数量: {equipment_count}"
    )

    samples_per_equipment = (
        df[
            "equipment_id"
        ]
        .value_counts()
    )

    print(
        "\n每台设备样本数统计:"
    )

    print(
        samples_per_equipment.describe()
    )

    samples_per_equipment.to_csv(
        REPORT_DIR
        / "samples_per_equipment.csv",
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # 检查一台设备是否出现多个标签
    # --------------------------------------------------------

    label_count_per_equipment = (
        df
        .groupby(
            "equipment_id"
        )[
            "label"
        ]
        .nunique()
    )

    mixed_equipment = (
        label_count_per_equipment[
            label_count_per_equipment > 1
        ]
    )

    print(
        "\n同时包含多个标签的设备数量:",
        len(mixed_equipment)
    )

    if len(mixed_equipment) > 0:

        print(
            "\n这些设备必须特别检查:"
        )

        print(
            mixed_equipment
        )

        mixed_equipment.to_csv(
            REPORT_DIR
            / "mixed_label_equipment.csv",
            encoding="utf-8-sig"
        )


# ============================================================
# 9. 标签之间的特征对比
# ============================================================

def analyze_features_by_label(df):

    print("\n")
    print("=" * 70)
    print("7. 两类 MSM-17D 特征比较")
    print("=" * 70)

    means = (
        df
        .groupby(
            "label"
        )[FEATURE_COLUMNS]
        .mean()
        .T
    )

    medians = (
        df
        .groupby(
            "label"
        )[FEATURE_COLUMNS]
        .median()
        .T
    )

    print(
        "\n各标签特征均值:"
    )

    print(means)

    print(
        "\n各标签特征中位数:"
    )

    print(medians)

    means.to_csv(
        REPORT_DIR
        / "feature_mean_by_label.csv",
        encoding="utf-8-sig"
    )

    medians.to_csv(
        REPORT_DIR
        / "feature_median_by_label.csv",
        encoding="utf-8-sig"
    )


# ============================================================
# 10. 17 个特征分布图
# ============================================================

def plot_feature_distributions(df):

    print("\n")
    print("=" * 70)
    print("8. 生成 f1 ~ f17 分布图")
    print("=" * 70)

    labels = list(
        df[
            "label"
        ].dropna().unique()
    )

    for feature in FEATURE_COLUMNS:

        plt.figure(
            figsize=(8, 5)
        )

        for label in labels:

            values = df.loc[
                df[
                    "label"
                ] == label,
                feature
            ]

            plt.hist(
                values,
                bins=40,
                alpha=0.45,
                density=True,
                label=str(label)
            )

        plt.xlabel(
            feature
        )

        plt.ylabel(
            "Density"
        )

        plt.title(
            f"{feature} Distribution by Label"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            REPORT_DIR
            / f"{feature}_distribution.png",
            dpi=150
        )

        plt.close()

    print(
        "17 张特征分布图生成完成"
    )


# ============================================================
# 11. 相关性矩阵
# ============================================================

def analyze_correlation(df):

    print("\n")
    print("=" * 70)
    print("9. MSM-17D 特征相关性")
    print("=" * 70)

    corr = df[
        FEATURE_COLUMNS
    ].corr()

    corr.to_csv(
        REPORT_DIR
        / "feature_correlation.csv",
        encoding="utf-8-sig"
    )

    plt.figure(
        figsize=(11, 9)
    )

    image = plt.imshow(
        corr,
        aspect="auto",
        vmin=-1,
        vmax=1
    )

    plt.colorbar(
        image,
        label="Correlation"
    )

    plt.xticks(
        range(17),
        FEATURE_COLUMNS,
        rotation=45
    )

    plt.yticks(
        range(17),
        FEATURE_COLUMNS
    )

    plt.title(
        "MSM-17D Feature Correlation"
    )

    plt.tight_layout()

    plt.savefig(
        REPORT_DIR
        / "feature_correlation.png",
        dpi=150
    )

    plt.close()

    print(
        "相关性矩阵已生成"
    )


# ============================================================
# 12. 主程序
# ============================================================

def main():

    df = load_features()

    basic_quality_check(
        df
    )

    analyze_labels(
        df
    )

    analyze_roi(
        df
    )

    analyze_gaussian(
        df
    )

    analyze_sampling_rate(
        df
    )

    analyze_equipment(
        df
    )

    analyze_features_by_label(
        df
    )

    plot_feature_distributions(
        df
    )

    analyze_correlation(
        df
    )

    print("\n")
    print("=" * 70)
    print("MSM-17D 特征质量分析完成")
    print("=" * 70)

    print(
        "\n分析结果目录:"
    )

    print(
        REPORT_DIR
    )


if __name__ == "__main__":
    main()