from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


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

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)

TRAIN_FILE = (
    OUTPUT_DIR
    / "train.csv"
)

TEST_FILE = (
    OUTPUT_DIR
    / "test.csv"
)

SUMMARY_FILE = (
    OUTPUT_DIR
    / "split_summary.txt"
)


# ============================================================
# 2. 参数
# ============================================================

TARGET_SAMPLING_RATE = 10240.0

TEST_SIZE = 0.20

RANDOM_STATE = 42

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. 标签映射
# ============================================================

LABEL_MAP = {
    "非轴承润滑不良": 0,
    "轴承润滑不良": 1,
}


# ============================================================
# 4. 读取并筛选
# ============================================================

def load_and_filter():

    print("=" * 70)
    print("设备级数据集准备")
    print("=" * 70)

    print("\n读取:")
    print(FEATURE_FILE)

    df = pd.read_csv(
        FEATURE_FILE
    )

    print(
        f"\n原始样本数: {len(df)}"
    )

    # --------------------------------------------------------
    # 筛选10240 Hz
    # --------------------------------------------------------

    df = df[
        np.isclose(
            df["sampling_rate"],
            TARGET_SAMPLING_RATE
        )
    ].copy()

    print(
        f"10240 Hz 样本数: {len(df)}"
    )

    # --------------------------------------------------------
    # 标签检查
    # --------------------------------------------------------

    unknown_labels = set(
        df["label"].dropna().unique()
    ) - set(
        LABEL_MAP.keys()
    )

    if unknown_labels:

        raise ValueError(
            f"发现未定义标签: {unknown_labels}"
        )

    df["target"] = df[
        "label"
    ].map(
        LABEL_MAP
    )

    if df["target"].isna().any():

        raise ValueError(
            "标签映射后存在 NaN"
        )

    df["target"] = df[
        "target"
    ].astype(int)

    # --------------------------------------------------------
    # 只保留有效17维特征
    # --------------------------------------------------------

    feature_matrix = df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    finite_mask = np.all(
        np.isfinite(
            feature_matrix
        ),
        axis=1
    )

    invalid_count = int(
        (~finite_mask).sum()
    )

    if invalid_count > 0:

        print(
            f"警告: 删除 {invalid_count} 条 "
            f"NaN/Inf 特征样本"
        )

        df = df[
            finite_mask
        ].copy()

    # --------------------------------------------------------
    # equipment_id 检查
    # --------------------------------------------------------

    if df[
        "equipment_id"
    ].isna().any():

        raise ValueError(
            "equipment_id 存在缺失，"
            "无法安全进行设备级划分"
        )

    return df


# ============================================================
# 5. 打印标签统计
# ============================================================

def print_label_distribution(
        df,
        title
):

    counts = (
        df["target"]
        .value_counts()
        .sort_index()
    )

    print(f"\n{title}")

    print("-" * 50)

    for label_value in [0, 1]:

        count = int(
            counts.get(
                label_value,
                0
            )
        )

        ratio = (
            count / len(df)
            if len(df) > 0
            else 0
        )

        label_name = (
            "非轴承润滑不良"
            if label_value == 0
            else "轴承润滑不良"
        )

        print(
            f"{label_value} "
            f"({label_name}): "
            f"{count} "
            f"({ratio:.2%})"
        )


# ============================================================
# 6. 设备级划分
# ============================================================

def group_split(df):

    """
    在多个随机种子中搜索一个更合理的设备级划分。

    要求：
    1. 同一个 equipment_id 绝不能跨 Train/Test
    2. Train/Test 都必须同时包含 0、1 两个类别
    3. 测试集样本比例尽量接近 TEST_SIZE
    4. Train/Test 正类比例尽量接近总体正类比例
    """

    groups = df["equipment_id"].astype(str)

    overall_positive_ratio = df["target"].mean()

    best_result = None
    best_score = np.inf

    print("\n搜索最佳设备级划分...")
    print("-" * 50)

    print(
        f"总体正类比例: "
        f"{overall_positive_ratio:.2%}"
    )

    # 搜索 0 ~ 999 共1000个随机种子
    for seed in range(1000):

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=seed,
        )

        train_idx, test_idx = next(
            splitter.split(
                df,
                y=df["target"],
                groups=groups
            )
        )

        train_part = df.iloc[train_idx]
        test_part = df.iloc[test_idx]

        # ---------------------------------------------
        # Train/Test 都必须有两个类别
        # ---------------------------------------------

        if train_part["target"].nunique() < 2:
            continue

        if test_part["target"].nunique() < 2:
            continue

        # ---------------------------------------------
        # 实际比例
        # ---------------------------------------------

        train_positive_ratio = (
            train_part["target"].mean()
        )

        test_positive_ratio = (
            test_part["target"].mean()
        )

        actual_test_size = (
            len(test_part) / len(df)
        )

        # ---------------------------------------------
        # 评分
        #
        # 越小越好：
        # - Train正类比例接近总体
        # - Test正类比例接近总体
        # - Test大小接近20%
        # ---------------------------------------------

        score = (
            abs(
                train_positive_ratio
                - overall_positive_ratio
            )
            +
            abs(
                test_positive_ratio
                - overall_positive_ratio
            )
            +
            abs(
                actual_test_size
                - TEST_SIZE
            )
        )

        if score < best_score:

            best_score = score

            best_result = (
                seed,
                train_idx,
                test_idx,
                train_positive_ratio,
                test_positive_ratio,
                actual_test_size,
            )

    # ========================================================
    # 检查是否搜索成功
    # ========================================================

    if best_result is None:

        raise RuntimeError(
            "无法找到满足条件的设备级划分"
        )

    (
        best_seed,
        train_idx,
        test_idx,
        train_positive_ratio,
        test_positive_ratio,
        actual_test_size,
    ) = best_result

    # ========================================================
    # 最终数据
    # ========================================================

    train_df = (
        df.iloc[train_idx]
        .copy()
        .reset_index(drop=True)
    )

    test_df = (
        df.iloc[test_idx]
        .copy()
        .reset_index(drop=True)
    )

    print(
        f"\n最佳随机种子: {best_seed}"
    )

    print(
        f"划分评分: {best_score:.6f}"
    )

    print(
        f"总体正类比例: "
        f"{overall_positive_ratio:.2%}"
    )

    print(
        f"训练集正类比例: "
        f"{train_positive_ratio:.2%}"
    )

    print(
        f"测试集正类比例: "
        f"{test_positive_ratio:.2%}"
    )

    print(
        f"实际测试集比例: "
        f"{actual_test_size:.2%}"
    )

    return (
        train_df,
        test_df
    )


# ============================================================
# 7. 验证设备是否泄漏
# ============================================================

def check_group_leakage(
        train_df,
        test_df
):

    train_devices = set(
        train_df[
            "equipment_id"
        ].astype(str)
    )

    test_devices = set(
        test_df[
            "equipment_id"
        ].astype(str)
    )

    overlap = (
        train_devices
        & test_devices
    )

    print("\n设备泄漏检查")
    print("-" * 50)

    print(
        f"训练设备数: "
        f"{len(train_devices)}"
    )

    print(
        f"测试设备数: "
        f"{len(test_devices)}"
    )

    print(
        f"设备重叠数: "
        f"{len(overlap)}"
    )

    if overlap:

        print(
            "重叠设备:"
        )

        print(
            sorted(
                overlap
            )
        )

        raise RuntimeError(
            "发现训练/测试设备重叠"
        )

    print(
        "设备级互斥检查: PASS"
    )

    return (
        train_devices,
        test_devices
    )


# ============================================================
# 8. 检查两类是否都存在
# ============================================================

def check_class_presence(
        train_df,
        test_df
):

    train_classes = set(
        train_df[
            "target"
        ].unique()
    )

    test_classes = set(
        test_df[
            "target"
        ].unique()
    )

    required = {
        0,
        1
    }

    if train_classes != required:

        raise RuntimeError(
            f"训练集类别异常: "
            f"{train_classes}"
        )

    if test_classes != required:

        raise RuntimeError(
            f"测试集类别异常: "
            f"{test_classes}"
        )

    print(
        "\n训练集和测试集均包含两个类别: PASS"
    )


# ============================================================
# 9. 保存
# ============================================================

def save_split(
        train_df,
        test_df
):

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    train_df.to_csv(
        TRAIN_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    test_df.to_csv(
        TEST_FILE,
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 10. 保存划分摘要
# ============================================================

def save_summary(
        df,
        train_df,
        test_df,
        train_devices,
        test_devices
):

    lines = []

    lines.append(
        "Bearing Lubrication Diagnosis Dataset Split"
    )

    lines.append(
        "=" * 60
    )

    lines.append(
        f"sampling_rate = "
        f"{TARGET_SAMPLING_RATE}"
    )

    lines.append(
        f"random_state = "
        f"{RANDOM_STATE}"
    )

    lines.append(
        f"test_size = "
        f"{TEST_SIZE}"
    )

    lines.append("")

    lines.append(
        f"Total samples: "
        f"{len(df)}"
    )

    lines.append(
        f"Train samples: "
        f"{len(train_df)}"
    )

    lines.append(
        f"Test samples: "
        f"{len(test_df)}"
    )

    lines.append("")

    lines.append(
        f"Train devices: "
        f"{len(train_devices)}"
    )

    lines.append(
        f"Test devices: "
        f"{len(test_devices)}"
    )

    lines.append(
        "Device overlap: 0"
    )

    lines.append("")

    for name, part in [
        ("TRAIN", train_df),
        ("TEST", test_df),
    ]:

        counts = (
            part[
                "target"
            ]
            .value_counts()
            .sort_index()
        )

        lines.append(
            f"{name} label 0: "
            f"{int(counts.get(0, 0))}"
        )

        lines.append(
            f"{name} label 1: "
            f"{int(counts.get(1, 0))}"
        )

    SUMMARY_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )


# ============================================================
# 11. 主程序
# ============================================================

def main():

    df = load_and_filter()

    print_label_distribution(
        df,
        "筛选后的总体标签分布"
    )

    print(
        f"\n总体设备数: "
        f"{df['equipment_id'].nunique()}"
    )

    # --------------------------------------------------------
    # Group Split
    # --------------------------------------------------------

    (
        train_df,
        test_df
    ) = group_split(
        df
    )

    # --------------------------------------------------------
    # 检查
    # --------------------------------------------------------

    (
        train_devices,
        test_devices
    ) = check_group_leakage(
        train_df,
        test_df
    )

    check_class_presence(
        train_df,
        test_df
    )

    # --------------------------------------------------------
    # 分布
    # --------------------------------------------------------

    print_label_distribution(
        train_df,
        "训练集标签分布"
    )

    print_label_distribution(
        test_df,
        "测试集标签分布"
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    save_split(
        train_df,
        test_df
    )

    save_summary(
        df,
        train_df,
        test_df,
        train_devices,
        test_devices
    )

    # --------------------------------------------------------
    # 完成
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("设备级数据集划分完成")
    print("=" * 70)

    print("\n训练集:")
    print(TRAIN_FILE)

    print("\n测试集:")
    print(TEST_FILE)

    print("\n划分摘要:")
    print(SUMMARY_FILE)


if __name__ == "__main__":
    main()