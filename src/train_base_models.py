from pathlib import Path
import argparse
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from imblearn.over_sampling import ADASYN

from xgboost import XGBClassifier


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "paper_stacking"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. 论文 MSM-17D 特征
# ============================================================

FEATURE_NAMES = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. 论文类别定义
#
# 论文：
# 0 = 非润滑失效
# 1 = 润滑失效
#
# 你的 CSV：
# 非轴承润滑不良
# 轴承润滑不良
# ============================================================

LABEL_MAP = {
    "非轴承润滑不良": 0,
    "轴承润滑不良": 1,
}


# ============================================================
# 4. 默认随机种子
#
# 论文重复实验：
# 0~9 + 42
#
# 最终部署模型先使用 42
# ============================================================

RANDOM_STATE = 42


# ============================================================
# 5. 创建 Level-0 基学习器
#
# 参数来自论文表 9
# ============================================================

def build_base_models(
    random_state: int,
):

    # --------------------------------------------------------
    # Random Forest
    #
    # 论文：
    # n_estimators = 300
    # max_depth = None
    # max_features = sqrt
    # min_samples_leaf = 1
    # --------------------------------------------------------

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,

        random_state=random_state,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # XGBoost
    #
    # 论文：
    # n_estimators = 300
    # max_depth = 3
    # learning_rate = 0.05
    # subsample = 0.8
    # colsample_bytree = 0.8
    # objective = binary logistic
    # eval_metric = logloss
    # --------------------------------------------------------

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,

        subsample=0.8,
        colsample_bytree=0.8,

        objective="binary:logistic",
        eval_metric="logloss",

        random_state=random_state,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # SVM
    #
    # 论文：
    # kernel = RBF
    # C = 2.0
    # gamma = scale
    # probability = True
    # --------------------------------------------------------

    svm = SVC(
        kernel="rbf",
        C=2.0,
        gamma="scale",

        probability=True,

        random_state=random_state,
    )

    return {
        "rf": rf,
        "xgb": xgb,
        "svm": svm,
    }


# ============================================================
# 6. 创建 Level-1 元学习器
#
# 论文：
# Logistic Regression
# C = 1.0
# class_weight = balanced
# max_iter = 3000
# ============================================================

def build_meta_model():

    meta_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
    )

    return meta_model


# ============================================================
# 7. 读取数据
# ============================================================

def load_feature_dataset(
    csv_path: Path,
):

    print(
        "=" * 80
    )

    print(
        "读取 MSM-17D 特征数据"
    )

    print(
        "=" * 80
    )

    if not csv_path.exists():

        raise FileNotFoundError(
            f"找不到特征 CSV 文件：\n{csv_path}"
        )

    df = pd.read_csv(
        csv_path
    )

    print(
        f"\n文件：{csv_path}"
    )

    print(
        f"数据 shape：{df.shape}"
    )

    print(
        f"字段数量：{len(df.columns)}"
    )

    # --------------------------------------------------------
    # 检查 f1 ~ f17
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "CSV 缺少以下 MSM 特征：\n"
            f"{missing_features}\n\n"
            "请确认 CSV 中是否存在 f1 ~ f17。"
        )

    # --------------------------------------------------------
    # 检查 label
    # --------------------------------------------------------

    if "label" not in df.columns:

        raise ValueError(
            "CSV 中没有 label 列。"
        )

    # --------------------------------------------------------
    # 检查 equipment_id
    # --------------------------------------------------------

    if "equipment_id" not in df.columns:

        raise ValueError(
            "CSV 中没有 equipment_id 列。\n"
            "后续无法按照论文执行设备级互斥验证。"
        )

    # --------------------------------------------------------
    # 特征
    # --------------------------------------------------------

    X_df = (
        df[
            FEATURE_NAMES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
    )

    # --------------------------------------------------------
    # 检查缺失 / 非法值
    # --------------------------------------------------------

    if X_df.isna().any().any():

        bad_columns = (
            X_df.columns[
                X_df.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "MSM 特征存在 NaN 或无法转换为数字的值。\n"
            f"涉及字段：{bad_columns}"
        )

    X = X_df.to_numpy(
        dtype=float
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "MSM 特征中存在 inf 或 -inf。"
        )

    # --------------------------------------------------------
    # 标签字符串清理
    # --------------------------------------------------------

    label_text = (
        df["label"]
        .astype(str)
        .str.strip()
    )

    # --------------------------------------------------------
    # 检查未知标签
    # --------------------------------------------------------

    existing_labels = set(
        label_text.unique()
    )

    valid_labels = set(
        LABEL_MAP.keys()
    )

    unknown_labels = (
        existing_labels
        - valid_labels
    )

    if unknown_labels:

        raise ValueError(
            "发现未识别的 label：\n"
            f"{sorted(unknown_labels)}\n\n"
            "当前代码只识别：\n"
            "非轴承润滑不良\n"
            "轴承润滑不良"
        )

    # --------------------------------------------------------
    # 中文标签 -> 0 / 1
    # --------------------------------------------------------

    y = (
        label_text
        .map(
            LABEL_MAP
        )
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # 设备编号
    #
    # 后续用于：
    # 设备级互斥 train/test split
    # --------------------------------------------------------

    groups = (
        df["equipment_id"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    # --------------------------------------------------------
    # 输出数据概况
    # --------------------------------------------------------

    print(
        "\nMSM 特征："
    )

    print(
        FEATURE_NAMES
    )

    print(
        f"\nX shape：{X.shape}"
    )

    print(
        f"y shape：{y.shape}"
    )

    print(
        f"设备数：{len(np.unique(groups))}"
    )

    print(
        "\n标签映射："
    )

    print(
        "0 = 非轴承润滑不良"
    )

    print(
        "1 = 轴承润滑不良"
    )

    print(
        "\n原始类别分布："
    )

    classes,
    counts = np.unique(
        y,
        return_counts=True,
    )

    for cls, count in zip(
        classes,
        counts,
    ):

        label_name = (
            "轴承润滑不良"
            if cls == 1
            else "非轴承润滑不良"
        )

        print(
            f"class {cls} "
            f"({label_name})："
            f"{count}"
        )

    return (
        df,
        X,
        y,
        groups,
    )


# ============================================================
# 8. 生成 5 折 OOF 概率
#
# OOF：
#
# RF
# XGBoost
# SVM
#
# 输出：
# [p_RF, p_XGB, p_SVM]
# ============================================================

def generate_oof_probabilities(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
):

    n_samples = (
        X_train.shape[0]
    )

    oof_probabilities = np.zeros(
        (
            n_samples,
            3,
        ),
        dtype=float,
    )

    skf = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "开始生成 Level-0 5折 OOF 概率"
    )

    print(
        "=" * 80
    )

    for fold_index, (
        train_index,
        valid_index,
    ) in enumerate(
        skf.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"\nFold "
            f"{fold_index}/5"
        )

        X_fold_train = (
            X_train[
                train_index
            ]
        )

        y_fold_train = (
            y_train[
                train_index
            ]
        )

        X_fold_valid = (
            X_train[
                valid_index
            ]
        )

        # ----------------------------------------------------
        # 每折重新创建模型
        # ----------------------------------------------------

        models = (
            build_base_models(
                random_state=random_state,
            )
        )

        # ----------------------------------------------------
        # RF
        # ----------------------------------------------------

        print(
            "  Training RF..."
        )

        models[
            "rf"
        ].fit(
            X_fold_train,
            y_fold_train,
        )

        rf_probability = (
            models[
                "rf"
            ]
            .predict_proba(
                X_fold_valid
            )[:, 1]
        )

        oof_probabilities[
            valid_index,
            0
        ] = rf_probability

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        print(
            "  Training XGBoost..."
        )

        models[
            "xgb"
        ].fit(
            X_fold_train,
            y_fold_train,
        )

        xgb_probability = (
            models[
                "xgb"
            ]
            .predict_proba(
                X_fold_valid
            )[:, 1]
        )

        oof_probabilities[
            valid_index,
            1
        ] = xgb_probability

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        print(
            "  Training SVM..."
        )

        models[
            "svm"
        ].fit(
            X_fold_train,
            y_fold_train,
        )

        svm_probability = (
            models[
                "svm"
            ]
            .predict_proba(
                X_fold_valid
            )[:, 1]
        )

        oof_probabilities[
            valid_index,
            2
        ] = svm_probability

    print(
        "\nOOF 概率生成完成"
    )

    print(
        "OOF shape："
        f"{oof_probabilities.shape}"
    )

    return (
        oof_probabilities
    )


# ============================================================
# 9. 完整训练论文 MSM-ADASYN-Stacking
# ============================================================

def train_paper_stacking(
    csv_path: Path,
    random_state: int = 42,
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "论文 MSM-ADASYN-Stacking 模型训练"
    )

    print(
        "=" * 80
    )

    print(
        f"\nRandom State："
        f"{random_state}"
    )

    # ========================================================
    # A. 加载特征
    # ========================================================

    (
        df,
        X,
        y,
        groups,
    ) = load_feature_dataset(
        csv_path
    )

    # ========================================================
    # B. StandardScaler
    #
    # 最终部署模型：
    # 使用全部训练数据拟合 scaler。
    #
    # 注意：
    # 严格论文性能验证不能这样做。
    # validate_paper_stacking.py 中必须：
    #
    # train/test 先设备级划分
    # scaler 只 fit train
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "StandardScaler"
    )

    print(
        "=" * 80
    )

    scaler = (
        StandardScaler()
    )

    X_scaled = (
        scaler.fit_transform(
            X
        )
    )

    print(
        "\n标准化完成"
    )

    print(
        f"shape：{X_scaled.shape}"
    )

    print(
        "scaler 仅保存训练数据统计量，"
        "部署推理阶段只执行 transform。"
    )

    # ========================================================
    # C. ADASYN
    #
    # 论文：
    # sampling_strategy = minority
    #
    # KNN = min(5, n_min - 1)
    #
    # 当前工业数据满足 n_min > 5
    # 因此一般为 5
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "ADASYN"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # class 1 = 轴承润滑不良
    # --------------------------------------------------------

    minority_count = int(
        np.sum(
            y == 1
        )
    )

    majority_count = int(
        np.sum(
            y == 0
        )
    )

    print(
        f"\n非轴承润滑不良："
        f"{majority_count}"
    )

    print(
        f"轴承润滑不良："
        f"{minority_count}"
    )

    if minority_count < 2:

        raise ValueError(
            "润滑失效样本数量不足，"
            "无法执行 ADASYN。"
        )

    k_neighbors = min(
        5,
        minority_count - 1,
    )

    print(
        f"\nADASYN KNN："
        f"{k_neighbors}"
    )

    adasyn = ADASYN(
        sampling_strategy="minority",

        n_neighbors=k_neighbors,

        random_state=random_state,
    )

    X_augmented, y_augmented = (
        adasyn.fit_resample(
            X_scaled,
            y,
        )
    )

    print(
        "\nADASYN 完成"
    )

    print(
        f"增强前：{len(y)}"
    )

    print(
        f"增强后："
        f"{len(y_augmented)}"
    )

    print(
        "\nADASYN 后类别分布："
    )

    classes,
    counts = np.unique(
        y_augmented,
        return_counts=True,
    )

    for cls, count in zip(
        classes,
        counts,
    ):

        print(
            f"class {cls}: "
            f"{count}"
        )

    # ========================================================
    # D. 5折 OOF
    # ========================================================

    oof_probabilities = (
        generate_oof_probabilities(
            X_train=X_augmented,
            y_train=y_augmented,
            random_state=random_state,
        )
    )

    # ========================================================
    # E. MSM 特征透传
    #
    # 论文 Level-1：
    #
    # z_i =
    # [
    #   p_RF,
    #   p_XGB,
    #   p_SVM,
    #   MSM-17D
    # ]
    #
    # MSM 使用标准化空间中的特征。
    #
    # 总维度：
    #
    # 3 + 17 = 20
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "构建 Level-1 融合特征"
    )

    print(
        "=" * 80
    )

    X_meta = np.hstack(
        [
            oof_probabilities,
            X_augmented,
        ]
    )

    print(
        f"\n基学习器概率："
        f"{oof_probabilities.shape}"
    )

    print(
        f"MSM 特征："
        f"{X_augmented.shape}"
    )

    print(
        f"Meta 特征："
        f"{X_meta.shape}"
    )

    if (
        X_meta.shape[1]
        != 20
    ):

        raise RuntimeError(
            "Level-1 特征维度异常。\n"
            f"当前：{X_meta.shape[1]}\n"
            "理论：20"
        )

    # ========================================================
    # F. Logistic Regression
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "训练 Level-1 Logistic Regression"
    )

    print(
        "=" * 80
    )

    meta_model = (
        build_meta_model()
    )

    meta_model.fit(
        X_meta,
        y_augmented,
    )

    print(
        "\nMeta Learner 训练完成"
    )

    print(
        "coef shape："
        f"{meta_model.coef_.shape}"
    )

    # ========================================================
    # G. 使用完整增强训练集
    #    重新训练三个 Level-0 模型
    #
    # 部署时不能使用 CV 中临时模型。
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "训练最终 Level-0 基学习器"
    )

    print(
        "=" * 80
    )

    final_models = (
        build_base_models(
            random_state=random_state,
        )
    )

    print(
        "\nTraining final RF..."
    )

    final_models[
        "rf"
    ].fit(
        X_augmented,
        y_augmented,
    )

    print(
        "RF 完成"
    )

    print(
        "\nTraining final XGBoost..."
    )

    final_models[
        "xgb"
    ].fit(
        X_augmented,
        y_augmented,
    )

    print(
        "XGBoost 完成"
    )

    print(
        "\nTraining final SVM..."
    )

    final_models[
        "svm"
    ].fit(
        X_augmented,
        y_augmented,
    )

    print(
        "SVM 完成"
    )

    # ========================================================
    # H. 构建模型 Bundle
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "保存模型 Bundle"
    )

    print(
        "=" * 80
    )

    model_bundle = {

        # ----------------------------------------------------
        # 基本信息
        # ----------------------------------------------------

        "model_name": (
            "MSM-ADASYN-Stacking"
        ),

        "model_version": (
            "paper_v1"
        ),

        "random_state": (
            random_state
        ),

        # ----------------------------------------------------
        # 特征
        # ----------------------------------------------------

        "feature_names": (
            FEATURE_NAMES.copy()
        ),

        "feature_count": 17,

        # ----------------------------------------------------
        # 标签
        # ----------------------------------------------------

        "label_map": (
            LABEL_MAP.copy()
        ),

        "class_names": {
            0: (
                "非轴承润滑不良"
            ),
            1: (
                "轴承润滑不良"
            ),
        },

        # ----------------------------------------------------
        # 标准化
        # ----------------------------------------------------

        "scaler": scaler,

        # ----------------------------------------------------
        # ADASYN
        #
        # 注意：
        # ADASYN 只用于训练。
        # 推理阶段绝对不能执行。
        # ----------------------------------------------------

        "adasyn_config": {
            "sampling_strategy": (
                "minority"
            ),
            "n_neighbors": (
                k_neighbors
            ),
            "random_state": (
                random_state
            ),
        },

        # ----------------------------------------------------
        # Level-0
        # ----------------------------------------------------

        "rf": final_models[
            "rf"
        ],

        "xgb": final_models[
            "xgb"
        ],

        "svm": final_models[
            "svm"
        ],

        # ----------------------------------------------------
        # Level-1
        # ----------------------------------------------------

        "meta": meta_model,

        # ----------------------------------------------------
        # 分类阈值
        # ----------------------------------------------------

        "threshold": 0.5,

        # ----------------------------------------------------
        # Meta 输入顺序
        # ----------------------------------------------------

        "meta_feature_order": [
            "p_rf",
            "p_xgb",
            "p_svm",
            *FEATURE_NAMES,
        ],

        # ----------------------------------------------------
        # 训练数据元数据
        # ----------------------------------------------------

        "training_metadata": {

            "original_sample_count": (
                int(
                    len(y)
                )
            ),

            "augmented_sample_count": (
                int(
                    len(y_augmented)
                )
            ),

            "equipment_count": (
                int(
                    len(
                        np.unique(
                            groups
                        )
                    )
                )
            ),

            "class_0_count": (
                int(
                    np.sum(
                        y == 0
                    )
                )
            ),

            "class_1_count": (
                int(
                    np.sum(
                        y == 1
                    )
                )
            ),
        },
    }

    # ========================================================
    # I. 保存 joblib
    # ========================================================

    model_path = (
        MODEL_DIR
        / "Paper_Stacking_Bundle.joblib"
    )

    joblib.dump(
        model_bundle,
        model_path,
    )

    print(
        "\n模型已保存："
    )

    print(
        model_path
    )

    # ========================================================
    # J. 保存论文模型参数 JSON
    # ========================================================

    config = {

        "model_name": (
            "MSM-ADASYN-Stacking"
        ),

        "random_state": (
            random_state
        ),

        "labels": {
            "0": (
                "非轴承润滑不良"
            ),
            "1": (
                "轴承润滑不良"
            ),
        },

        "MSM": {
            "dimension": 17,
            "features": (
                FEATURE_NAMES
            ),
        },

        "standardization": {
            "type": (
                "StandardScaler"
            ),
            "fit_on_training_data_only": (
                True
            ),
        },

        "ADASYN": {
            "sampling_strategy": (
                "minority"
            ),
            "n_neighbors": (
                k_neighbors
            ),
            "random_state": (
                random_state
            ),
        },

        "RandomForest": {
            "n_estimators": 300,
            "max_depth": None,
            "max_features": (
                "sqrt"
            ),
            "min_samples_leaf": 1,
        },

        "XGBoost": {
            "n_estimators": 300,
            "max_depth": 3,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": (
                "binary:logistic"
            ),
            "eval_metric": (
                "logloss"
            ),
        },

        "SVM": {
            "kernel": (
                "rbf"
            ),
            "C": 2.0,
            "gamma": (
                "scale"
            ),
            "probability": (
                True
            ),
        },

        "Stacking": {
            "cv": 5,
            "shuffle": (
                True
            ),
            "base_probability_features": [
                "RF",
                "XGBoost",
                "SVM",
            ],
            "msm_feature_passthrough": (
                True
            ),
            "meta_input_dimension": 20,
        },

        "LogisticRegression": {
            "C": 1.0,
            "class_weight": (
                "balanced"
            ),
            "max_iter": 3000,
        },

        "classification_threshold": (
            0.5
        ),
    }

    config_path = (
        REPORT_DIR
        / "paper_model_config.json"
    )

    config_path.write_text(
        json.dumps(
            config,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n参数配置已保存："
    )

    print(
        config_path
    )

    # ========================================================
    # K. 完成
    # ========================================================

    print(
        "\n"
        + "=" * 80
    )

    print(
        "MSM-ADASYN-Stacking "
        "最终部署模型训练完成"
    )

    print(
        "=" * 80
    )

    print(
        "\n注意："
    )

    print(
        "该 joblib 用于最终部署模型。"
    )

    print(
        "论文性能复现必须另外执行"
        "设备级互斥 train/test 验证，"
        "不能用当前训练结果直接报告 Accuracy/F1/AUC。"
    )

    return model_bundle


# ============================================================
# 10. 命令行入口
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "训练论文 MSM-ADASYN-Stacking 模型"
        )
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help=(
            "包含 f1~f17、label、equipment_id "
            "的 CSV 文件路径"
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help=(
            "随机种子，默认 42"
        ),
    )

    return parser.parse_args()


# ============================================================
# 11. main
# ============================================================

if __name__ == "__main__":

    args = parse_args()

    csv_path = Path(
        args.data
    ).expanduser().resolve()

    train_paper_stacking(
        csv_path=csv_path,
        random_state=args.seed,
    )