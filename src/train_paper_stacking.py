from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from imblearn.over_sampling import ADASYN

from xgboost import XGBClassifier


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "features.csv"
)

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
# 2. 论文参数
# ============================================================

RANDOM_STATE = 42

SHAP_BACKGROUND_SIZE = 64

FEATURE_NAMES = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. 创建三个 Level-0 基学习器
# ============================================================

def build_base_models(
    random_state
):

    # --------------------------------------------------------
    # Random Forest
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
# 4. Meta Learner
# ============================================================

def build_meta_model():

    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
    )


# ============================================================
# 5. 基学习器 OOF
# ============================================================

def generate_oof_probabilities(
    X_train,
    y_train,
    random_state
):

    n_samples = len(
        X_train
    )

    # 三个基模型的 OOF 概率
    oof_prob = np.zeros(
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

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        skf.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"\nFold {fold}/5"
        )

        X_fold_train = X_train[
            train_idx
        ]

        y_fold_train = y_train[
            train_idx
        ]

        X_fold_valid = X_train[
            valid_idx
        ]

        models = build_base_models(
            random_state
        )

        # ----------------------------------------------------
        # RF
        # ----------------------------------------------------

        models["rf"].fit(
            X_fold_train,
            y_fold_train,
        )

        oof_prob[
            valid_idx,
            0
        ] = models[
            "rf"
        ].predict_proba(
            X_fold_valid
        )[:, 1]

        # ----------------------------------------------------
        # XGBoost
        # ----------------------------------------------------

        models["xgb"].fit(
            X_fold_train,
            y_fold_train,
        )

        oof_prob[
            valid_idx,
            1
        ] = models[
            "xgb"
        ].predict_proba(
            X_fold_valid
        )[:, 1]

        # ----------------------------------------------------
        # SVM
        # ----------------------------------------------------

        models["svm"].fit(
            X_fold_train,
            y_fold_train,
        )

        oof_prob[
            valid_idx,
            2
        ] = models[
            "svm"
        ].predict_proba(
            X_fold_valid
        )[:, 1]

    return oof_prob


# ============================================================
# 6. 训练论文 Stacking
# ============================================================

def train_paper_stacking():

    print(
        "=" * 80
    )

    print(
        "MSM-ADASYN-Stacking 论文模型训练"
    )

    print(
        "=" * 80
    )

    # --------------------------------------------------------
    # A. 读取特征
    # --------------------------------------------------------

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\n数据集形状: {df.shape}"
    )

    missing_features = [
        name
        for name in FEATURE_NAMES
        if name not in df.columns
    ]

    if missing_features:

        raise ValueError(
            "缺少 MSM 特征列："
            f"{missing_features}"
        )

    if "label" not in df.columns:

        raise ValueError(
            "train.csv 中没有 label 列"
        )

    X = df[
        FEATURE_NAMES
    ].astype(
        float
    ).to_numpy()

    LABEL_MAP = {
        "非轴承润滑不良": 0,
        "轴承润滑不良": 1,
    }

    label_text = (
        df["label"]
        .astype(str)
        .str.strip()
    )

    unknown_labels = (
            set(label_text.unique())
            - set(LABEL_MAP.keys())
    )

    if unknown_labels:
        raise ValueError(
            f"发现未知标签：{unknown_labels}"
        )

    y = (
        label_text
        .map(LABEL_MAP)
        .astype(int)
        .to_numpy()
    )

    groups = (
        df["equipment_id"]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    print(
        "\n原始类别分布:"
    )

    classes, counts = np.unique(
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
            f"({label_name}): "
            f"{count}"
        )

    # --------------------------------------------------------
    # B. 标准化
    #
    # 注意：
    # 正式论文跨设备验证时 scaler 只能拟合训练集。
    # 当前脚本用于训练最终部署 bundle，因此这里拟合当前输入训练集。
    # --------------------------------------------------------

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(
        X
    )

    print(
        "\n标准化完成"
    )

    # --------------------------------------------------------
    # C. ADASYN
    # --------------------------------------------------------

    # ============================================================
    # ADASYN
    # 只用于训练阶段
    # ============================================================

    minority_count = int(
        np.sum(y == 1)
    )

    k_neighbors = min(
        5,
        minority_count - 1
    )

    if k_neighbors < 1:
        raise ValueError(
            "少数类样本数量不足，无法执行 ADASYN"
        )

    adasyn = ADASYN(
        sampling_strategy="minority",
        n_neighbors=k_neighbors,
        random_state=RANDOM_STATE,
    )

    X_aug, y_aug = adasyn.fit_resample(
        X_scaled,
        y,
    )

    print(
        "\nADASYN 完成"
    )

    print(
        f"增强前样本数: {len(y)}"
    )

    print(
        f"增强后样本数: {len(y_aug)}"
    )

    print(
        "\nADASYN 后类别分布:"
    )

    classes_aug, counts_aug = np.unique(
        y_aug,
        return_counts=True,
    )

    for cls, count in zip(
            classes_aug,
            counts_aug,
    ):
        print(
            f"class {cls}: {count}"
        )
    # ============================================================
    # 5折 OOF 概率
    # RF + XGBoost + SVM
    # ============================================================

    print(
        "\n开始生成 5 折 OOF 概率..."
    )

    oof_prob = generate_oof_probabilities(
        X_aug,
        y_aug,
        RANDOM_STATE,
    )

    print(
        "\nOOF 概率生成完成"
    )

    print(
        f"OOF shape: {oof_prob.shape}"
    )
    # --------------------------------------------------------
    # E. MSM 特征透传
    #
    # Fusion =
    # [RF prob, XGB prob, SVM prob, MSM-17D]
    #
    # 总维度 = 20
    # --------------------------------------------------------

    X_meta = np.hstack(
        [
            oof_prob,
            X_aug,
        ]
    )

    print(
        f"\nMeta 输入 shape: {X_meta.shape}"
    )

    if X_meta.shape[1] != 20:

        raise RuntimeError(
            "Meta 特征维度错误，"
            f"当前为 {X_meta.shape[1]}，"
            "理论应为 20"
        )

    # --------------------------------------------------------
    # F. 训练 Logistic Regression 元学习器
    # --------------------------------------------------------

    meta_model = (
        build_meta_model()
    )

    meta_model.fit(
        X_meta,
        y_aug,
    )

    print(
        "\nMeta learner 训练完成"
    )

    # --------------------------------------------------------
    # G. 用完整增强训练集重新训练 Level-0
    #
    # 部署推理时使用这三个完整基模型
    # --------------------------------------------------------

    final_models = (
        build_base_models(
            RANDOM_STATE
        )
    )

    print(
        "\n重新训练完整 Level-0 模型..."
    )

    for name, model in (
        final_models.items()
    ):

        print(
            f"训练 {name}..."
        )

        model.fit(
            X_aug,
            y_aug,
        )

    # --------------------------------------------------------
    # H. 保存 bundle
    # --------------------------------------------------------

    bundle = {

        "model_name": (
            "MSM-ADASYN-Stacking"
        ),

        "model_version": (
            "paper_v2_f6_full_stacking_shap"
        ),

        "feature_names": (
            FEATURE_NAMES
        ),

        "scaler": scaler,

        "adasyn_config": {
            "sampling_strategy": (
                "minority"
            ),
            "n_neighbors": (
                k_neighbors
            ),
            "random_state": (
                RANDOM_STATE
            ),
        },

        "rf": final_models[
            "rf"
        ],

        "xgb": final_models[
            "xgb"
        ],

        "svm": final_models[
            "svm"
        ],

        "meta": meta_model,

        "threshold": 0.5,

        "meta_feature_order": [
            "p_rf",
            "p_xgb",
            "p_svm",
            *FEATURE_NAMES,
        ],

        # 完整 Stacking SHAP 的背景数据必须来自原始真实样本，
        # 不能使用 ADASYN 合成样本。固定随机抽样保证可复现。
        "shap_background_raw": X[
            np.random.default_rng(
                RANDOM_STATE
            ).choice(
                len(X),
                size=min(
                    SHAP_BACKGROUND_SIZE,
                    len(X),
                ),
                replace=False,
            )
        ],

        "shap_config": {
            "explained_model": "complete_stacking_probability",
            "algorithm": "kernel_shap",
            "background_source": "original_real_samples_before_adasyn",
            "background_size": min(
                SHAP_BACKGROUND_SIZE,
                len(X),
            ),
            "nsamples": 256,
        },
    }

    model_path = (
        MODEL_DIR
        / "Paper_Stacking_Bundle.joblib"
    )

    joblib.dump(
        bundle,
        model_path,
    )

    print(
        "\n模型已保存:"
    )

    print(
        model_path
    )

    # --------------------------------------------------------
    # I. 保存配置说明
    # --------------------------------------------------------

    config = {
        "model": (
            "MSM-ADASYN-Stacking"
        ),
        "random_state": (
            RANDOM_STATE
        ),
        "feature_count": 17,
        "base_learners": {
            "RF": {
                "n_estimators": 300,
                "max_depth": None,
                "max_features": "sqrt",
                "min_samples_leaf": 1,
            },
            "XGBoost": {
                "n_estimators": 300,
                "max_depth": 3,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "SVM": {
                "kernel": "rbf",
                "C": 2.0,
                "gamma": "scale",
                "probability": True,
            },
        },
        "meta_learner": {
            "type": (
                "LogisticRegression"
            ),
            "C": 1.0,
            "class_weight": (
                "balanced"
            ),
            "max_iter": 3000,
        },
        "stacking": {
            "cv": 5,
            "shuffle": True,
            "threshold": 0.5,
            "feature_passthrough": True,
        },
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
        "\n配置已保存:"
    )

    print(
        config_path
    )

    print(
        "\n论文模型训练完成"
    )


# ============================================================
# 7. main
# ============================================================

if __name__ == "__main__":

    train_paper_stacking()
