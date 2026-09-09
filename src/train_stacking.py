from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import ADASYN
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAIN_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "train.csv"
)

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "test.csv"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "stacking"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. 参数
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

FEATURE_COLUMNS = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. 读取数据
# ============================================================

def load_dataset():

    train_df = pd.read_csv(
        TRAIN_FILE
    )

    test_df = pd.read_csv(
        TEST_FILE
    )

    print("=" * 80)
    print("OOF Stacking 训练")
    print("=" * 80)

    print(
        f"\n训练样本数: "
        f"{len(train_df)}"
    )

    print(
        f"测试样本数: "
        f"{len(test_df)}"
    )

    print(
        f"训练设备数: "
        f"{train_df['equipment_id'].nunique()}"
    )

    print(
        f"测试设备数: "
        f"{test_df['equipment_id'].nunique()}"
    )

    # --------------------------------------------------------
    # 再次检查设备是否泄漏
    # --------------------------------------------------------

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

    if overlap:

        raise RuntimeError(
            f"Train/Test 存在设备重叠: "
            f"{overlap}"
        )

    print(
        "训练集 / 测试集设备互斥检查: PASS"
    )

    # --------------------------------------------------------
    # 特征
    # --------------------------------------------------------

    X_train = train_df[
        FEATURE_COLUMNS
    ].to_numpy(
        dtype=float
    )

    y_train = train_df[
        "target"
    ].to_numpy(
        dtype=int
    )

    groups_train = train_df[
        "equipment_id"
    ].astype(str).to_numpy()

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

    return (
        train_df,
        test_df,
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
    )


# ============================================================
# 4. 三个基模型
# ============================================================

def build_base_models():

    models = {}

    # --------------------------------------------------------
    # Random Forest
    # --------------------------------------------------------

    models[
        "RandomForest"
    ] = Pipeline(
        steps=[
            (
                "adasyn",
                ADASYN(
                    random_state=RANDOM_STATE,
                    n_neighbors=5,
                )
            ),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=400,
                    max_depth=None,
                    min_samples_split=2,
                    min_samples_leaf=1,
                    max_features="sqrt",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # SVM
    # --------------------------------------------------------

    models[
        "SVM"
    ] = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "adasyn",
                ADASYN(
                    random_state=RANDOM_STATE,
                    n_neighbors=5,
                )
            ),
            (
                "model",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    probability=True,
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    models[
        "XGBoost"
    ] = Pipeline(
        steps=[
            (
                "adasyn",
                ADASYN(
                    random_state=RANDOM_STATE,
                    n_neighbors=5,
                )
            ),
            (
                "model",
                XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    objective="binary:logistic",
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                )
            ),
        ]
    )

    return models


# ============================================================
# 5. 指标函数
# ============================================================

def calculate_metrics(
        y_true,
        y_pred,
        y_prob
):

    return {
        "accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "recall":
            recall_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "f1":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "auc":
            roc_auc_score(
                y_true,
                y_prob
            ),
    }


# ============================================================
# 6. 生成 OOF 概率
# ============================================================

def generate_oof_predictions(
        models,
        X_train,
        y_train,
        groups_train
):
    """
    每个训练样本的 OOF 概率，都必须来自一个
    没见过该样本、也没见过该设备的基模型。

    返回：
    oof_matrix:
        shape = (n_train, 3)
    """

    print("\n")
    print("=" * 80)
    print("生成基学习器 OOF 概率")
    print("=" * 80)

    model_names = list(
        models.keys()
    )

    n_samples = len(
        X_train
    )

    n_models = len(
        model_names
    )

    oof_matrix = np.full(
        (
            n_samples,
            n_models
        ),
        np.nan,
        dtype=float
    )

    # --------------------------------------------------------
    # 用于确认每个样本只预测一次
    # --------------------------------------------------------

    prediction_count = np.zeros(
        n_samples,
        dtype=int
    )

    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # ========================================================
    # CV
    # ========================================================

    for fold, (
            fold_train_idx,
            fold_valid_idx
    ) in enumerate(
        cv.split(
            X_train,
            y_train,
            groups_train
        ),
        start=1
    ):

        print("\n")
        print("-" * 80)

        print(
            f"OOF Fold {fold}/{N_SPLITS}"
        )

        print("-" * 80)

        # ----------------------------------------------------
        # 设备泄漏检查
        # ----------------------------------------------------

        fold_train_groups = set(
            groups_train[
                fold_train_idx
            ]
        )

        fold_valid_groups = set(
            groups_train[
                fold_valid_idx
            ]
        )

        overlap = (
            fold_train_groups
            & fold_valid_groups
        )

        if overlap:

            raise RuntimeError(
                f"OOF 第 {fold} 折发现设备泄漏"
            )

        print(
            f"训练样本: "
            f"{len(fold_train_idx)}"
        )

        print(
            f"验证样本: "
            f"{len(fold_valid_idx)}"
        )

        print(
            f"训练设备: "
            f"{len(fold_train_groups)}"
        )

        print(
            f"验证设备: "
            f"{len(fold_valid_groups)}"
        )

        # ====================================================
        # 三个模型
        # ====================================================

        for model_idx, model_name in enumerate(
            model_names
        ):

            print(
                f"  -> {model_name}"
            )

            fold_model = clone(
                models[
                    model_name
                ]
            )

            # ------------------------------------------------
            # Scaler / ADASYN 都只在当前训练折 fit
            # ------------------------------------------------

            fold_model.fit(
                X_train[
                    fold_train_idx
                ],
                y_train[
                    fold_train_idx
                ]
            )

            prob = (
                fold_model.predict_proba(
                    X_train[
                        fold_valid_idx
                    ]
                )[:, 1]
            )

            oof_matrix[
                fold_valid_idx,
                model_idx
            ] = prob

        prediction_count[
            fold_valid_idx
        ] += 1

    # ========================================================
    # OOF 完整性检查
    # ========================================================

    print("\n")
    print("=" * 80)
    print("OOF 完整性检查")
    print("=" * 80)

    nan_count = int(
        np.isnan(
            oof_matrix
        ).sum()
    )

    print(
        f"OOF NaN 数量: "
        f"{nan_count}"
    )

    wrong_count = int(
        np.sum(
            prediction_count != 1
        )
    )

    print(
        f"OOF预测次数异常样本数: "
        f"{wrong_count}"
    )

    if nan_count > 0:

        raise RuntimeError(
            "OOF矩阵存在 NaN"
        )

    if wrong_count > 0:

        raise RuntimeError(
            "部分样本没有恰好获得一次OOF预测"
        )

    print(
        "OOF完整性检查: PASS"
    )

    return (
        oof_matrix,
        model_names
    )


# ============================================================
# 7. 在完整训练集训练基模型
#    并生成独立测试集概率
# ============================================================

def train_full_base_models(
        models,
        X_train,
        y_train,
        X_test
):
    """
    TEST 仍然完全不参与 fit。

    返回：
    test_probability_matrix
    trained_models
    """

    print("\n")
    print("=" * 80)
    print("训练完整基模型并生成 Test 概率")
    print("=" * 80)

    model_names = list(
        models.keys()
    )

    test_probability_matrix = np.zeros(
        (
            len(X_test),
            len(model_names)
        ),
        dtype=float
    )

    trained_models = {}

    for model_idx, model_name in enumerate(
        model_names
    ):

        print(
            f"\n训练完整 {model_name}..."
        )

        final_model = clone(
            models[
                model_name
            ]
        )

        final_model.fit(
            X_train,
            y_train
        )

        probability = (
            final_model.predict_proba(
                X_test
            )[:, 1]
        )

        test_probability_matrix[
            :,
            model_idx
        ] = probability

        trained_models[
            model_name
        ] = final_model

        # 保存最终基模型
        joblib.dump(
            final_model,
            MODEL_DIR
            / f"stacking_{model_name}.joblib"
        )

    return (
        test_probability_matrix,
        trained_models
    )


# ============================================================
# 8. Logistic Regression baseline
#    仅使用 MSM-17D
# ============================================================

def train_lr_baseline(
        X_train,
        y_train,
        X_test,
        y_test
):
    """
    一个非常重要的基准实验：

    MSM-17D
        ↓
    StandardScaler
        ↓
    Logistic Regression
    """

    print("\n")
    print("=" * 80)
    print("Baseline：MSM-17D + Logistic Regression")
    print("=" * 80)

    lr_pipeline = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    lr_pipeline.fit(
        X_train,
        y_train
    )

    y_pred = lr_pipeline.predict(
        X_test
    )

    y_prob = (
        lr_pipeline.predict_proba(
            X_test
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    print_metrics(
        "LR_Baseline",
        metrics
    )

    joblib.dump(
        lr_pipeline,
        MODEL_DIR
        / "LR_Baseline.joblib"
    )

    return (
        lr_pipeline,
        metrics
    )


# ============================================================
# 9. Stacking 元学习器
# ============================================================

def train_meta_model(
        X_train,
        y_train,
        oof_matrix,
        X_test,
        y_test,
        test_probability_matrix
):
    """
    Meta输入：

    [f1...f17,
     P_RF,
     P_SVM,
     P_XGB]

    一共20维。
    """

    print("\n")
    print("=" * 80)
    print("训练 Logistic Regression Stacking 元学习器")
    print("=" * 80)

    # --------------------------------------------------------
    # 训练元特征
    # --------------------------------------------------------

    X_meta_train = np.hstack(
        [
            X_train,
            oof_matrix
        ]
    )

    # --------------------------------------------------------
    # 测试元特征
    # --------------------------------------------------------

    X_meta_test = np.hstack(
        [
            X_test,
            test_probability_matrix
        ]
    )

    print(
        f"Meta训练特征维度: "
        f"{X_meta_train.shape}"
    )

    print(
        f"Meta测试特征维度: "
        f"{X_meta_test.shape}"
    )

    if X_meta_train.shape[1] != 20:

        raise RuntimeError(
            f"Meta特征应该为20维，"
            f"当前为 "
            f"{X_meta_train.shape[1]}"
        )

    # --------------------------------------------------------
    # 元学习器
    #
    # 不再做 ADASYN：
    # 因为基模型 OOF 概率已经来自训练折内 ADASYN。
    #
    # 使用 class_weight="balanced"
    # --------------------------------------------------------

    meta_model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=5000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                )
            ),
        ]
    )

    meta_model.fit(
        X_meta_train,
        y_train
    )

    y_pred = meta_model.predict(
        X_meta_test
    )

    y_prob = (
        meta_model.predict_proba(
            X_meta_test
        )[:, 1]
    )

    metrics = calculate_metrics(
        y_test,
        y_pred,
        y_prob
    )

    print_metrics(
        "Stacking",
        metrics
    )

    # --------------------------------------------------------
    # 混淆矩阵
    # --------------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        confusion_matrix(
            y_test,
            y_pred
        )
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred,
            digits=4,
            zero_division=0
        )
    )

    # --------------------------------------------------------
    # 保存
    # --------------------------------------------------------

    joblib.dump(
        meta_model,
        MODEL_DIR
        / "Stacking_Meta.joblib"
    )

    prediction_df = pd.DataFrame(
        {
            "y_true":
                y_test,

            "y_pred":
                y_pred,

            "probability":
                y_prob,
        }
    )

    prediction_df.to_csv(
        REPORT_DIR
        / "stacking_test_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    return (
        meta_model,
        metrics,
        X_meta_train,
        X_meta_test,
    )


# ============================================================
# 10. 打印指标
# ============================================================

def print_metrics(
        name,
        metrics
):

    print(
        f"\n{name}"
    )

    print("-" * 50)

    print(
        f"Accuracy : "
        f"{metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{metrics['recall']:.4f}"
    )

    print(
        f"F1       : "
        f"{metrics['f1']:.4f}"
    )

    print(
        f"AUC      : "
        f"{metrics['auc']:.4f}"
    )


# ============================================================
# 11. 单模型测试性能
# ============================================================

def evaluate_base_models(
        trained_models,
        X_test,
        y_test
):

    results = {}

    print("\n")
    print("=" * 80)
    print("重新汇总三个基学习器 Test 性能")
    print("=" * 80)

    for model_name, model in trained_models.items():

        y_pred = model.predict(
            X_test
        )

        y_prob = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics = calculate_metrics(
            y_test,
            y_pred,
            y_prob
        )

        results[
            model_name
        ] = metrics

        print_metrics(
            model_name,
            metrics
        )

    return results


# ============================================================
# 12. 保存 OOF 数据
# ============================================================

def save_oof_data(
        train_df,
        oof_matrix,
        model_names
):

    output = train_df[
        [
            "sample_index",
            "equipment_id",
            "label",
            "target",
        ]
    ].copy()

    for idx, model_name in enumerate(
        model_names
    ):

        output[
            f"oof_{model_name}"
        ] = oof_matrix[
            :,
            idx
        ]

    output.to_csv(
        REPORT_DIR
        / "oof_predictions.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 13. 保存20维 Meta 数据
# ============================================================

def save_meta_features(
        train_df,
        test_df,
        X_meta_train,
        X_meta_test,
        model_names
):

    meta_columns = (
        FEATURE_COLUMNS
        +
        [
            f"prob_{name}"
            for name in model_names
        ]
    )

    # --------------------------------------------------------
    # Train Meta
    # --------------------------------------------------------

    train_meta_df = pd.DataFrame(
        X_meta_train,
        columns=meta_columns
    )

    train_meta_df[
        "target"
    ] = train_df[
        "target"
    ].to_numpy()

    train_meta_df[
        "equipment_id"
    ] = train_df[
        "equipment_id"
    ].to_numpy()

    train_meta_df.to_csv(
        REPORT_DIR
        / "stacking_train_meta.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Test Meta
    # --------------------------------------------------------

    test_meta_df = pd.DataFrame(
        X_meta_test,
        columns=meta_columns
    )

    test_meta_df[
        "target"
    ] = test_df[
        "target"
    ].to_numpy()

    test_meta_df[
        "equipment_id"
    ] = test_df[
        "equipment_id"
    ].to_numpy()

    test_meta_df.to_csv(
        REPORT_DIR
        / "stacking_test_meta.csv",
        index=False,
        encoding="utf-8-sig"
    )


# ============================================================
# 14. 最终模型比较
# ============================================================

def save_comparison(
        base_results,
        lr_metrics,
        stacking_metrics
):

    rows = []

    for name, metrics in base_results.items():

        rows.append(
            {
                "model":
                    name,
                **metrics,
            }
        )

    rows.append(
        {
            "model":
                "LR_Baseline",
            **lr_metrics,
        }
    )

    rows.append(
        {
            "model":
                "Stacking",
            **stacking_metrics,
        }
    )

    comparison = pd.DataFrame(
        rows
    )

    comparison = comparison[
        [
            "model",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
        ]
    ]

    comparison.to_csv(
        REPORT_DIR
        / "stacking_model_comparison.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\n")
    print("=" * 80)
    print("最终模型比较")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # 找出 Test F1 最好的模型
    # --------------------------------------------------------

    best_row = comparison.loc[
        comparison[
            "f1"
        ].idxmax()
    ]

    print("\n")
    print(
        f"Test F1 最佳模型: "
        f"{best_row['model']}"
    )

    print(
        f"Best F1: "
        f"{best_row['f1']:.4f}"
    )

    print(
        f"Best AUC: "
        f"{best_row['auc']:.4f}"
    )

    return comparison


# ============================================================
# 15. 主程序
# ============================================================

def main():

    warnings.filterwarnings(
        "ignore",
        category=UserWarning
    )

    # ========================================================
    # 数据
    # ========================================================

    (
        train_df,
        test_df,
        X_train,
        y_train,
        groups_train,
        X_test,
        y_test,
    ) = load_dataset()

    # ========================================================
    # 基模型定义
    # ========================================================

    models = build_base_models()

    # ========================================================
    # OOF概率
    # ========================================================

    (
        oof_matrix,
        model_names
    ) = generate_oof_predictions(
        models,
        X_train,
        y_train,
        groups_train
    )

    # 保存OOF
    save_oof_data(
        train_df,
        oof_matrix,
        model_names
    )

    # ========================================================
    # 完整训练三个基模型
    # ========================================================

    (
        test_probability_matrix,
        trained_models
    ) = train_full_base_models(
        models,
        X_train,
        y_train,
        X_test
    )

    # ========================================================
    # 基模型独立测试结果
    # ========================================================

    base_results = evaluate_base_models(
        trained_models,
        X_test,
        y_test
    )

    # ========================================================
    # LR baseline
    # ========================================================

    (
        lr_model,
        lr_metrics
    ) = train_lr_baseline(
        X_train,
        y_train,
        X_test,
        y_test
    )

    # ========================================================
    # Stacking
    # ========================================================

    (
        meta_model,
        stacking_metrics,
        X_meta_train,
        X_meta_test,
    ) = train_meta_model(
        X_train,
        y_train,
        oof_matrix,
        X_test,
        y_test,
        test_probability_matrix
    )

    # ========================================================
    # 保存Meta特征
    # ========================================================

    save_meta_features(
        train_df,
        test_df,
        X_meta_train,
        X_meta_test,
        model_names
    )

    # ========================================================
    # 最终比较
    # ========================================================

    save_comparison(
        base_results,
        lr_metrics,
        stacking_metrics
    )

    # ========================================================
    # 保存 Stacking 整体信息
    # ========================================================

    stacking_bundle = {
        "feature_columns":
            FEATURE_COLUMNS,

        "base_model_names":
            model_names,

        "base_models":
            trained_models,

        "meta_model":
            meta_model,
    }

    joblib.dump(
        stacking_bundle,
        MODEL_DIR
        / "Stacking_Bundle.joblib"
    )

    print("\n")
    print("=" * 80)
    print("OOF Stacking 全部完成")
    print("=" * 80)

    print(
        "\nStacking模型:"
    )

    print(
        MODEL_DIR
        / "Stacking_Bundle.joblib"
    )

    print(
        "\nOOF与评估报告:"
    )

    print(
        REPORT_DIR
    )


if __name__ == "__main__":
    main()