from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GroupShuffleSplit,
    StratifiedKFold,
)

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

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "features.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "paper_stacking_validation"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. MSM-17D
# ============================================================

FEATURE_NAMES = [
    f"f{i}"
    for i in range(1, 18)
]


# ============================================================
# 3. 标签
#
# 论文：
# 0 = 非润滑失效
# 1 = 润滑失效
# ============================================================

LABEL_MAP = {
    "非轴承润滑不良": 0,
    "轴承润滑不良": 1,
}


# ============================================================
# 4. 论文 11 个随机种子
# ============================================================

RANDOM_SEEDS = list(
    range(10)
) + [42]


# ============================================================
# 5. 基学习器
#
# 参数来自论文表 9
# ============================================================

def build_base_models(
    random_state,
):

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=random_state,
        n_jobs=-1,
    )

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
# 6. 元学习器
# ============================================================

def build_meta_model():

    return LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3000,
    )


# ============================================================
# 7. 读取 features.csv
# ============================================================

def load_dataset():

    print(
        "=" * 80
    )

    print(
        "读取论文 MSM-17D 工业数据"
    )

    print(
        "=" * 80
    )

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"找不到：{DATA_FILE}"
        )

    df = pd.read_csv(
        DATA_FILE
    )

    print(
        f"\n数据 shape：{df.shape}"
    )

    # --------------------------------------------------------
    # 检查 MSM 特征
    # --------------------------------------------------------

    missing = [
        f
        for f in FEATURE_NAMES
        if f not in df.columns
    ]

    if missing:

        raise ValueError(
            f"缺少 MSM 特征：{missing}"
        )

    # --------------------------------------------------------
    # 检查必要字段
    # --------------------------------------------------------

    for column in [
        "label",
        "equipment_id",
    ]:

        if column not in df.columns:

            raise ValueError(
                f"缺少字段：{column}"
            )

    # --------------------------------------------------------
    # X
    # --------------------------------------------------------

    X = (
        df[
            FEATURE_NAMES
        ]
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .to_numpy(
            dtype=float
        )
    )

    if not np.isfinite(X).all():

        raise ValueError(
            "f1~f17 中存在 NaN / inf"
        )

    # --------------------------------------------------------
    # y
    # --------------------------------------------------------

    labels = (
        df["label"]
        .astype(str)
        .str.strip()
    )

    unknown = set(
        labels.unique()
    ) - set(
        LABEL_MAP.keys()
    )

    if unknown:

        raise ValueError(
            f"发现未知标签：{unknown}"
        )

    y = (
        labels
        .map(
            LABEL_MAP
        )
        .astype(int)
        .to_numpy()
    )

    # --------------------------------------------------------
    # groups = equipment_id
    # --------------------------------------------------------

    groups = (
        df[
            "equipment_id"
        ]
        .astype(str)
        .str.strip()
        .to_numpy()
    )

    print(
        f"样本数：{len(y)}"
    )

    print(
        f"设备数："
        f"{len(np.unique(groups))}"
    )

    print(
        f"class 0："
        f"{np.sum(y == 0)}"
    )

    print(
        f"class 1："
        f"{np.sum(y == 1)}"
    )

    print(
        f"全数据润滑失效比例："
        f"{np.mean(y):.4f}"
    )

    return (
        df,
        X,
        y,
        groups,
    )


# ============================================================
# 8. 设备级互斥划分
#
# 论文：
# Test 样本数约为总样本的 30%
# 同时 Test 中正类比例尽量接近全数据比例
#
# 我们对每个随机种子生成多个候选 GroupShuffleSplit，
# 选择最接近论文目标的一个。
# ============================================================

def device_level_split(
    X,
    y,
    groups,
    random_state,
    target_test_size=0.30,
    n_candidates=500,
):

    target_positive_ratio = (
        float(
            np.mean(y)
        )
    )

    best_score = None
    best_split = None

    for offset in range(
        n_candidates
    ):

        candidate_seed = (
            random_state * 10000
            + offset
        )

        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=target_test_size,
            random_state=candidate_seed,
        )

        train_idx, test_idx = next(
            splitter.split(
                X,
                y,
                groups,
            )
        )

        y_train = y[
            train_idx
        ]

        y_test = y[
            test_idx
        ]

        # ----------------------------------------------------
        # Train/Test 都必须包含两个类别
        # ----------------------------------------------------

        if (
            len(
                np.unique(
                    y_train
                )
            )
            < 2
        ):

            continue

        if (
            len(
                np.unique(
                    y_test
                )
            )
            < 2
        ):

            continue

        actual_test_ratio = (
            len(test_idx)
            / len(y)
        )

        test_positive_ratio = (
            float(
                np.mean(
                    y_test
                )
            )
        )

        # ----------------------------------------------------
        # 与论文两个目标的差距
        #
        # 1. Test ≈ 30%
        # 2. 正类比例接近全数据
        # ----------------------------------------------------

        size_error = abs(
            actual_test_ratio
            - target_test_size
        )

        positive_error = abs(
            test_positive_ratio
            - target_positive_ratio
        )

        score = (
            size_error
            + positive_error
        )

        if (
            best_score is None
            or score < best_score
        ):

            best_score = score

            best_split = (
                train_idx,
                test_idx,
            )

    if best_split is None:

        raise RuntimeError(
            "无法找到有效的设备级互斥划分。"
        )

    train_idx, test_idx = (
        best_split
    )

    # --------------------------------------------------------
    # 再检查设备不能重叠
    # --------------------------------------------------------

    train_devices = set(
        groups[
            train_idx
        ]
    )

    test_devices = set(
        groups[
            test_idx
        ]
    )

    overlap = (
        train_devices
        & test_devices
    )

    if overlap:

        raise RuntimeError(
            "设备级划分失败："
            "Train/Test 存在设备重叠。"
        )

    return (
        train_idx,
        test_idx,
    )


# ============================================================
# 9. 生成 OOF
#
# 输入已经是：
# StandardScaler
# +
# ADASYN
#
# 之后在增强后的训练集上执行论文 5-fold OOF
# ============================================================

def generate_oof(
    X_train,
    y_train,
    random_state,
):

    n = len(
        y_train
    )

    oof = np.zeros(
        (
            n,
            3,
        ),
        dtype=float,
    )

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=random_state,
    )

    for fold, (
        train_idx,
        valid_idx,
    ) in enumerate(
        cv.split(
            X_train,
            y_train,
        ),
        start=1,
    ):

        print(
            f"    OOF Fold "
            f"{fold}/5"
        )

        models = (
            build_base_models(
                random_state
            )
        )

        X_fold_train = (
            X_train[
                train_idx
            ]
        )

        y_fold_train = (
            y_train[
                train_idx
            ]
        )

        X_valid = (
            X_train[
                valid_idx
            ]
        )

        # RF
        models["rf"].fit(
            X_fold_train,
            y_fold_train,
        )

        oof[
            valid_idx,
            0
        ] = models[
            "rf"
        ].predict_proba(
            X_valid
        )[:, 1]

        # XGB
        models["xgb"].fit(
            X_fold_train,
            y_fold_train,
        )

        oof[
            valid_idx,
            1
        ] = models[
            "xgb"
        ].predict_proba(
            X_valid
        )[:, 1]

        # SVM
        models["svm"].fit(
            X_fold_train,
            y_fold_train,
        )

        oof[
            valid_idx,
            2
        ] = models[
            "svm"
        ].predict_proba(
            X_valid
        )[:, 1]

    return oof


# ============================================================
# 10. 单个随机种子验证
# ============================================================

def run_one_seed(
    df,
    X,
    y,
    groups,
    random_state,
):

    print(
        "\n"
        + "#" * 80
    )

    print(
        f"Random Seed = "
        f"{random_state}"
    )

    print(
        "#" * 80
    )

    # ========================================================
    # A. 设备级互斥划分
    # ========================================================

    (
        train_idx,
        test_idx,
    ) = device_level_split(
        X=X,
        y=y,
        groups=groups,
        random_state=random_state,
    )

    X_train_raw = X[
        train_idx
    ]

    X_test_raw = X[
        test_idx
    ]

    y_train = y[
        train_idx
    ]

    y_test = y[
        test_idx
    ]

    train_groups = groups[
        train_idx
    ]

    test_groups = groups[
        test_idx
    ]

    print(
        "\n设备级互斥划分"
    )

    print(
        f"Train 样本："
        f"{len(train_idx)}"
    )

    print(
        f"Test 样本："
        f"{len(test_idx)}"
    )

    print(
        f"Train 设备："
        f"{len(np.unique(train_groups))}"
    )

    print(
        f"Test 设备："
        f"{len(np.unique(test_groups))}"
    )

    print(
        f"Test 比例："
        f"{len(test_idx) / len(y):.4f}"
    )

    print(
        f"Test 润滑失效比例："
        f"{np.mean(y_test):.4f}"
    )

    # ========================================================
    # B. StandardScaler
    #
    # 只 fit Train
    # ========================================================

    scaler = StandardScaler()

    X_train_scaled = (
        scaler.fit_transform(
            X_train_raw
        )
    )

    X_test_scaled = (
        scaler.transform(
            X_test_raw
        )
    )

    # ========================================================
    # C. ADASYN
    #
    # 只作用 Train
    # ========================================================

    n_min = int(
        np.sum(
            y_train == 1
        )
    )

    k_neighbors = min(
        5,
        n_min - 1,
    )

    if k_neighbors < 1:

        raise RuntimeError(
            "Train 少数类不足，"
            "无法执行 ADASYN。"
        )

    print(
        "\nADASYN 前"
    )

    print(
        f"  class 0 = "
        f"{np.sum(y_train == 0)}"
    )

    print(
        f"  class 1 = "
        f"{np.sum(y_train == 1)}"
    )

    adasyn = ADASYN(
        sampling_strategy="minority",
        n_neighbors=k_neighbors,
        random_state=random_state,
    )

    (
        X_train_aug,
        y_train_aug,
    ) = adasyn.fit_resample(
        X_train_scaled,
        y_train,
    )

    print(
        "\nADASYN 后"
    )

    print(
        f"  class 0 = "
        f"{np.sum(y_train_aug == 0)}"
    )

    print(
        f"  class 1 = "
        f"{np.sum(y_train_aug == 1)}"
    )

    # ========================================================
    # D. Stacking OOF
    # ========================================================

    print(
        "\n生成 Stacking OOF..."
    )

    oof = generate_oof(
        X_train=X_train_aug,
        y_train=y_train_aug,
        random_state=random_state,
    )

    # ========================================================
    # E. Meta 输入
    #
    # 3概率 + 17 MSM
    # ========================================================

    X_meta_train = np.hstack(
        [
            oof,
            X_train_aug,
        ]
    )

    if (
        X_meta_train.shape[1]
        != 20
    ):

        raise RuntimeError(
            "Meta 特征不是20维"
        )

    # ========================================================
    # F. Meta Learner
    # ========================================================

    meta = build_meta_model()

    meta.fit(
        X_meta_train,
        y_train_aug,
    )

    # ========================================================
    # G. 最终三个基模型
    #
    # 用完整增强 Train 重新训练
    # ========================================================

    final_models = (
        build_base_models(
            random_state
        )
    )

    final_models[
        "rf"
    ].fit(
        X_train_aug,
        y_train_aug,
    )

    final_models[
        "xgb"
    ].fit(
        X_train_aug,
        y_train_aug,
    )

    final_models[
        "svm"
    ].fit(
        X_train_aug,
        y_train_aug,
    )

    # ========================================================
    # H. Test 基模型概率
    #
    # 注意：
    # Test 只 transform
    # 不做 ADASYN
    # ========================================================

    p_rf = final_models[
        "rf"
    ].predict_proba(
        X_test_scaled
    )[:, 1]

    p_xgb = final_models[
        "xgb"
    ].predict_proba(
        X_test_scaled
    )[:, 1]

    p_svm = final_models[
        "svm"
    ].predict_proba(
        X_test_scaled
    )[:, 1]

    # ========================================================
    # I. Test Meta 输入
    # ========================================================

    X_meta_test = np.hstack(
        [
            p_rf.reshape(
                -1,
                1,
            ),

            p_xgb.reshape(
                -1,
                1,
            ),

            p_svm.reshape(
                -1,
                1,
            ),

            X_test_scaled,
        ]
    )

    # ========================================================
    # J. Stacking 最终概率
    # ========================================================

    y_probability = (
        meta.predict_proba(
            X_meta_test
        )[:, 1]
    )

    y_pred = (
        y_probability
        >= 0.5
    ).astype(int)

    # ========================================================
    # K. 指标
    # ========================================================

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0,
    )

    auc = roc_auc_score(
        y_test,
        y_probability,
    )

    cm = confusion_matrix(
        y_test,
        y_pred,
        labels=[
            0,
            1,
        ],
    )

    tn, fp, fn, tp = (
        cm.ravel()
    )

    print(
        "\n验证结果"
    )

    print(
        "-" * 60
    )

    print(
        f"Accuracy  : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"F1        : "
        f"{f1:.4f}"
    )

    print(
        f"AUC       : "
        f"{auc:.4f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        f"TN={tn}"
    )

    print(
        f"FP={fp}"
    )

    print(
        f"FN={fn}"
    )

    print(
        f"TP={tp}"
    )

    # ========================================================
    # L. 保存每个 Test 样本结果
    # ========================================================

    prediction_df = (
        df.iloc[
            test_idx
        ]
        .copy()
        .reset_index(
            drop=True
        )
    )

    prediction_df[
        "y_true"
    ] = y_test

    prediction_df[
        "p_rf"
    ] = p_rf

    prediction_df[
        "p_xgb"
    ] = p_xgb

    prediction_df[
        "p_svm"
    ] = p_svm

    prediction_df[
        "stacking_probability"
    ] = y_probability

    prediction_df[
        "y_pred"
    ] = y_pred

    prediction_path = (
        REPORT_DIR
        / (
            f"seed_"
            f"{random_state:02d}"
            f"_predictions.csv"
        )
    )

    prediction_df.to_csv(
        prediction_path,
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "seed": random_state,

        "train_samples": (
            len(train_idx)
        ),

        "test_samples": (
            len(test_idx)
        ),

        "train_devices": (
            len(
                np.unique(
                    train_groups
                )
            )
        ),

        "test_devices": (
            len(
                np.unique(
                    test_groups
                )
            )
        ),

        "test_ratio": (
            len(test_idx)
            / len(y)
        ),

        "test_positive_ratio": (
            float(
                np.mean(
                    y_test
                )
            )
        ),

        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


# ============================================================
# 11. 11 次论文重复实验
# ============================================================

def validate_paper_stacking():

    (
        df,
        X,
        y,
        groups,
    ) = load_dataset()

    results = []

    for seed in RANDOM_SEEDS:

        result = run_one_seed(
            df=df,
            X=X,
            y=y,
            groups=groups,
            random_state=seed,
        )

        results.append(
            result
        )

    # ========================================================
    # 保存 11 次结果
    # ========================================================

    results_df = pd.DataFrame(
        results
    )

    csv_path = (
        REPORT_DIR
        / "validation_11_seeds.csv"
    )

    results_df.to_csv(
        csv_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ========================================================
    # 均值 ± 标准差
    # ========================================================

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]

    summary = {}

    print(
        "\n\n"
        + "=" * 80
    )

    print(
        "论文 11 组重复实验最终结果"
    )

    print(
        "=" * 80
    )

    for metric in metric_names:

        values = (
            results_df[
                metric
            ]
            .to_numpy(
                dtype=float
            )
        )

        mean_value = float(
            np.mean(
                values
            )
        )

        std_value = float(
            np.std(
                values,
                ddof=1,
            )
        )

        summary[
            metric
        ] = {
            "mean": mean_value,
            "std": std_value,
        }

        print(
            f"{metric:10s}: "
            f"{mean_value:.4f} "
            f"± "
            f"{std_value:.4f}"
        )

    # ========================================================
    # 保存 summary
    # ========================================================

    summary_path = (
        REPORT_DIR
        / "validation_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\n结果目录："
    )

    print(
        REPORT_DIR
    )


# ============================================================
# 12. main
# ============================================================

if __name__ == "__main__":

    validate_paper_stacking()