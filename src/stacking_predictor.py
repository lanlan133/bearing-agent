from pathlib import Path

import joblib
import numpy as np


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "Paper_Stacking_Bundle.joblib"
)


# ============================================================
# 2. 加载论文模型
# ============================================================

if not MODEL_FILE.exists():

    raise FileNotFoundError(
        "找不到论文 Stacking 模型：\n"
        f"{MODEL_FILE}\n\n"
        "请先运行 train_paper_stacking.py。"
    )


MODEL_BUNDLE = joblib.load(
    MODEL_FILE
)


# ============================================================
# 3. 取出模型组件
# ============================================================

SCALER = MODEL_BUNDLE[
    "scaler"
]

RF_MODEL = MODEL_BUNDLE[
    "rf"
]

XGB_MODEL = MODEL_BUNDLE[
    "xgb"
]

SVM_MODEL = MODEL_BUNDLE[
    "svm"
]

META_MODEL = MODEL_BUNDLE[
    "meta"
]

FEATURE_NAMES = MODEL_BUNDLE.get(
    "feature_names",
    [
        f"f{i}"
        for i in range(1, 18)
    ],
)

THRESHOLD = float(
    MODEL_BUNDLE.get(
        "threshold",
        0.5,
    )
)


# ============================================================
# 4. 将输入转换成 17D numpy
# ============================================================

def prepare_features(
    features,
):

    # --------------------------------------------------------
    # 支持 dict：
    #
    # {
    #     "f1": ...,
    #     ...
    #     "f17": ...
    # }
    # --------------------------------------------------------

    if isinstance(
        features,
        dict,
    ):

        missing = [
            name
            for name in FEATURE_NAMES
            if name not in features
        ]

        if missing:

            raise ValueError(
                "缺少 MSM 特征："
                f"{missing}"
            )

        values = [
            features[name]
            for name in FEATURE_NAMES
        ]

        X = np.asarray(
            values,
            dtype=float,
        )

    else:

        X = np.asarray(
            features,
            dtype=float,
        )

    # --------------------------------------------------------
    # 拉平成 17D
    # --------------------------------------------------------

    X = X.reshape(
        -1
    )

    if len(X) != 17:

        raise ValueError(
            "Paper Stacking 模型要求 "
            "17 维 MSM 特征，"
            f"当前输入维度为 {len(X)}。"
        )

    if not np.isfinite(
        X
    ).all():

        raise ValueError(
            "MSM 特征存在 NaN 或 Inf。"
        )

    return X.reshape(
        1,
        -1,
    )


# ============================================================
# 5. 论文 Stacking 推理
# ============================================================

def predict_paper_stacking(
    features,
):

    # ========================================================
    # A. 17D
    # ========================================================

    X = prepare_features(
        features
    )

    # ========================================================
    # B. 标准化
    #
    # 推理阶段绝对不能重新 fit
    # ========================================================

    X_scaled = SCALER.transform(
        X
    )

    # ========================================================
    # C. Level-0 三个模型
    # ========================================================

    p_rf = float(
        RF_MODEL.predict_proba(
            X_scaled
        )[0, 1]
    )

    p_xgb = float(
        XGB_MODEL.predict_proba(
            X_scaled
        )[0, 1]
    )

    p_svm = float(
        SVM_MODEL.predict_proba(
            X_scaled
        )[0, 1]
    )

    # ========================================================
    # D. 20D Meta 输入
    #
    # [p_RF, p_XGB, p_SVM, f1...f17]
    # ========================================================

    probability_features = np.array(
        [
            [
                p_rf,
                p_xgb,
                p_svm,
            ]
        ],
        dtype=float,
    )

    X_meta = np.hstack(
        [
            probability_features,
            X_scaled,
        ]
    )

    if X_meta.shape != (
        1,
        20,
    ):

        raise RuntimeError(
            "Meta 输入维度异常："
            f"{X_meta.shape}"
        )

    # ========================================================
    # E. Level-1 Logistic Regression
    # ========================================================

    final_probability = float(
        META_MODEL.predict_proba(
            X_meta
        )[0, 1]
    )

    # ========================================================
    # F. 最终二分类
    # ========================================================

    final_class = int(
        final_probability
        >= THRESHOLD
    )

    if final_class == 1:

        diagnosis = (
            "轴承润滑不良"
        )

    else:

        diagnosis = (
            "非轴承润滑不良"
        )

    # ========================================================
    # G. 三模型一致程度
    # ========================================================

    base_probabilities = np.array(
        [
            p_rf,
            p_xgb,
            p_svm,
        ],
        dtype=float,
    )

    base_mean = float(
        np.mean(
            base_probabilities
        )
    )

    base_std = float(
        np.std(
            base_probabilities
        )
    )

    # ========================================================
    # H. 返回结构
    # ========================================================

    return {

        "model_name": (
            "MSM-ADASYN-Stacking"
        ),

        "model_version": (
            MODEL_BUNDLE.get(
                "model_version",
                "paper_v1",
            )
        ),

        # ----------------------------------------------------
        # 最终结果
        # ----------------------------------------------------

        "diagnosis": (
            diagnosis
        ),

        "predicted_class": (
            final_class
        ),

        "final_probability": (
            final_probability
        ),

        "risk_probability": (
            final_probability
        ),

        "threshold": (
            THRESHOLD
        ),

        # ----------------------------------------------------
        # 三个 Level-0
        # ----------------------------------------------------

        "base_probabilities": {

            "rf": p_rf,

            "xgb": p_xgb,

            "svm": p_svm,
        },

        "rf_probability": (
            p_rf
        ),

        "xgb_probability": (
            p_xgb
        ),

        "svm_probability": (
            p_svm
        ),

        # ----------------------------------------------------
        # 模型一致性
        # ----------------------------------------------------

        "base_probability_mean": (
            base_mean
        ),

        "base_probability_std": (
            base_std
        ),

        # ----------------------------------------------------
        # 输入
        # ----------------------------------------------------

        "feature_names": (
            FEATURE_NAMES
        ),

        "features_raw": (
            X.reshape(-1).tolist()
        ),

        "features_scaled": (
            X_scaled.reshape(-1).tolist()
        ),

        "meta_features": (
            X_meta.reshape(-1).tolist()
        ),
    }


# ============================================================
# 6. 简单测试
# ============================================================

if __name__ == "__main__":

    print(
        "=" * 80
    )

    print(
        "Paper Stacking Predictor"
    )

    print(
        "=" * 80
    )

    print(
        f"模型：{MODEL_FILE}"
    )

    print(
        f"特征：{FEATURE_NAMES}"
    )

    print(
        f"阈值：{THRESHOLD}"
    )

    print(
        "\n模型加载成功。"
    )