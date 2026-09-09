import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

if str(SRC_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(SRC_DIR)
    )


from stacking_predictor import (
    predict_paper_stacking,
)


# ============================================================
# 读取 features.csv
# ============================================================

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "features.csv"
)

df = pd.read_csv(
    DATA_FILE
)


# ============================================================
# 测试第一个样本
# ============================================================

sample_index = 0

row = df.iloc[
    sample_index
]


features = {
    f"f{i}": float(
        row[
            f"f{i}"
        ]
    )
    for i in range(
        1,
        18
    )
}


# ============================================================
# Paper Stacking
# ============================================================

result = (
    predict_paper_stacking(
        features
    )
)


# ============================================================
# 输出
# ============================================================

print(
    "=" * 80
)

print(
    f"Sample Index: "
    f"{sample_index}"
)

print(
    f"真实标签: "
    f"{row['label']}"
)

print(
    "-" * 80
)

print(
    "Level-0:"
)

print(
    f"RF       : "
    f"{result['rf_probability']:.4f}"
)

print(
    f"XGBoost  : "
    f"{result['xgb_probability']:.4f}"
)

print(
    f"SVM      : "
    f"{result['svm_probability']:.4f}"
)

print(
    "-" * 80
)

print(
    f"Stacking : "
    f"{result['final_probability']:.4f}"
)

print(
    f"最终诊断 : "
    f"{result['diagnosis']}"
)

print(
    "=" * 80
)