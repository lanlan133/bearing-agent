from pathlib import Path

import numpy as np
import pandas as pd

from diagnosis_agent import run_diagnosis_agent
from data_loader import load_mat_dataset, matlab_string


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "features"
    / "test.csv"
)

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "agent_validation_review"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    REPORT_DIR
    / "agent_review_validation.csv"
)

REVIEW_FILE = (
    REPORT_DIR
    / "review_samples.csv"
)

AUTO_ERROR_FILE = (
    REPORT_DIR
    / "remaining_auto_errors.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "agent_review_summary.txt"
)


# ============================================================
# 2. 安全转换
# ============================================================

def safe_float(
        value,
        default=np.nan
):
    try:
        return float(
            np.asarray(
                value
            ).squeeze()
        )
    except Exception:
        return default


# ============================================================
# 3. MAT 行解析
# ============================================================

def parse_mat_row(
        row
):

    measurement_point = matlab_string(
        row[0]
    )

    speed = safe_float(
        row[1]
    )

    sampling_rate = safe_float(
        row[2]
    )

    signal_data = np.asarray(
        row[3],
        dtype=float
    ).squeeze()

    if signal_data.ndim != 1:
        signal_data = signal_data.ravel()

    label = matlab_string(
        row[4]
    )

    equipment_id = matlab_string(
        row[5]
    )

    equipment_name = matlab_string(
        row[6]
    )

    return {
        "measurement_point":
            measurement_point,

        "speed":
            speed,

        "sampling_rate":
            sampling_rate,

        "signal":
            signal_data,

        "label":
            label,

        "equipment_id":
            equipment_id,

        "equipment_name":
            equipment_name,
    }


# ============================================================
# 4. 标签映射
# ============================================================

def label_to_target(
        label
):

    if label == "轴承润滑不良":
        return 1

    if label == "非轴承润滑不良":
        return 0

    raise ValueError(
        f"未知标签: {label}"
    )


# ============================================================
# 5. 原始二分类错误类型
# ============================================================

def get_binary_error_type(
        y_true,
        y_pred
):

    if (
        y_true == 0
        and y_pred == 0
    ):
        return "TN"

    if (
        y_true == 0
        and y_pred == 1
    ):
        return "FP"

    if (
        y_true == 1
        and y_pred == 0
    ):
        return "FN"

    if (
        y_true == 1
        and y_pred == 1
    ):
        return "TP"

    return "UNKNOWN"


# ============================================================
# 6. 三态 Agent 验证
# ============================================================

def validate_agent():

    print("=" * 85)
    print("Diagnosis Agent 三态 Review Policy 独立测试验证")
    print("=" * 85)

    # --------------------------------------------------------
    # 测试集
    # --------------------------------------------------------

    test_df = pd.read_csv(
        TEST_FILE
    )

    print(
        f"\n测试样本数: "
        f"{len(test_df)}"
    )

    # --------------------------------------------------------
    # MAT只加载一次
    # --------------------------------------------------------

    data = load_mat_dataset(
        MAT_FILE
    )

    print(
        f"MAT shape: "
        f"{data.shape}"
    )

    results = []

    processing_errors = []

    total = len(
        test_df
    )

    # ========================================================
    # 批量执行 Agent
    # ========================================================

    for i, row in test_df.iterrows():

        sample_index = int(
            row[
                "sample_index"
            ]
        )

        try:

            sample = parse_mat_row(
                data[
                    sample_index
                ]
            )

            report = run_diagnosis_agent(
                signal_data=
                    sample[
                        "signal"
                    ],

                sampling_rate=
                    sample[
                        "sampling_rate"
                    ],

                top_n=5
            )

            # ------------------------------------------------
            # 真实标签
            # ------------------------------------------------

            y_true = int(
                row[
                    "target"
                ]
            )

            # ------------------------------------------------
            # XGBoost原始二分类结果
            # ------------------------------------------------

            model_prediction = int(
                report[
                    "model_prediction"
                ]
            )

            probability = float(
                report[
                    "risk_probability"
                ]
            )

            binary_error_type = (
                get_binary_error_type(
                    y_true,
                    model_prediction
                )
            )

            model_correct = bool(
                y_true
                ==
                model_prediction
            )

            # ------------------------------------------------
            # Agent三态决策
            # ------------------------------------------------

            decision = report[
                "decision"
            ]

            decision_status = report[
                "decision_status"
            ]

            needs_review = bool(
                report[
                    "needs_review"
                ]
            )

            # ------------------------------------------------
            # 自动诊断正确性
            # review样本不算对，也不算错
            # ------------------------------------------------

            auto_correct = None

            auto_pred = None

            if decision_status == "auto_normal":

                auto_pred = 0

                auto_correct = bool(
                    y_true == 0
                )

            elif decision_status == "auto_fault":

                auto_pred = 1

                auto_correct = bool(
                    y_true == 1
                )

            elif decision_status == "review":

                auto_pred = None
                auto_correct = None

            else:

                raise ValueError(
                    f"未知 decision_status: "
                    f"{decision_status}"
                )

            # ------------------------------------------------
            # 原始模型错误是否被 Review 拦截
            # ------------------------------------------------

            error_intercepted = bool(
                (
                    not model_correct
                )
                and
                needs_review
            )

            # ------------------------------------------------
            # 保存
            # ------------------------------------------------

            result_row = {

                "sample_index":
                    sample_index,

                "equipment_id":
                    sample[
                        "equipment_id"
                    ],

                "equipment_name":
                    sample[
                        "equipment_name"
                    ],

                "measurement_point":
                    sample[
                        "measurement_point"
                    ],

                "sampling_rate":
                    sample[
                        "sampling_rate"
                    ],

                "true_label":
                    sample[
                        "label"
                    ],

                "y_true":
                    y_true,

                # 原模型
                "model_prediction":
                    model_prediction,

                "model_diagnosis":
                    report[
                        "model_diagnosis"
                    ],

                "risk_probability":
                    probability,

                "model_correct":
                    model_correct,

                "binary_error_type":
                    binary_error_type,

                # Agent三态
                "agent_decision":
                    decision,

                "decision_status":
                    decision_status,

                "needs_review":
                    needs_review,

                "review_reason":
                    report[
                        "review_reason"
                    ],

                # 自动区
                "auto_pred":
                    auto_pred,

                "auto_correct":
                    auto_correct,

                # 是否成功拦截原始错误
                "error_intercepted":
                    error_intercepted,

                "domain_validated":
                    report[
                        "domain"
                    ][
                        "validated"
                    ],

                "roi_active":
                    report[
                        "roi"
                    ][
                        "active"
                    ],

                "gaussian_success":
                    report[
                        "gaussian"
                    ][
                        "success"
                    ],
            }

            # ------------------------------------------------
            # Top-5 SHAP
            # ------------------------------------------------

            top_features = report[
                "top_evidence"
            ]

            for rank in range(
                1,
                6
            ):

                if rank <= len(
                    top_features
                ):

                    item = top_features[
                        rank - 1
                    ]

                    result_row[
                        f"top{rank}_feature"
                    ] = item[
                        "feature"
                    ]

                    result_row[
                        f"top{rank}_name"
                    ] = item[
                        "feature_name"
                    ]

                    result_row[
                        f"top{rank}_shap"
                    ] = item[
                        "shap_value"
                    ]

                else:

                    result_row[
                        f"top{rank}_feature"
                    ] = None

                    result_row[
                        f"top{rank}_name"
                    ] = None

                    result_row[
                        f"top{rank}_shap"
                    ] = np.nan

            results.append(
                result_row
            )

            # ------------------------------------------------
            # 进度输出
            # ------------------------------------------------

            current = i + 1

            if (
                current == 1
                or current % 20 == 0
                or current == total
            ):

                print(
                    f"[{current:3d}/{total}] "
                    f"sample={sample_index} "
                    f"| true={y_true} "
                    f"| model={model_prediction} "
                    f"| prob={probability:.4f} "
                    f"| agent={decision_status}"
                )

        except Exception as exc:

            print(
                f"[ERROR] "
                f"sample_index="
                f"{sample_index}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            processing_errors.append(
                {
                    "sample_index":
                        sample_index,

                    "error_type":
                        type(
                            exc
                        ).__name__,

                    "error_message":
                        str(
                            exc
                        ),
                }
            )

    # ========================================================
    # 7. DataFrame
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    result_df.to_csv(
        OUTPUT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 8. 三态数量
    # ========================================================

    auto_normal_df = result_df[
        result_df[
            "decision_status"
        ] == "auto_normal"
    ].copy()

    auto_fault_df = result_df[
        result_df[
            "decision_status"
        ] == "auto_fault"
    ].copy()

    review_df = result_df[
        result_df[
            "decision_status"
        ] == "review"
    ].copy()

    auto_df = result_df[
        result_df[
            "decision_status"
        ].isin(
            [
                "auto_normal",
                "auto_fault",
            ]
        )
    ].copy()

    # ========================================================
    # 9. 原始模型错误
    # ========================================================

    model_error_df = result_df[
        result_df[
            "model_correct"
        ] == False
    ].copy()

    model_fn_df = result_df[
        result_df[
            "binary_error_type"
        ] == "FN"
    ].copy()

    model_fp_df = result_df[
        result_df[
            "binary_error_type"
        ] == "FP"
    ].copy()

    # --------------------------------------------------------
    # 被Review拦截
    # --------------------------------------------------------

    intercepted_error_df = (
        model_error_df[
            model_error_df[
                "needs_review"
            ] == True
        ].copy()
    )

    intercepted_fn_df = (
        model_fn_df[
            model_fn_df[
                "needs_review"
            ] == True
        ].copy()
    )

    intercepted_fp_df = (
        model_fp_df[
            model_fp_df[
                "needs_review"
            ] == True
        ].copy()
    )

    # ========================================================
    # 10. 自动诊断错误
    # ========================================================

    auto_error_df = auto_df[
        auto_df[
            "auto_correct"
        ] == False
    ].copy()

    remaining_auto_fn_df = auto_df[
        (
            auto_df[
                "y_true"
            ] == 1
        )
        &
        (
            auto_df[
                "auto_pred"
            ] == 0
        )
    ].copy()

    remaining_auto_fp_df = auto_df[
        (
            auto_df[
                "y_true"
            ] == 0
        )
        &
        (
            auto_df[
                "auto_pred"
            ] == 1
        )
    ].copy()

    # ========================================================
    # 11. 指标
    # ========================================================

    n = len(
        result_df
    )

    auto_count = len(
        auto_df
    )

    review_count = len(
        review_df
    )

    auto_correct_count = int(
        (
            auto_df[
                "auto_correct"
            ] == True
        ).sum()
    )

    auto_accuracy = (
        auto_correct_count
        /
        auto_count
        if auto_count > 0
        else 0
    )

    review_rate = (
        review_count
        /
        n
        if n > 0
        else 0
    )

    auto_rate = (
        auto_count
        /
        n
        if n > 0
        else 0
    )

    model_error_count = len(
        model_error_df
    )

    model_fn_count = len(
        model_fn_df
    )

    model_fp_count = len(
        model_fp_df
    )

    intercepted_error_count = len(
        intercepted_error_df
    )

    intercepted_fn_count = len(
        intercepted_fn_df
    )

    intercepted_fp_count = len(
        intercepted_fp_df
    )

    error_coverage = (
        intercepted_error_count
        /
        model_error_count
        if model_error_count > 0
        else 0
    )

    fn_coverage = (
        intercepted_fn_count
        /
        model_fn_count
        if model_fn_count > 0
        else 0
    )

    fp_coverage = (
        intercepted_fp_count
        /
        model_fp_count
        if model_fp_count > 0
        else 0
    )

    # ========================================================
    # 12. 输出结果
    # ========================================================

    print("\n")
    print("=" * 85)
    print("三态 Diagnosis Agent 验证结果")
    print("=" * 85)

    print(
        f"成功处理样本: "
        f"{n}"
    )

    print(
        f"程序失败样本: "
        f"{len(processing_errors)}"
    )

    print("\n三态分布")

    print("-" * 85)

    print(
        f"自动正常: "
        f"{len(auto_normal_df)}"
    )

    print(
        f"自动故障: "
        f"{len(auto_fault_df)}"
    )

    print(
        f"建议复核: "
        f"{review_count}"
    )

    print(
        f"自动诊断率: "
        f"{auto_rate:.2%}"
    )

    print(
        f"复核率: "
        f"{review_rate:.2%}"
    )

    print("\n自动诊断性能")

    print("-" * 85)

    print(
        f"自动诊断样本数: "
        f"{auto_count}"
    )

    print(
        f"自动诊断正确: "
        f"{auto_correct_count}"
    )

    print(
        f"自动诊断错误: "
        f"{len(auto_error_df)}"
    )

    print(
        f"自动诊断准确率: "
        f"{auto_accuracy:.2%}"
    )

    print("\n原始 XGBoost 错误拦截")

    print("-" * 85)

    print(
        f"原始模型错误总数: "
        f"{model_error_count}"
    )

    print(
        f"其中 FN: "
        f"{model_fn_count}"
    )

    print(
        f"其中 FP: "
        f"{model_fp_count}"
    )

    print(
        f"被Review拦截错误数: "
        f"{intercepted_error_count}"
    )

    print(
        f"总错误覆盖率: "
        f"{error_coverage:.2%}"
    )

    print(
        f"FN拦截数: "
        f"{intercepted_fn_count}"
    )

    print(
        f"FN覆盖率: "
        f"{fn_coverage:.2%}"
    )

    print(
        f"FP拦截数: "
        f"{intercepted_fp_count}"
    )

    print(
        f"FP覆盖率: "
        f"{fp_coverage:.2%}"
    )

    print("\n剩余自动诊断错误")

    print("-" * 85)

    print(
        f"剩余自动FN: "
        f"{len(remaining_auto_fn_df)}"
    )

    print(
        f"剩余自动FP: "
        f"{len(remaining_auto_fp_df)}"
    )

    # ========================================================
    # 13. 保存专题文件
    # ========================================================

    review_df.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    auto_error_df.to_csv(
        AUTO_ERROR_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    intercepted_error_df.to_csv(
        REPORT_DIR
        / "intercepted_model_errors.csv",
        index=False,
        encoding="utf-8-sig"
    )

    remaining_auto_fn_df.to_csv(
        REPORT_DIR
        / "remaining_auto_fn.csv",
        index=False,
        encoding="utf-8-sig"
    )

    remaining_auto_fp_df.to_csv(
        REPORT_DIR
        / "remaining_auto_fp.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 14. 摘要
    # ========================================================

    summary_lines = [

        "Diagnosis Agent Three-State Validation",
        "=" * 70,

        f"Total samples: {n}",
        f"Processing errors: {len(processing_errors)}",

        "",

        f"Auto normal: {len(auto_normal_df)}",
        f"Auto fault: {len(auto_fault_df)}",
        f"Review: {review_count}",

        f"Auto rate: {auto_rate:.6f}",
        f"Review rate: {review_rate:.6f}",

        "",

        f"Auto samples: {auto_count}",
        f"Auto correct: {auto_correct_count}",
        f"Auto errors: {len(auto_error_df)}",
        f"Auto accuracy: {auto_accuracy:.6f}",

        "",

        f"Base model errors: {model_error_count}",
        f"Base model FN: {model_fn_count}",
        f"Base model FP: {model_fp_count}",

        f"Intercepted errors: {intercepted_error_count}",
        f"Error coverage: {error_coverage:.6f}",

        f"Intercepted FN: {intercepted_fn_count}",
        f"FN coverage: {fn_coverage:.6f}",

        f"Intercepted FP: {intercepted_fp_count}",
        f"FP coverage: {fp_coverage:.6f}",

        "",

        f"Remaining auto FN: {len(remaining_auto_fn_df)}",
        f"Remaining auto FP: {len(remaining_auto_fp_df)}",
    ]

    SUMMARY_FILE.write_text(
        "\n".join(
            summary_lines
        ),
        encoding="utf-8"
    )

    print("\n报告目录:")

    print(
        REPORT_DIR
    )

    print("\n")
    print("=" * 85)
    print("三态 Agent 独立测试验证完成")
    print("=" * 85)


# ============================================================
# 15. 主程序
# ============================================================

if __name__ == "__main__":

    validate_agent()