from pathlib import Path
import json

import numpy as np
import pandas as pd

from data_loader import (
    load_mat_dataset,
    matlab_string,
)

from diagnosis_agent import (
    run_diagnosis_agent,
)


# ============================================================
# 1. 路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "batch_diagnosis"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULT_FILE = (
    REPORT_DIR
    / "batch_results.csv"
)

SUMMARY_JSON = (
    REPORT_DIR
    / "batch_summary.json"
)

HIGH_RISK_FILE = (
    REPORT_DIR
    / "high_risk_samples.csv"
)

REVIEW_FILE = (
    REPORT_DIR
    / "review_samples.csv"
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

    signal = np.asarray(
        row[3],
        dtype=float
    ).squeeze()

    if signal.ndim != 1:
        signal = signal.ravel()

    return {
        "measurement_point":
            matlab_string(
                row[0]
            ),

        "speed":
            safe_float(
                row[1]
            ),

        "sampling_rate":
            safe_float(
                row[2]
            ),

        "signal":
            signal,

        "label":
            matlab_string(
                row[4]
            ),

        "equipment_id":
            matlab_string(
                row[5]
            ),

        "equipment_name":
            matlab_string(
                row[6]
            ),

        "timestamp":
            matlab_string(
                row[8]
            ),
    }


# ============================================================
# 4. 单条结果扁平化
# ============================================================

def flatten_result(
        sample_index,
        sample,
        report
):

    row = {
        "sample_index":
            int(
                sample_index
            ),

        "equipment_id":
            str(
                sample[
                    "equipment_id"
                ]
            ),

        "equipment_name":
            str(
                sample[
                    "equipment_name"
                ]
            ),

        "measurement_point":
            str(
                sample[
                    "measurement_point"
                ]
            ),

        "sampling_rate":
            float(
                sample[
                    "sampling_rate"
                ]
            ),

        "speed":
            float(
                sample[
                    "speed"
                ]
            ),

        "timestamp":
            str(
                sample[
                    "timestamp"
                ]
            ),

        "true_label":
            str(
                sample[
                    "label"
                ]
            ),

        "risk_probability":
            float(
                report[
                    "risk_probability"
                ]
            ),

        "risk_level":
            str(
                report[
                    "risk_level"
                ]
            ),

        "model_diagnosis":
            str(
                report[
                    "model_diagnosis"
                ]
            ),

        "agent_decision":
            str(
                report[
                    "decision"
                ]
            ),

        "decision_status":
            str(
                report[
                    "decision_status"
                ]
            ),

        "needs_review":
            bool(
                report[
                    "needs_review"
                ]
            ),

        "domain_validated":
            bool(
                report[
                    "domain"
                ][
                    "validated"
                ]
            ),

        "roi_active":
            bool(
                report[
                    "roi"
                ][
                    "active"
                ]
            ),

        "gaussian_success":
            bool(
                report[
                    "gaussian"
                ][
                    "success"
                ]
            ),
    }

    # --------------------------------------------------------
    # Top-5 SHAP
    # --------------------------------------------------------

    evidence = report[
        "top_evidence"
    ]

    for rank in range(
        1,
        6
    ):

        if rank <= len(
            evidence
        ):

            item = evidence[
                rank - 1
            ]

            row[
                f"top{rank}_feature"
            ] = item[
                "feature"
            ]

            row[
                f"top{rank}_name"
            ] = item[
                "feature_name"
            ]

            row[
                f"top{rank}_shap"
            ] = float(
                item[
                    "shap_value"
                ]
            )

        else:

            row[
                f"top{rank}_feature"
            ] = None

            row[
                f"top{rank}_name"
            ] = None

            row[
                f"top{rank}_shap"
            ] = np.nan

    return row


# ============================================================
# 5. 批量诊断
# ============================================================

def run_batch_diagnosis(
        start_index=0,
        end_index=None,
        top_n=5
):

    print("=" * 85)
    print("轴承润滑 Diagnosis Agent 批量巡检")
    print("=" * 85)

    # --------------------------------------------------------
    # MAT 只加载一次
    # --------------------------------------------------------

    data = load_mat_dataset(
        MAT_FILE
    )

    total_samples = int(
        data.shape[0]
    )

    if end_index is None:

        end_index = total_samples

    start_index = max(
        0,
        int(
            start_index
        )
    )

    end_index = min(
        total_samples,
        int(
            end_index
        )
    )

    if start_index >= end_index:

        raise ValueError(
            "start_index 必须小于 end_index"
        )

    print(
        f"\n数据集总样本数: "
        f"{total_samples}"
    )

    print(
        f"本次诊断范围: "
        f"{start_index} ~ "
        f"{end_index - 1}"
    )

    print(
        f"本次样本数: "
        f"{end_index - start_index}"
    )

    results = []

    errors = []

    # ========================================================
    # 逐条诊断
    # ========================================================

    for counter, sample_index in enumerate(
        range(
            start_index,
            end_index
        ),
        start=1
    ):

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

                top_n=top_n
            )

            row = flatten_result(
                sample_index,
                sample,
                report
            )

            results.append(
                row
            )

            # ------------------------------------------------
            # 进度
            # ------------------------------------------------

            if (
                counter == 1
                or counter % 50 == 0
                or sample_index == end_index - 1
            ):

                print(
                    f"[{counter:4d}/"
                    f"{end_index - start_index}] "
                    f"sample={sample_index} "
                    f"| p="
                    f"{row['risk_probability']:.4f} "
                    f"| "
                    f"{row['decision_status']}"
                )

        except Exception as exc:

            print(
                f"[ERROR] "
                f"sample={sample_index} "
                f"| "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            errors.append(
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
    # 6. DataFrame
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    if len(
        result_df
    ) == 0:

        raise RuntimeError(
            "没有成功诊断任何样本"
        )

    result_df.to_csv(
        RESULT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 7. 三态统计
    # ========================================================

    auto_normal_df = result_df[
        result_df[
            "decision_status"
        ] == "auto_normal"
    ].copy()

    review_df = result_df[
        result_df[
            "decision_status"
        ] == "review"
    ].copy()

    auto_fault_df = result_df[
        result_df[
            "decision_status"
        ] == "auto_fault"
    ].copy()

    total = len(
        result_df
    )

    auto_count = (
        len(
            auto_normal_df
        )
        +
        len(
            auto_fault_df
        )
    )

    review_count = len(
        review_df
    )

    auto_rate = (
        auto_count
        /
        total
    )

    review_rate = (
        review_count
        /
        total
    )

    # ========================================================
    # 8. 高风险样本
    # ========================================================

    high_risk_df = (
        result_df
        .sort_values(
            "risk_probability",
            ascending=False
        )
        .copy()
    )

    high_risk_df.to_csv(
        HIGH_RISK_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    review_df = (
        review_df
        .sort_values(
            "risk_probability",
            ascending=False
        )
    )

    review_df.to_csv(
        REVIEW_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 9. 高频 Top 特征统计
    # ========================================================

    feature_counter = {}

    for rank in range(
        1,
        6
    ):

        col = (
            f"top{rank}_feature"
        )

        for feature in result_df[
            col
        ].dropna():

            feature = str(
                feature
            )

            feature_counter.setdefault(
                feature,
                0
            )

            feature_counter[
                feature
            ] += 1

    feature_ranking = sorted(
        feature_counter.items(),
        key=lambda x: x[1],
        reverse=True
    )

    top_global_features = [
        {
            "feature":
                feature,

            "count":
                int(
                    count
                ),
        }
        for feature, count
        in feature_ranking[:10]
    ]

    # ========================================================
    # 10. 设备级统计
    # ========================================================

    equipment_summary = (
        result_df
        .groupby(
            [
                "equipment_id",
                "equipment_name",
            ],
            dropna=False
        )
        .agg(
            sample_count=(
                "sample_index",
                "count"
            ),

            mean_risk_probability=(
                "risk_probability",
                "mean"
            ),

            max_risk_probability=(
                "risk_probability",
                "max"
            ),

            review_count=(
                "needs_review",
                "sum"
            ),

            auto_fault_count=(
                "decision_status",
                lambda x:
                int(
                    np.sum(
                        x == "auto_fault"
                    )
                )
            ),
        )
        .reset_index()
    )

    equipment_summary = (
        equipment_summary
        .sort_values(
            [
                "max_risk_probability",
                "mean_risk_probability",
            ],
            ascending=False
        )
    )

    equipment_summary.to_csv(
        REPORT_DIR
        / "equipment_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 11. DeepSeek 将来使用的摘要
    # ========================================================

    summary = {

        "data_scope": {

            "start_index":
                int(
                    start_index
                ),

            "end_index":
                int(
                    end_index - 1
                ),

            "total_processed":
                int(
                    total
                ),

            "processing_errors":
                int(
                    len(
                        errors
                    )
                ),
        },

        "decision_distribution": {

            "auto_normal":
                int(
                    len(
                        auto_normal_df
                    )
                ),

            "review":
                int(
                    review_count
                ),

            "auto_fault":
                int(
                    len(
                        auto_fault_df
                    )
                ),

            "auto_rate":
                float(
                    auto_rate
                ),

            "review_rate":
                float(
                    review_rate
                ),
        },

        "risk_statistics": {

            "mean_probability":
                float(
                    result_df[
                        "risk_probability"
                    ].mean()
                ),

            "median_probability":
                float(
                    result_df[
                        "risk_probability"
                    ].median()
                ),

            "max_probability":
                float(
                    result_df[
                        "risk_probability"
                    ].max()
                ),

            "min_probability":
                float(
                    result_df[
                        "risk_probability"
                    ].min()
                ),
        },

        "top_global_features":
            top_global_features,

        "high_risk_top10":
            high_risk_df[
                [
                    "sample_index",
                    "equipment_id",
                    "equipment_name",
                    "measurement_point",
                    "risk_probability",
                    "agent_decision",
                ]
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            ),

        "equipment_top10":
            equipment_summary[
                [
                    "equipment_id",
                    "equipment_name",
                    "sample_count",
                    "mean_risk_probability",
                    "max_risk_probability",
                    "review_count",
                    "auto_fault_count",
                ]
            ]
            .head(
                10
            )
            .to_dict(
                orient="records"
            ),
    }

    # ========================================================
    # 12. JSON 类型转换
    # ========================================================

    summary = json.loads(
        json.dumps(
            summary,
            ensure_ascii=False,
            default=str
        )
    )

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            summary,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ========================================================
    # 13. 控制台输出
    # ========================================================

    print("\n")
    print("=" * 85)
    print("批量巡检结果")
    print("=" * 85)

    print(
        f"成功诊断: "
        f"{total}"
    )

    print(
        f"程序失败: "
        f"{len(errors)}"
    )

    print("\n三态分布")

    print("-" * 85)

    print(
        f"🟢 自动正常: "
        f"{len(auto_normal_df)}"
    )

    print(
        f"🟡 建议复核: "
        f"{review_count}"
    )

    print(
        f"🔴 自动故障: "
        f"{len(auto_fault_df)}"
    )

    print(
        f"自动诊断率: "
        f"{auto_rate:.2%}"
    )

    print(
        f"复核率: "
        f"{review_rate:.2%}"
    )

    print("\n风险最高 Top-10")

    print("-" * 85)

    print(
        high_risk_df[
            [
                "sample_index",
                "equipment_id",
                "measurement_point",
                "risk_probability",
                "agent_decision",
            ]
        ]
        .head(
            10
        )
        .to_string(
            index=False
        )
    )

    print("\n高频 SHAP 特征 Top-10")

    print("-" * 85)

    for rank, item in enumerate(
        top_global_features,
        start=1
    ):

        print(
            f"{rank:2d}. "
            f"{item['feature']} "
            f"| 出现次数="
            f"{item['count']}"
        )

    print("\n输出目录:")

    print(
        REPORT_DIR
    )

    return (
        result_df,
        summary
    )


# ============================================================
# 14. 主程序
# ============================================================

if __name__ == "__main__":

    run_batch_diagnosis()