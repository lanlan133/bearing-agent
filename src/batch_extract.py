from pathlib import Path

import numpy as np
import pandas as pd

from data_loader import (
    load_mat_dataset,
    matlab_string,
)

from msm_features import analyze_signal


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)


OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "features"
)


OUTPUT_FILE = (
    OUTPUT_DIR
    / "features.csv"
)


ERROR_FILE = (
    OUTPUT_DIR
    / "failed_samples.csv"
)


# ============================================================
# 2. 安全数值转换
# ============================================================

def safe_float(
        value,
        default=np.nan
):
    """
    MATLAB cell 中的数值安全转换为 float。
    """

    try:

        return float(
            np.asarray(
                value
            ).squeeze()
        )

    except Exception:

        return default


# ============================================================
# 3. 直接从已加载的 data 中解析一行
# ============================================================

def parse_sample_from_row(
        row,
        sample_index
):
    """
    解析 data[sample_index]。

    当前已确认的 MAT 列结构：

    0  测点
    1  转速
    2  采样频率
    3  振动信号
    4  标签
    5  设备编号
    6  设备名称
    7  未确认字段
    8  时间
    9  未确认字段
    10 未确认字段
    11 来源/其他字符字段
    """

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

        signal_data = (
            signal_data.ravel()
        )

    label = matlab_string(
        row[4]
    )

    equipment_id = matlab_string(
        row[5]
    )

    equipment_name = matlab_string(
        row[6]
    )

    value_8 = safe_float(
        row[7]
    )

    timestamp = matlab_string(
        row[8]
    )

    value_10 = safe_float(
        row[9]
    )

    value_11 = safe_float(
        row[10]
    )

    source = matlab_string(
        row[11]
    )

    if len(signal_data) == 0:

        raise ValueError(
            "振动信号为空"
        )

    if not np.all(
        np.isfinite(
            signal_data
        )
    ):

        raise ValueError(
            "振动信号包含 NaN 或 Inf"
        )

    if (
        not np.isfinite(
            sampling_rate
        )
        or sampling_rate <= 0
    ):

        raise ValueError(
            f"采样频率异常: {sampling_rate}"
        )

    return {
        "sample_index":
            sample_index,

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

        "value_8":
            value_8,

        "timestamp":
            timestamp,

        "value_10":
            value_10,

        "value_11":
            value_11,

        "source":
            source,
    }


# ============================================================
# 4. 单条特征提取
# ============================================================

def extract_one_sample(
        row,
        sample_index
):
    """
    从已经加载到内存的 MAT 行中提取一条 MSM-17D。
    """

    sample = parse_sample_from_row(
        row,
        sample_index
    )

    signal_data = sample[
        "signal"
    ]

    fs = sample[
        "sampling_rate"
    ]

    # ========================================================
    # MSM-17D
    # ========================================================

    result = analyze_signal(
        signal_data,
        fs
    )

    features = result[
        "features"
    ]

    roi = result[
        "roi"
    ]

    gaussian_result = result[
        "gaussian"
    ]

    # ========================================================
    # 输出行
    # ========================================================

    output = {
        "sample_index":
            sample_index,

        "measurement_point":
            sample[
                "measurement_point"
            ],

        "speed":
            sample[
                "speed"
            ],

        "sampling_rate":
            sample[
                "sampling_rate"
            ],

        "label":
            sample[
                "label"
            ],

        "equipment_id":
            sample[
                "equipment_id"
            ],

        "equipment_name":
            sample[
                "equipment_name"
            ],

        "timestamp":
            sample[
                "timestamp"
            ],

        "source":
            sample[
                "source"
            ],

        "value_8":
            sample[
                "value_8"
            ],

        "value_10":
            sample[
                "value_10"
            ],

        "value_11":
            sample[
                "value_11"
            ],

        "signal_length":
            len(
                signal_data
            ),

        "nyquist":
            fs / 2.0,

        "nfft":
            result[
                "nfft"
            ],
    }

    # ========================================================
    # ROI 信息
    # ========================================================

    output[
        "roi_active"
    ] = bool(
        roi[
            "active"
        ]
    )

    output[
        "roi_start"
    ] = (
        roi[
            "start"
        ]
        if roi[
            "start"
        ] is not None
        else np.nan
    )

    output[
        "roi_end"
    ] = (
        roi[
            "end"
        ]
        if roi[
            "end"
        ] is not None
        else np.nan
    )

    output[
        "roi_peak_frequency"
    ] = (
        roi[
            "peak_frequency"
        ]
        if roi[
            "peak_frequency"
        ] is not None
        else np.nan
    )

    output[
        "roi_centroid"
    ] = (
        roi[
            "centroid"
        ]
        if roi[
            "centroid"
        ] is not None
        else np.nan
    )

    output[
        "roi_spread"
    ] = (
        roi[
            "spread"
        ]
        if roi[
            "spread"
        ] is not None
        else np.nan
    )

    output[
        "roi_candidate_count"
    ] = roi[
        "candidate_count"
    ]

    # ========================================================
    # 高斯拟合信息
    # ========================================================

    output[
        "gaussian_success"
    ] = bool(
        gaussian_result[
            "success"
        ]
    )

    output[
        "gaussian_mu"
    ] = (
        gaussian_result[
            "mu"
        ]
        if gaussian_result[
            "success"
        ]
        else np.nan
    )

    output[
        "gaussian_sigma"
    ] = (
        gaussian_result[
            "sigma"
        ]
        if gaussian_result[
            "success"
        ]
        else np.nan
    )

    output[
        "gaussian_rmse"
    ] = (
        gaussian_result[
            "rmse"
        ]
        if gaussian_result[
            "success"
        ]
        else np.nan
    )

    # ========================================================
    # f1 ~ f17
    # ========================================================

    for i in range(
        1,
        18
    ):

        feature_name = (
            f"f{i}"
        )

        output[
            feature_name
        ] = features[
            feature_name
        ]

    # ========================================================
    # 数值检查
    # ========================================================

    feature_values = np.array(
        [
            output[
                f"f{i}"
            ]
            for i in range(
                1,
                18
            )
        ],
        dtype=float
    )

    output[
        "feature_nan_count"
    ] = int(
        np.isnan(
            feature_values
        ).sum()
    )

    output[
        "feature_inf_count"
    ] = int(
        np.isinf(
            feature_values
        ).sum()
    )

    output[
        "feature_valid"
    ] = bool(
        np.all(
            np.isfinite(
                feature_values
            )
        )
    )

    return output


# ============================================================
# 5. 批量处理
# ============================================================

def batch_extract(
        mat_path=MAT_FILE,
        output_file=OUTPUT_FILE,
        error_file=ERROR_FILE
):
    """
    整个 MAT 只读取一次，
    然后在内存中循环 1753 条样本。
    """

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ========================================================
    # MAT 只加载一次
    # ========================================================

    data = load_mat_dataset(
        mat_path
    )

    total_samples = int(
        data.shape[0]
    )

    print("\n")

    print(
        "=" * 75
    )

    print(
        "MSM-17D 全量批量特征提取"
    )

    print(
        "=" * 75
    )

    print(
        f"总样本数: "
        f"{total_samples}"
    )

    print(
        f"MAT只加载一次: "
        f"{mat_path}"
    )

    print(
        f"输出文件: "
        f"{output_file}"
    )

    print(
        "=" * 75
    )

    results = []

    failed_records = []

    # ========================================================
    # 循环
    # ========================================================

    for sample_index in range(
        total_samples
    ):

        try:

            row = extract_one_sample(
                data[
                    sample_index
                ],
                sample_index
            )

            results.append(
                row
            )

            current = (
                sample_index + 1
            )

            # -----------------------------------------------
            # 每 25 条打印一次进度
            # -----------------------------------------------

            if (
                current == 1
                or current % 25 == 0
                or current == total_samples
            ):

                progress = (
                    100.0
                    * current
                    / total_samples
                )

                print(
                    f"[{current:4d}/"
                    f"{total_samples}] "
                    f"{progress:6.2f}% "
                    f"| label="
                    f"{row['label']} "
                    f"| ROI="
                    f"{row['roi_active']} "
                    f"| Gaussian="
                    f"{row['gaussian_success']}"
                )

        except Exception as exc:

            print(
                f"[ERROR] "
                f"sample_index="
                f"{sample_index}: "
                f"{type(exc).__name__}: "
                f"{exc}"
            )

            failed_records.append(
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
    # 6. 保存 features.csv
    # ========================================================

    df = pd.DataFrame(
        results
    )

    df.to_csv(
        output_file,
        index=False,
        encoding="utf-8-sig"
    )

    # ========================================================
    # 7. 保存失败记录
    # ========================================================

    failed_df = pd.DataFrame(
        failed_records
    )

    if len(
        failed_df
    ) > 0:

        failed_df.to_csv(
            error_file,
            index=False,
            encoding="utf-8-sig"
        )

    # ========================================================
    # 8. 汇总
    # ========================================================

    success_count = len(
        df
    )

    fail_count = len(
        failed_df
    )

    print("\n")

    print(
        "=" * 75
    )

    print(
        "批量提取完成"
    )

    print(
        "=" * 75
    )

    print(
        f"总样本数: "
        f"{total_samples}"
    )

    print(
        f"成功样本数: "
        f"{success_count}"
    )

    print(
        f"失败样本数: "
        f"{fail_count}"
    )

    if success_count > 0:

        # -----------------------------------------------
        # ROI 未激活
        # -----------------------------------------------

        roi_inactive_count = int(
            (
                ~df[
                    "roi_active"
                ].astype(bool)
            ).sum()
        )

        # -----------------------------------------------
        # Gaussian失败
        # -----------------------------------------------

        gaussian_fail_count = int(
            (
                ~df[
                    "gaussian_success"
                ].astype(bool)
            ).sum()
        )

        # -----------------------------------------------
        # 特征异常
        # -----------------------------------------------

        invalid_feature_count = int(
            (
                ~df[
                    "feature_valid"
                ].astype(bool)
            ).sum()
        )

        feature_columns = [
            f"f{i}"
            for i in range(
                1,
                18
            )
        ]

        feature_array = (
            df[
                feature_columns
            ]
            .to_numpy(
                dtype=float
            )
        )

        total_nan = int(
            np.isnan(
                feature_array
            ).sum()
        )

        total_inf = int(
            np.isinf(
                feature_array
            ).sum()
        )

        print(
            f"ROI未激活样本数: "
            f"{roi_inactive_count}"
        )

        print(
            f"高斯拟合失败数: "
            f"{gaussian_fail_count}"
        )

        print(
            f"17维特征异常样本数: "
            f"{invalid_feature_count}"
        )

        print(
            f"17维特征 NaN 总数: "
            f"{total_nan}"
        )

        print(
            f"17维特征 Inf 总数: "
            f"{total_inf}"
        )

        # -----------------------------------------------
        # 标签分布
        # -----------------------------------------------

        print("\n标签分布:")

        print(
            df[
                "label"
            ].value_counts(
                dropna=False
            )
        )

        # -----------------------------------------------
        # 设备数
        # -----------------------------------------------

        print(
            "\n设备编号数量:"
        )

        print(
            df[
                "equipment_id"
            ].nunique()
        )

        # -----------------------------------------------
        # 采样率
        # -----------------------------------------------

        print(
            "\n采样频率分布:"
        )

        print(
            df[
                "sampling_rate"
            ].value_counts(
                dropna=False
            )
        )

        # -----------------------------------------------
        # 信号长度
        # -----------------------------------------------

        print(
            "\n信号长度分布:"
        )

        print(
            df[
                "signal_length"
            ].value_counts(
                dropna=False
            )
        )

        # -----------------------------------------------
        # f1~f17 基本统计
        # -----------------------------------------------

        print(
            "\nMSM-17D 基本统计:"
        )

        print(
            df[
                feature_columns
            ].describe().T[
                [
                    "mean",
                    "std",
                    "min",
                    "max"
                ]
            ]
        )

    print("\n")

    print(
        f"features.csv 已保存:"
    )

    print(
        output_file
    )

    if fail_count > 0:

        print(
            "\n失败记录已保存:"
        )

        print(
            error_file
        )

    print(
        "=" * 75
    )

    return (
        df,
        failed_df
    )


# ============================================================
# 9. 主程序
# ============================================================

if __name__ == "__main__":

    batch_extract()