import sys
from pathlib import Path

import numpy as np


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = (
    PROJECT_ROOT
    / "src"
)

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# 2. 导入自己的模块
# ============================================================

from data_loader import load_single_sample

from msm_features import analyze_signal

from visualization import (
    plot_time_signal,
    plot_spectrum_with_roi,
    plot_spectrum_envelopes,
    plot_gaussian_fit,
)


# ============================================================
# 3. 数据文件路径
# ============================================================

MAT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "1753data_2.mat"
)


# ============================================================
# 4. 报告目录
# ============================================================

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 5. 读取第一条样本
# ============================================================

sample = load_single_sample(
    MAT_FILE,
    sample_index=0
)

signal_data = sample[
    "signal"
]

fs = sample[
    "sampling_rate"
]


# ============================================================
# 6. MSM-17D 分析
# ============================================================

result = analyze_signal(
    signal_data,
    fs
)


frequency = result[
    "frequency"
]

amplitude = result[
    "amplitude"
]

roi = result[
    "roi"
]

envelopes = result[
    "envelopes"
]

gaussian_result = result[
    "gaussian"
]

features = result[
    "features"
]


# ============================================================
# 7. 样本信息
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "MSM-17D 单样本测试"
)

print(
    "=" * 70
)


print(
    f"设备编号: "
    f"{sample['equipment_id']}"
)

print(
    f"设备名称: "
    f"{sample['equipment_name']}"
)

print(
    f"测点: "
    f"{sample['measurement_point']}"
)

print(
    f"标签: "
    f"{sample['label']}"
)

print(
    f"转速: "
    f"{sample['speed']} rpm"
)

print(
    f"采样频率: "
    f"{fs} Hz"
)

print(
    f"采样点数: "
    f"{len(signal_data)}"
)

print(
    f"FFT长度: "
    f"{result['nfft']}"
)

print(
    f"Nyquist: "
    f"{fs / 2:.2f} Hz"
)


# ============================================================
# 8. ROI 信息
# ============================================================

print("\n")

print(
    "ROI 检测结果"
)

print(
    "-" * 70
)


print(
    f"ROI active: "
    f"{roi['active']}"
)

print(
    f"原因: "
    f"{roi['reason']}"
)


if roi["peak_frequency"] is not None:

    print(
        f"峰值频率: "
        f"{roi['peak_frequency']:.2f} Hz"
    )


if roi["peak_amplitude"] is not None:

    print(
        f"峰值幅值: "
        f"{roi['peak_amplitude']:.6f}"
    )


if roi["noise_floor"] is not None:

    print(
        f"噪声底: "
        f"{roi['noise_floor']:.6f}"
    )


if roi["activation_threshold"] is not None:

    print(
        f"ROI激活阈值: "
        f"{roi['activation_threshold']:.6f}"
    )


if roi["candidate_threshold"] is not None:

    print(
        f"候选谱点阈值: "
        f"{roi['candidate_threshold']:.6f}"
    )


print(
    f"候选谱点数量: "
    f"{roi['candidate_count']}"
)


if roi["active"]:

    print(
        f"加权质心: "
        f"{roi['centroid']:.2f} Hz"
    )

    print(
        f"加权离散度: "
        f"{roi['spread']:.2f} Hz"
    )

    print(
        f"ROI起点: "
        f"{roi['start']:.2f} Hz"
    )

    print(
        f"ROI终点: "
        f"{roi['end']:.2f} Hz"
    )

    roi_width = (
        roi["end"]
        - roi["start"]
    )

    print(
        f"ROI宽度: "
        f"{roi_width:.2f} Hz"
    )


# ============================================================
# 9. MSM-17D
# ============================================================

print("\n")

print(
    "MSM 17维特征"
)

print(
    "-" * 70
)


print("\n第一组：频谱能量迁移")


print(
    f"f1  宽带能量占比 : "
    f"{features['f1']:.6f}"
)

print(
    f"f2  高频能量占比 : "
    f"{features['f2']:.6f}"
)

print(
    f"f3  宽带质心     : "
    f"{features['f3']:.2f} Hz"
)

print(
    f"f4  宽带扩展度   : "
    f"{features['f4']:.2f} Hz"
)

print(
    f"f5  宽带谱熵     : "
    f"{features['f5']:.6f}"
)


print("\n第二组：宽带扩展与统计分散")


print(
    f"f6  高频分段谱熵 : "
    f"{features['f6']:.6f}"
)

print(
    f"f7  谱平坦度     : "
    f"{features['f7']:.6f}"
)

print(
    f"f8  频谱峭度     : "
    f"{features['f8']:.6f}"
)

print(
    f"f9  频谱偏度     : "
    f"{features['f9']:.6f}"
)


print("\n第三组：冲击-摩擦包络")


print(
    f"f10 下包络面积比 : "
    f"{features['f10']:.6f}"
)

print(
    f"f11 下包络均值   : "
    f"{features['f11']:.6f}"
)

print(
    f"f12 下包络中位数 : "
    f"{features['f12']:.6f}"
)

print(
    f"f13 下包络RMS    : "
    f"{features['f13']:.6f}"
)

print(
    f"f14 上包络RMS    : "
    f"{features['f14']:.6f}"
)


print("\n第四组：高斯频谱轮廓")


print(
    f"f15 高斯峰值A    : "
    f"{features['f15']:.6f}"
)

print(
    f"f16 高斯带宽sigma: "
    f"{features['f16']:.2f} Hz"
)

print(
    f"f17 拟合RMSE     : "
    f"{features['f17']:.6f}"
)


# ============================================================
# 10. 高斯拟合信息
# ============================================================

print("\n")

print(
    "高斯拟合信息"
)

print(
    "-" * 70
)


print(
    f"拟合成功: "
    f"{gaussian_result['success']}"
)

print(
    f"状态: "
    f"{gaussian_result['reason']}"
)


if gaussian_result[
    "success"
]:

    print(
        f"搜索主峰频率: "
        f"{gaussian_result['peak_frequency']:.2f} Hz"
    )

    print(
        f"拟合中心 mu: "
        f"{gaussian_result['mu']:.2f} Hz"
    )

    print(
        f"拟合带宽 sigma: "
        f"{gaussian_result['sigma']:.2f} Hz"
    )

    print(
        f"拟合区间: "
        f"{gaussian_result['fit_start']:.2f}"
        f" ~ "
        f"{gaussian_result['fit_end']:.2f} Hz"
    )

    print(
        f"拟合 RMSE: "
        f"{gaussian_result['rmse']:.6f}"
    )


# ============================================================
# 11. 特征数值检查
# ============================================================

print("\n")

print(
    "特征数值检查"
)

print(
    "-" * 70
)


feature_values = np.array(
    [
        features[
            f"f{i}"
        ]
        for i in range(
            1,
            18
        )
    ],
    dtype=float
)


nan_count = int(
    np.sum(
        np.isnan(
            feature_values
        )
    )
)

inf_count = int(
    np.sum(
        np.isinf(
            feature_values
        )
    )
)


print(
    f"特征数量: "
    f"{len(feature_values)}"
)

print(
    f"NaN 数量: "
    f"{nan_count}"
)

print(
    f"Inf 数量: "
    f"{inf_count}"
)


if (
    nan_count == 0
    and inf_count == 0
    and len(feature_values) == 17
):

    print(
        "MSM-17D 数值检查：PASS"
    )

else:

    print(
        "MSM-17D 数值检查：FAIL"
    )


print(
    "=" * 70
)


# ============================================================
# 12. 保存17维特征
# ============================================================

feature_output_path = (
    REPORT_DIR
    / "sample_001_msm17.csv"
)


feature_matrix = np.array(
    [
        [
            features[
                f"f{i}"
            ]
            for i in range(
                1,
                18
            )
        ]
    ],
    dtype=float
)


header = ",".join(
    [
        f"f{i}"
        for i in range(
            1,
            18
        )
    ]
)


np.savetxt(
    feature_output_path,
    feature_matrix,
    delimiter=",",
    header=header,
    comments="",
)


print("\n17维特征已保存:")

print(
    feature_output_path
)


# ============================================================
# 13. 生成检查图
# ============================================================

print("\n正在生成检查图...")


# ------------------------------------------------------------
# 图1：时域
# ------------------------------------------------------------

time_plot_path = (
    REPORT_DIR
    / "sample_001_time.png"
)


plot_time_signal(
    signal_data,
    fs,
    save_path=time_plot_path
)


# ------------------------------------------------------------
# 图2：频谱 + ROI
# ------------------------------------------------------------

spectrum_plot_path = (
    REPORT_DIR
    / "sample_001_spectrum_roi.png"
)


plot_spectrum_with_roi(
    frequency,
    amplitude,
    roi,
    save_path=spectrum_plot_path
)


# ------------------------------------------------------------
# 图3：上下包络
# ------------------------------------------------------------

envelope_plot_path = (
    REPORT_DIR
    / "sample_001_envelope.png"
)


plot_spectrum_envelopes(
    frequency,
    amplitude,
    envelopes["upper"],
    envelopes["lower"],
    save_path=envelope_plot_path
)


# ------------------------------------------------------------
# 图4：高斯拟合
# ------------------------------------------------------------

gaussian_plot_path = (
    REPORT_DIR
    / "sample_001_gaussian.png"
)


plot_gaussian_fit(
    gaussian_result,
    save_path=gaussian_plot_path
)


# ============================================================
# 14. 完成
# ============================================================

print("\n")

print(
    "=" * 70
)

print(
    "单样本 MSM-17D 测试完成"
)

print(
    "=" * 70
)


print("\n时域图:")
print(time_plot_path)

print("\n频谱 + ROI 图:")
print(spectrum_plot_path)

print("\n频谱上下包络图:")
print(envelope_plot_path)

print("\n高斯拟合图:")
print(gaussian_plot_path)

print("\n17维特征:")
print(feature_output_path)