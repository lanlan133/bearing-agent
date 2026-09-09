from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 1. 时域波形
# ============================================================

def plot_time_signal(
        signal,
        fs,
        save_path=None
):
    """
    绘制时域振动波形
    """

    signal = np.asarray(
        signal,
        dtype=float
    ).ravel()

    time = (
        np.arange(len(signal))
        / fs
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        time,
        signal,
        linewidth=0.8
    )

    plt.xlabel(
        "Time (s)"
    )

    plt.ylabel(
        "Amplitude"
    )

    plt.title(
        "Vibration Signal - Time Domain"
    )

    plt.grid(
        alpha=0.3
    )

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=150
        )

    plt.show()


# ============================================================
# 2. 单边频谱 + 动态 ROI
# ============================================================

def plot_spectrum_with_roi(
        frequency,
        amplitude,
        roi,
        save_path=None
):
    """
    绘制归一化单边频谱和动态 ROI
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    amplitude = np.asarray(
        amplitude,
        dtype=float
    )

    plt.figure(
        figsize=(12, 5)
    )

    # 原始频谱
    plt.plot(
        frequency,
        amplitude,
        linewidth=0.8,
        label="Spectrum"
    )

    # ROI
    if roi["active"]:

        plt.axvspan(
            roi["start"],
            roi["end"],
            alpha=0.25,
            label="Dynamic ROI"
        )

        # 最大峰
        if roi["peak_frequency"] is not None:

            plt.axvline(
                roi["peak_frequency"],
                linestyle="--",
                linewidth=1.2,
                label=(
                    f"Peak = "
                    f"{roi['peak_frequency']:.1f} Hz"
                )
            )

        # 加权质心
        if roi["centroid"] is not None:

            plt.axvline(
                roi["centroid"],
                linestyle="-.",
                linewidth=1.2,
                label=(
                    f"Centroid = "
                    f"{roi['centroid']:.1f} Hz"
                )
            )

    # ROI搜索起点
    plt.axvline(
        1200,
        linestyle=":",
        linewidth=1.0,
        label="Search Start = 1200 Hz"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Normalized Amplitude"
    )

    plt.title(
        "Normalized One-sided Spectrum and Dynamic ROI"
    )

    plt.xlim(
        0,
        frequency[-1]
    )

    plt.ylim(
        bottom=0
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=150
        )

    plt.show()


# ============================================================
# 3. 频谱上下包络
# ============================================================

def plot_spectrum_envelopes(
        frequency,
        amplitude,
        upper_envelope,
        lower_envelope,
        save_path=None
):
    """
    绘制单边频谱及上下包络
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    amplitude = np.asarray(
        amplitude,
        dtype=float
    )

    upper_envelope = np.asarray(
        upper_envelope,
        dtype=float
    )

    lower_envelope = np.asarray(
        lower_envelope,
        dtype=float
    )

    plt.figure(
        figsize=(12, 5)
    )

    # 原频谱
    plt.plot(
        frequency,
        amplitude,
        linewidth=0.7,
        label="Spectrum"
    )

    # 上包络
    plt.plot(
        frequency,
        upper_envelope,
        linewidth=1.3,
        label="Upper Envelope"
    )

    # 下包络
    plt.plot(
        frequency,
        lower_envelope,
        linewidth=1.3,
        label="Lower Envelope"
    )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Normalized Amplitude"
    )

    plt.title(
        "Spectrum Upper and Lower Envelopes"
    )

    plt.xlim(
        0,
        frequency[-1]
    )

    plt.ylim(
        bottom=0
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=150
        )

    plt.show()


# ============================================================
# 4. 高斯拟合检查图
# ============================================================

def plot_gaussian_fit(
        gaussian_result,
        save_path=None
):
    """
    绘制 Welch 高频频谱轮廓与高斯拟合结果
    """

    frequency = np.asarray(
        gaussian_result["frequency"],
        dtype=float
    )

    spectrum = np.asarray(
        gaussian_result["spectrum"],
        dtype=float
    )

    fitted = np.asarray(
        gaussian_result["fitted"],
        dtype=float
    )

    # --------------------------------
    # 没有数据
    # --------------------------------

    if len(frequency) == 0:

        print(
            "没有可绘制的高斯拟合数据"
        )

        return

    plt.figure(
        figsize=(12, 5)
    )

    # --------------------------------
    # Welch PSD
    # --------------------------------

    plt.plot(
        frequency,
        spectrum,
        linewidth=1.0,
        label="Normalized Welch PSD"
    )

    # --------------------------------
    # Gaussian Fit
    # --------------------------------

    plt.plot(
        frequency,
        fitted,
        linewidth=2.0,
        label="Gaussian Fit"
    )

    # --------------------------------
    # 拟合成功时显示中心
    # --------------------------------

    if gaussian_result["success"]:

        plt.axvline(
            gaussian_result["mu"],
            linestyle="--",
            linewidth=1.2,
            label=(
                f"mu = "
                f"{gaussian_result['mu']:.1f} Hz"
            )
        )

        title = (
            "High-frequency Gaussian Spectrum Fitting\n"
            f"A = {gaussian_result['A']:.4f}, "
            f"sigma = {gaussian_result['sigma']:.1f} Hz, "
            f"RMSE = {gaussian_result['rmse']:.4f}"
        )

    else:

        title = (
            "High-frequency Gaussian Spectrum Fitting "
            "(Fit Failed)"
        )

    plt.xlabel(
        "Frequency (Hz)"
    )

    plt.ylabel(
        "Normalized PSD"
    )

    plt.title(
        title
    )

    plt.grid(
        alpha=0.3
    )

    plt.legend()

    plt.tight_layout()

    # --------------------------------
    # 保存
    # --------------------------------

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        plt.savefig(
            save_path,
            dpi=150
        )

    plt.show()