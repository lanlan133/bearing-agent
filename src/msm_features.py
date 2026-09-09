import numpy as np
from scipy import signal
from scipy.stats import kurtosis, skew
from scipy.optimize import curve_fit

from config import (
    F_SEARCH,
    GAMMA,
    RHO,
    ROI_MAX_EXPANSION,
    ROI_MIN_HALF_WIDTH,
    HIGH_FREQ_RATIO,
    HIGH_FREQ_BANDS,
    ENVELOPE_SPAN,
    GAUSSIAN_SEARCH_START,
    GAUSSIAN_HALF_WIDTH,
)


# ============================================================
# 1. 信号预处理
# ============================================================

def preprocess_signal(x):
    """
    信号预处理：
    1. 转为一维 float 数组
    2. 检查 NaN / Inf
    3. 线性去趋势
    """

    x = np.asarray(x, dtype=float).ravel()

    if len(x) == 0:
        raise ValueError("振动信号为空")

    if not np.all(np.isfinite(x)):
        raise ValueError("振动信号中存在 NaN 或 Inf")

    x = signal.detrend(
        x,
        type="linear"
    )

    return x


# ============================================================
# 2. 单边 FFT
# ============================================================

def calculate_fft(x, fs):
    """
    计算单边 FFT。

    返回：
    frequency
    amplitude_norm
    amplitude_raw
    nfft
    """

    if fs <= 0:
        raise ValueError(
            f"采样频率必须大于0，当前为: {fs}"
        )

    x = preprocess_signal(x)

    n = len(x)

    nfft = 2 ** int(
        np.ceil(
            np.log2(n)
        )
    )

    fft_result = np.fft.rfft(
        x,
        n=nfft
    )

    amplitude_raw = np.abs(
        fft_result
    )

    frequency = np.fft.rfftfreq(
        nfft,
        d=1.0 / fs
    )

    max_amp = np.max(amplitude_raw)

    if max_amp > 0:
        amplitude_norm = (
            amplitude_raw
            / max_amp
        )
    else:
        amplitude_norm = (
            amplitude_raw.copy()
        )

    return (
        frequency,
        amplitude_norm,
        amplitude_raw,
        nfft
    )


# ============================================================
# 3. 加权质心
# ============================================================

def weighted_centroid(
        frequency,
        weights
):
    """
    mu = sum(f*w) / sum(w)
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    weight_sum = np.sum(weights)

    if weight_sum <= 0:
        return None

    centroid = np.sum(
        frequency * weights
    ) / weight_sum

    return float(centroid)


# ============================================================
# 4. 加权离散度
# ============================================================

def weighted_spread(
        frequency,
        weights,
        centroid
):
    """
    sigma =
    sqrt(
        sum(w * (f-mu)^2)
        /
        sum(w)
    )
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    weights = np.asarray(
        weights,
        dtype=float
    )

    weight_sum = np.sum(weights)

    if (
        weight_sum <= 0
        or centroid is None
    ):
        return None

    variance = np.sum(
        weights
        * (
            frequency - centroid
        ) ** 2
    ) / weight_sum

    spread = np.sqrt(
        max(
            variance,
            0.0
        )
    )

    return float(spread)


# ============================================================
# 5. 动态 ROI
# ============================================================

def detect_roi(
        frequency,
        amplitude,
        fs
):
    """
    动态高频 ROI：

    1. 在 F_SEARCH ~ Nyquist 搜索
    2. 非零谱幅值中位数作为噪声底
    3. 最大峰 >= GAMMA * noise_floor 时激活
    4. >= RHO * peak_amplitude 的频点作为候选
    5. 候选谱点计算加权质心和加权离散度
    6. ROI = centroid ± spread
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    amplitude = np.asarray(
        amplitude,
        dtype=float
    )

    nyquist = fs / 2.0

    search_mask = (
        (frequency >= F_SEARCH)
        &
        (frequency <= nyquist)
    )

    f_search = frequency[
        search_mask
    ]

    a_search = amplitude[
        search_mask
    ]

    if len(f_search) == 0:

        return {
            "active": False,
            "reason": "没有可用的高频搜索区域",
            "start": None,
            "end": None,
            "peak_frequency": None,
            "peak_amplitude": None,
            "noise_floor": None,
            "activation_threshold": None,
            "candidate_threshold": None,
            "centroid": None,
            "spread": None,
            "candidate_count": 0,
        }

    nonzero = a_search[
        a_search > 0
    ]

    if len(nonzero) == 0:

        return {
            "active": False,
            "reason": "高频搜索区谱幅值全部为0",
            "start": None,
            "end": None,
            "peak_frequency": None,
            "peak_amplitude": None,
            "noise_floor": 0.0,
            "activation_threshold": 0.0,
            "candidate_threshold": None,
            "centroid": None,
            "spread": None,
            "candidate_count": 0,
        }

    # 噪声底
    noise_floor = np.median(nonzero)

    # 最大峰
    peak_idx = np.argmax(a_search)

    peak_frequency = float(
        f_search[peak_idx]
    )

    peak_amplitude = float(
        a_search[peak_idx]
    )

    activation_threshold = (
        GAMMA * noise_floor
    )

    # ROI 未激活
    if (
        peak_amplitude
        < activation_threshold
    ):

        return {
            "active": False,
            "reason": "峰值未达到ROI激活阈值",
            "start": None,
            "end": None,
            "peak_frequency": peak_frequency,
            "peak_amplitude": peak_amplitude,
            "noise_floor": float(noise_floor),
            "activation_threshold": float(
                activation_threshold
            ),
            "candidate_threshold": None,
            "centroid": None,
            "spread": None,
            "candidate_count": 0,
        }

    # 候选谱点
    candidate_threshold = (
        RHO * peak_amplitude
    )

    candidate_mask = (
        a_search
        >= candidate_threshold
    )

    candidate_f = f_search[
        candidate_mask
    ]

    candidate_a = a_search[
        candidate_mask
    ]

    candidate_count = len(
        candidate_f
    )

    if candidate_count < 2:

        centroid = peak_frequency
        spread = ROI_MIN_HALF_WIDTH

    else:

        centroid = weighted_centroid(
            candidate_f,
            candidate_a
        )

        spread = weighted_spread(
            candidate_f,
            candidate_a,
            centroid
        )

        if spread is None:
            spread = ROI_MIN_HALF_WIDTH

    half_width = max(
        spread,
        ROI_MIN_HALF_WIDTH
    )

    half_width = min(
        half_width,
        ROI_MAX_EXPANSION
    )

    roi_start = (
        centroid
        - half_width
    )

    roi_end = (
        centroid
        + half_width
    )

    roi_start = max(
        F_SEARCH,
        roi_start
    )

    roi_end = min(
        nyquist,
        roi_end
    )

    return {
        "active": True,
        "reason": (
            "ROI successfully activated "
            "using weighted centroid and weighted spread"
        ),
        "start": float(roi_start),
        "end": float(roi_end),
        "peak_frequency": float(
            peak_frequency
        ),
        "peak_amplitude": float(
            peak_amplitude
        ),
        "noise_floor": float(
            noise_floor
        ),
        "activation_threshold": float(
            activation_threshold
        ),
        "candidate_threshold": float(
            candidate_threshold
        ),
        "centroid": float(
            centroid
        ),
        "spread": float(
            spread
        ),
        "candidate_count": int(
            candidate_count
        ),
    }


# ============================================================
# 6. 谱熵
# ============================================================

def spectral_entropy(values):
    """
    归一化谱熵
    """

    values = np.asarray(
        values,
        dtype=float
    )

    values = np.maximum(
        values,
        0.0
    )

    total = np.sum(values)

    if (
        total <= 0
        or len(values) <= 1
    ):
        return 0.0

    p = values / total

    p = p[
        p > 0
    ]

    entropy = -np.sum(
        p * np.log(p)
    )

    normalized_entropy = (
        entropy
        / np.log(len(values))
    )

    return float(
        normalized_entropy
    )


# ============================================================
# 7. f1 ~ f5
# ============================================================

def extract_features_f1_f5(
        frequency,
        amplitude_raw,
        roi,
        fs
):
    """
    f1 宽带能量占比
    f2 高频能量占比
    f3 宽带质心
    f4 宽带扩展度
    f5 宽带谱熵
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    amplitude_raw = np.asarray(
        amplitude_raw,
        dtype=float
    )

    power = amplitude_raw ** 2

    total_energy = np.sum(power)

    nyquist = fs / 2.0

    high_freq_start = (
        HIGH_FREQ_RATIO
        * nyquist
    )

    high_mask = (
        frequency
        >= high_freq_start
    )

    high_energy = np.sum(
        power[
            high_mask
        ]
    )

    if total_energy > 0:
        f2 = (
            high_energy
            / total_energy
        )
    else:
        f2 = 0.0

    # ROI 未激活
    if not roi["active"]:

        return {
            "f1": 0.0,
            "f2": float(f2),
            "f3": 0.0,
            "f4": 0.0,
            "f5": 0.0,
        }

    roi_mask = (
        (frequency >= roi["start"])
        &
        (frequency <= roi["end"])
    )

    roi_f = frequency[
        roi_mask
    ]

    roi_power = power[
        roi_mask
    ]

    roi_energy = np.sum(
        roi_power
    )

    # f1
    if total_energy > 0:
        f1 = (
            roi_energy
            / total_energy
        )
    else:
        f1 = 0.0

    # f3
    if (
        len(roi_f) > 0
        and np.sum(roi_power) > 0
    ):

        f3 = np.sum(
            roi_f
            * roi_power
        ) / np.sum(
            roi_power
        )

    else:
        f3 = 0.0

    # f4
    if (
        len(roi_f) > 0
        and np.sum(roi_power) > 0
    ):

        variance = np.sum(
            roi_power
            * (
                roi_f - f3
            ) ** 2
        ) / np.sum(
            roi_power
        )

        f4 = np.sqrt(
            max(
                variance,
                0.0
            )
        )

    else:
        f4 = 0.0

    # f5
    f5 = spectral_entropy(
        roi_power
    )

    return {
        "f1": float(f1),
        "f2": float(f2),
        "f3": float(f3),
        "f4": float(f4),
        "f5": float(f5),
    }


# ============================================================
# 8. Welch PSD
# ============================================================

def calculate_welch_psd(
        x,
        fs
):
    """
    Welch PSD

    窗长：
    min(1024, N/4)

    重叠：
    50%
    """

    x = preprocess_signal(x)

    n = len(x)

    nperseg = min(
        1024,
        max(
            8,
            n // 4
        )
    )

    noverlap = (
        nperseg // 2
    )

    frequency, psd = signal.welch(
        x,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
        detrend=False,
        scaling="density"
    )

    return (
        frequency,
        psd
    )


# ============================================================
# 9. f6
# ============================================================

def calculate_f6(
        frequency,
        power,
        fs
):
    """
    高频区域分成 HIGH_FREQ_BANDS 个子频带，
    对子频带能量计算归一化谱熵。
    """

    nyquist = fs / 2.0

    high_start = (
        HIGH_FREQ_RATIO
        * nyquist
    )

    edges = np.linspace(
        high_start,
        nyquist,
        HIGH_FREQ_BANDS + 1
    )

    band_energies = []

    for i in range(
        HIGH_FREQ_BANDS
    ):

        left = edges[i]
        right = edges[i + 1]

        if i == HIGH_FREQ_BANDS - 1:

            mask = (
                (frequency >= left)
                &
                (frequency <= right)
            )

        else:

            mask = (
                (frequency >= left)
                &
                (frequency < right)
            )

        energy = np.sum(
            power[
                mask
            ]
        )

        band_energies.append(
            energy
        )

    band_energies = np.asarray(
        band_energies,
        dtype=float
    )

    return spectral_entropy(
        band_energies
    )


# ============================================================
# 10. f7
# ============================================================

def calculate_f7(
        x,
        fs
):
    """
    高频谱平坦度
    """

    frequency, psd = calculate_welch_psd(
        x,
        fs
    )

    nyquist = fs / 2.0

    high_start = (
        HIGH_FREQ_RATIO
        * nyquist
    )

    mask = (
        frequency
        >= high_start
    )

    high_psd = psd[
        mask
    ]

    if len(high_psd) == 0:
        return 0.0

    eps = np.finfo(float).eps

    high_psd = np.maximum(
        high_psd,
        eps
    )

    geometric_mean = np.exp(
        np.mean(
            np.log(
                high_psd
            )
        )
    )

    arithmetic_mean = np.mean(
        high_psd
    )

    if arithmetic_mean <= 0:
        return 0.0

    flatness = (
        geometric_mean
        / arithmetic_mean
    )

    return float(flatness)


# ============================================================
# 11. f8 / f9
# ============================================================

def calculate_f8_f9(
        frequency,
        amplitude_raw,
        fs
):
    """
    f8 频谱峭度
    f9 频谱偏度
    """

    nyquist = fs / 2.0

    high_start = (
        HIGH_FREQ_RATIO
        * nyquist
    )

    mask = (
        frequency
        >= high_start
    )

    high_amplitude = amplitude_raw[
        mask
    ]

    if len(high_amplitude) < 4:
        return (
            0.0,
            0.0
        )

    if np.std(high_amplitude) <= 0:
        return (
            0.0,
            0.0
        )

    f8 = kurtosis(
        high_amplitude,
        fisher=False,
        bias=False
    )

    f9 = skew(
        high_amplitude,
        bias=False
    )

    if not np.isfinite(f8):
        f8 = 0.0

    if not np.isfinite(f9):
        f9 = 0.0

    return (
        float(f8),
        float(f9)
    )


# ============================================================
# 12. f6 ~ f9
# ============================================================

def extract_features_f6_f9(
        x,
        frequency,
        amplitude_raw,
        fs
):
    """
    f6 高频分段谱熵
    f7 谱平坦度
    f8 频谱峭度
    f9 频谱偏度
    """

    power = (
        amplitude_raw ** 2
    )

    f6 = calculate_f6(
        frequency,
        power,
        fs
    )

    f7 = calculate_f7(
        x,
        fs
    )

    f8, f9 = calculate_f8_f9(
        frequency,
        amplitude_raw,
        fs
    )

    return {
        "f6": float(f6),
        "f7": float(f7),
        "f8": float(f8),
        "f9": float(f9),
    }


# ============================================================
# 13. 频谱上下包络
# ============================================================

def extract_spectrum_envelopes(
        frequency,
        amplitude,
        span=ENVELOPE_SPAN
):
    """
    从归一化单边频谱提取上下包络。

    当前 Python 工程实现：
    局部峰/谷 + 线性插值。
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    amplitude = np.asarray(
        amplitude,
        dtype=float
    )

    if len(amplitude) < 3:
        raise ValueError(
            "频谱点数太少，无法提取包络"
        )

    span = max(
        1,
        int(span)
    )

    # 上包络节点
    upper_idx, _ = signal.find_peaks(
        amplitude,
        distance=span
    )

    # 下包络节点
    lower_idx, _ = signal.find_peaks(
        -amplitude,
        distance=span
    )

    # 加首尾点
    upper_idx = np.unique(
        np.concatenate(
            (
                [0],
                upper_idx,
                [len(amplitude) - 1]
            )
        )
    )

    lower_idx = np.unique(
        np.concatenate(
            (
                [0],
                lower_idx,
                [len(amplitude) - 1]
            )
        )
    )

    # 插值
    upper_envelope = np.interp(
        frequency,
        frequency[
            upper_idx
        ],
        amplitude[
            upper_idx
        ]
    )

    lower_envelope = np.interp(
        frequency,
        frequency[
            lower_idx
        ],
        amplitude[
            lower_idx
        ]
    )

    # 保证上下关系
    upper_envelope = np.maximum(
        upper_envelope,
        amplitude
    )

    lower_envelope = np.minimum(
        lower_envelope,
        amplitude
    )

    lower_envelope = np.maximum(
        lower_envelope,
        0.0
    )

    upper_envelope = np.clip(
        upper_envelope,
        0.0,
        1.0
    )

    lower_envelope = np.clip(
        lower_envelope,
        0.0,
        1.0
    )

    return {
        "upper": upper_envelope,
        "lower": lower_envelope,
        "upper_indices": upper_idx,
        "lower_indices": lower_idx,
    }


# ============================================================
# 14. f10 ~ f14
# ============================================================

def extract_features_f10_f14(
        frequency,
        amplitude,
        envelopes
):
    """
    f10 下包络面积比
    f11 下包络均值
    f12 下包络中位数
    f13 下包络 RMS
    f14 上包络 RMS
    """

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    amplitude = np.asarray(
        amplitude,
        dtype=float
    )

    lower = np.asarray(
        envelopes["lower"],
        dtype=float
    )

    upper = np.asarray(
        envelopes["upper"],
        dtype=float
    )

    # f10
    spectrum_area = np.trapezoid(
        amplitude,
        frequency
    )

    lower_area = np.trapezoid(
        lower,
        frequency
    )

    if spectrum_area > 0:
        f10 = (
            lower_area
            / spectrum_area
        )
    else:
        f10 = 0.0

    # f11
    if len(lower) > 0:
        f11 = np.mean(lower)
    else:
        f11 = 0.0

    # f12
    if len(lower) > 0:
        f12 = np.median(lower)
    else:
        f12 = 0.0

    # f13
    if len(lower) > 0:

        f13 = np.sqrt(
            np.mean(
                lower ** 2
            )
        )

    else:
        f13 = 0.0

    # f14
    if len(upper) > 0:

        f14 = np.sqrt(
            np.mean(
                upper ** 2
            )
        )

    else:
        f14 = 0.0

    return {
        "f10": float(f10),
        "f11": float(f11),
        "f12": float(f12),
        "f13": float(f13),
        "f14": float(f14),
    }


# ============================================================
# 15. 高斯模型
# ============================================================

def gaussian_model(
        frequency,
        amplitude,
        center,
        sigma
):
    """
    G(f) =
    A * exp(
        -(f-mu)^2 / (2*sigma^2)
    )
    """

    return (
        amplitude
        * np.exp(
            -(
                (
                    frequency
                    - center
                ) ** 2
            )
            /
            (
                2.0
                * sigma ** 2
            )
        )
    )


# ============================================================
# 16. 高频高斯拟合
# ============================================================

def fit_gaussian_spectrum(
        x,
        fs
):
    """
    Welch PSD 高频轮廓高斯拟合。

    1. Welch PSD
    2. 500 Hz以上寻找主峰
    3. 主峰 ± 2000 Hz 内拟合
    4. 输出 A、mu、sigma、RMSE
    """

    frequency, psd = calculate_welch_psd(
        x,
        fs
    )

    frequency = np.asarray(
        frequency,
        dtype=float
    )

    psd = np.asarray(
        psd,
        dtype=float
    )

    nyquist = fs / 2.0

    # 搜索范围
    search_mask = (
        (frequency >= GAUSSIAN_SEARCH_START)
        &
        (frequency <= nyquist)
    )

    search_f = frequency[
        search_mask
    ]

    search_psd = psd[
        search_mask
    ]

    if (
        len(search_f) < 5
        or np.max(search_psd) <= 0
    ):

        return {
            "success": False,
            "reason": "没有足够的高频数据用于高斯拟合",
            "A": 0.0,
            "mu": 0.0,
            "sigma": 0.0,
            "rmse": 0.0,
            "peak_frequency": 0.0,
            "fit_start": 0.0,
            "fit_end": 0.0,
            "frequency": np.array([]),
            "spectrum": np.array([]),
            "fitted": np.array([]),
        }

    # 主峰
    peak_index = np.argmax(
        search_psd
    )

    peak_frequency = float(
        search_f[
            peak_index
        ]
    )

    # 拟合区间
    fit_start = max(
        GAUSSIAN_SEARCH_START,
        peak_frequency
        - GAUSSIAN_HALF_WIDTH
    )

    fit_end = min(
        nyquist,
        peak_frequency
        + GAUSSIAN_HALF_WIDTH
    )

    fit_mask = (
        (frequency >= fit_start)
        &
        (frequency <= fit_end)
    )

    fit_f = frequency[
        fit_mask
    ]

    fit_psd = psd[
        fit_mask
    ]

    if (
        len(fit_f) < 5
        or np.max(fit_psd) <= 0
    ):

        return {
            "success": False,
            "reason": "高斯拟合区域数据不足",
            "A": 0.0,
            "mu": 0.0,
            "sigma": 0.0,
            "rmse": 0.0,
            "peak_frequency": peak_frequency,
            "fit_start": float(fit_start),
            "fit_end": float(fit_end),
            "frequency": fit_f,
            "spectrum": fit_psd,
            "fitted": np.zeros_like(
                fit_psd
            ),
        }

    # 归一化
    fit_psd_norm = (
        fit_psd
        / np.max(fit_psd)
    )

    # 初值
    A0 = 1.0

    mu0 = peak_frequency

    sigma0 = min(
        500.0,
        GAUSSIAN_HALF_WIDTH / 2.0
    )

    # 频率分辨率
    if len(fit_f) > 1:

        df = np.median(
            np.diff(
                fit_f
            )
        )

    else:
        df = 1.0

    min_sigma = max(
        float(df),
        1.0
    )

    max_sigma = max(
        GAUSSIAN_HALF_WIDTH,
        min_sigma * 2.0
    )

    try:

        params, _ = curve_fit(
            gaussian_model,
            fit_f,
            fit_psd_norm,
            p0=[
                A0,
                mu0,
                sigma0
            ],
            bounds=(
                [
                    0.0,
                    fit_start,
                    min_sigma
                ],
                [
                    2.0,
                    fit_end,
                    max_sigma
                ]
            ),
            maxfev=20000
        )

        A = float(
            params[0]
        )

        mu = float(
            params[1]
        )

        sigma = float(
            params[2]
        )

        fitted = gaussian_model(
            fit_f,
            A,
            mu,
            sigma
        )

        rmse = np.sqrt(
            np.mean(
                (
                    fit_psd_norm
                    - fitted
                ) ** 2
            )
        )

        success = True
        reason = "Gaussian fit successful"

    except Exception as exc:

        A = 0.0
        mu = 0.0
        sigma = 0.0
        rmse = 0.0

        fitted = np.zeros_like(
            fit_psd_norm
        )

        success = False

        reason = (
            f"Gaussian fit failed: {exc}"
        )

    return {
        "success": success,
        "reason": reason,
        "A": float(A),
        "mu": float(mu),
        "sigma": float(sigma),
        "rmse": float(rmse),
        "peak_frequency": float(
            peak_frequency
        ),
        "fit_start": float(
            fit_start
        ),
        "fit_end": float(
            fit_end
        ),
        "frequency": fit_f,
        "spectrum": fit_psd_norm,
        "fitted": fitted,
    }


# ============================================================
# 17. f15 ~ f17
# ============================================================

def extract_features_f15_f17(
        gaussian_result
):
    """
    f15 高斯峰值 A
    f16 高斯带宽 sigma
    f17 拟合 RMSE
    """

    if not gaussian_result[
        "success"
    ]:

        return {
            "f15": 0.0,
            "f16": 0.0,
            "f17": 0.0,
        }

    return {
        "f15": float(
            gaussian_result["A"]
        ),
        "f16": float(
            gaussian_result["sigma"]
        ),
        "f17": float(
            gaussian_result["rmse"]
        ),
    }


# ============================================================
# 18. 完整 MSM-17D
# ============================================================

def analyze_signal(
        x,
        fs
):
    """
    完整流程：

    原始信号
        ↓
    预处理
        ↓
    FFT
        ↓
    动态ROI
        ↓
    f1 ~ f5
        ↓
    f6 ~ f9
        ↓
    上下包络
        ↓
    f10 ~ f14
        ↓
    高斯拟合
        ↓
    f15 ~ f17
    """

    (
        frequency,
        amplitude,
        amplitude_raw,
        nfft
    ) = calculate_fft(
        x,
        fs
    )

    # ROI
    roi = detect_roi(
        frequency,
        amplitude,
        fs
    )

    # f1 ~ f5
    features_1 = extract_features_f1_f5(
        frequency,
        amplitude_raw,
        roi,
        fs
    )

    # f6 ~ f9
    features_2 = extract_features_f6_f9(
        x,
        frequency,
        amplitude_raw,
        fs
    )

    # 包络
    envelopes = extract_spectrum_envelopes(
        frequency,
        amplitude,
        span=ENVELOPE_SPAN
    )

    # f10 ~ f14
    features_3 = extract_features_f10_f14(
        frequency,
        amplitude,
        envelopes
    )

    # 高斯拟合
    gaussian_result = fit_gaussian_spectrum(
        x,
        fs
    )

    # f15 ~ f17
    features_4 = extract_features_f15_f17(
        gaussian_result
    )

    features = {
        **features_1,
        **features_2,
        **features_3,
        **features_4,
    }

    return {
        "frequency": frequency,
        "amplitude": amplitude,
        "amplitude_raw": amplitude_raw,
        "roi": roi,
        "envelopes": envelopes,
        "gaussian": gaussian_result,
        "features": features,
        "nfft": nfft,
    }