import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from config import HIGH_FREQ_BANDS, HIGH_FREQ_RATIO
from msm_features import calculate_f6, spectral_entropy


def expected_paper_f6(frequency, power, fs):
    nyquist = fs / 2.0
    f_cut = HIGH_FREQ_RATIO * nyquist
    edges = np.linspace(0.0, nyquist, HIGH_FREQ_BANDS + 1)
    start_band = np.searchsorted(edges, f_cut, side="right") - 1
    start_band = int(np.clip(start_band, 0, HIGH_FREQ_BANDS - 1))

    energies = []
    for index in range(start_band, HIGH_FREQ_BANDS):
        left = edges[index]
        right = edges[index + 1]
        if index == HIGH_FREQ_BANDS - 1:
            mask = (frequency >= left) & (frequency <= right)
        else:
            mask = (frequency >= left) & (frequency < right)
        energies.append(np.sum(power[mask]))

    return spectral_entropy(np.asarray(energies, dtype=float))


def test_f6_uses_full_spectrum_band_boundaries():
    fs = 10240.0
    frequency = np.linspace(0.0, fs / 2.0, 513)
    power = 1.0 + (frequency / frequency.max()) ** 2

    actual = calculate_f6(frequency, power, fs)
    expected = expected_paper_f6(frequency, power, fs)

    assert np.isclose(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    test_f6_uses_full_spectrum_band_boundaries()
    print("f6 论文公式测试通过")
