"""Interactive signal plots tied to the same MSM result used by the workbench."""

import altair as alt
import numpy as np
import pandas as pd


SPECTRUM_COLOR = "#244F79"
ROI_COLOR = "#CA9349"


def peak_preserving_indices(values, limit=1800):
    """Limit browser payload without dropping narrow positive/negative impulses."""
    values = np.asarray(values).ravel()
    if len(values) <= limit:
        return np.arange(len(values))
    bins = np.array_split(np.arange(len(values)), max(1, (limit - 2) // 2))
    selected = [0, len(values) - 1]
    for indices in bins:
        selected.extend((indices[np.argmin(values[indices])], indices[np.argmax(values[indices])]))
    return np.unique(selected)


def envelope_window(analysis, view="full"):
    """A full-spectrum overview, or a padded ROI that retains nearby contours."""
    frequency = np.asarray(analysis["frequency"])
    start, end = float(frequency[0]), float(frequency[-1])
    roi = analysis["roi"]
    if view == "roi" and roi.get("active"):
        padding = max((float(roi["end"]) - float(roi["start"])) * 0.65, 100.0)
        start = max(start, float(roi["start"]) - padding)
        end = min(end, float(roi["end"]) + padding)
    return start, end


def signal_frames(sample, analysis, envelope_view="full"):
    signal = np.asarray(sample["signal"], dtype=float).ravel()
    fs = float(sample["sampling_rate"])
    indices = peak_preserving_indices(signal)
    waveform = pd.DataFrame({"time": indices / fs, "value": signal[indices]})
    frequency = np.asarray(analysis["frequency"])
    amplitude = np.asarray(analysis["amplitude"])
    indices = peak_preserving_indices(amplitude)
    spectrum = pd.DataFrame({"frequency": frequency[indices], "amplitude": amplitude[indices]})
    start, end = envelope_window(analysis, envelope_view)
    window_indices = np.flatnonzero((frequency >= start) & (frequency <= end))
    curves = {
        "频谱曲线": amplitude,
        "上谱包络": np.asarray(analysis["envelopes"]["upper"]),
        "下谱包络": np.asarray(analysis["envelopes"]["lower"]),
    }
    # Preserve extrema of each envelope as well as the underlying spectrum.
    # Three curves remain under Altair's 5000-row browser payload limit.
    selected = np.unique(np.concatenate([
        peak_preserving_indices(values[window_indices], limit=450)
        for values in curves.values()
    ]))
    indices = window_indices[selected]
    envelope = pd.DataFrame({
        "frequency": frequency[indices],
        **{name: values[indices] for name, values in curves.items()},
    }).melt(id_vars="frequency", var_name="curve", value_name="amplitude")
    return waveform, spectrum, envelope


def _style(chart):
    return chart.configure_view(fill="#FFFFFF", stroke="#DCE5EF", strokeWidth=0.8).configure_axis(
        labelFontSize=11, titleFontSize=12, labelColor="#425870", titleColor="#334C63",
        gridColor="#E7EDF4", gridDash=[2, 2], tickColor="#B7C8DA", domainColor="#B7C8DA",
        labelPadding=5, titlePadding=9,
    ).configure_legend(
        labelFontSize=11, labelColor="#334C63", title=None, orient="top", padding=4,
        symbolStrokeWidth=2,
    )


def _frequency_axis():
    return alt.Axis(
        tickCount=7, labelOverlap="greedy", labelFlush=True,
        labelExpr="format(datum.value / 1000, '.1f')",
    )


def waveform_chart(frame=None, height=200):
    frame = pd.DataFrame(columns=["time", "value"]) if frame is None else frame
    duration = 0.4 if frame.empty else float(frame["time"].max())
    decimals = max(2, int(np.ceil(-np.log10(duration))) + 1) if duration > 0 else 2
    y_scale = alt.Scale(domain=[-1.2, 1.2], zero=False) if frame.empty else alt.Scale(zero=False)
    chart = alt.Chart(frame).mark_line(color="#356F9D", strokeWidth=1, clip=True).encode(
        x=alt.X("time:Q", title="时间 / s", scale=alt.Scale(domain=[0, duration], nice=True),
                axis=alt.Axis(tickCount=7, format=f".{decimals}f", labelOverlap="greedy", labelFlush=True)),
        y=alt.Y("value:Q", title="原始幅值", scale=y_scale, axis=alt.Axis(tickCount=4)),
        tooltip=[alt.Tooltip("time:Q", title="时间 / s", format=".5f"),
                 alt.Tooltip("value:Q", title="原始幅值", format=".5g")],
    ).properties(height=height).interactive(bind_y=False)
    return _style(chart)


def spectrum_chart(frame=None, roi=None, height=210):
    frame = pd.DataFrame(columns=["frequency", "amplitude"]) if frame is None else frame
    roi = roi or {}
    domain = [0, 5120] if frame.empty else [float(frame["frequency"].min()), float(frame["frequency"].max())]
    line = alt.Chart(frame).mark_line(color=SPECTRUM_COLOR, strokeWidth=1.15, clip=True).encode(
        x=alt.X("frequency:Q", title="频率 / kHz", axis=_frequency_axis(),
                scale=alt.Scale(domain=domain, nice=False, zero=False)),
        y=alt.Y("amplitude:Q", title="归一化幅值", scale=alt.Scale(domain=[0, 1.05])),
        tooltip=[alt.Tooltip("frequency:Q", title="频率 / Hz", format=".1f"),
                 alt.Tooltip("amplitude:Q", title="归一化幅值", format=".4f")],
    )
    layers = []
    if not frame.empty and roi.get("active"):
        band = pd.DataFrame({"start": [float(roi["start"])], "end": [float(roi["end"])]})
        layers.append(alt.Chart(band).mark_rect(color=ROI_COLOR, opacity=0.18, clip=True).encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=domain, nice=False, zero=False)),
            x2="end:Q", y=alt.datum(0), y2=alt.datum(1.05)
        ))
    layers.append(line)
    chart = alt.layer(*layers).properties(height=height).interactive(bind_y=False)
    return _style(chart)


def roi_chart(frame=None, height=280, roi=None):
    frame = pd.DataFrame(columns=["frequency", "amplitude", "curve"]) if frame is None else frame
    domain = [0, 5120] if frame.empty else [float(frame["frequency"].min()), float(frame["frequency"].max())]
    y_max = 1.08 if frame.empty else max(float(frame["amplitude"].max()) * 1.08, 1e-6)
    curve_names = ["频谱曲线", "上谱包络", "下谱包络"]
    chart = alt.Chart(frame).mark_line(strokeWidth=1.4, clip=True).encode(
        x=alt.X("frequency:Q", title="频率 / kHz", axis=_frequency_axis(),
                scale=alt.Scale(domain=domain, nice=False, zero=False)),
        y=alt.Y("amplitude:Q", title="归一化幅值", scale=alt.Scale(domain=[0, y_max]),
                axis=alt.Axis(tickCount=5)),
        color=alt.Color("curve:N", scale=alt.Scale(
            domain=curve_names,
            range=[SPECTRUM_COLOR, "#AB4C26", "#18796D"],
        )),
        strokeDash=alt.StrokeDash("curve:N", scale=alt.Scale(
            domain=curve_names,
            range=[[1, 0], [1, 0], [5, 3]],
        )),
        order=alt.Order("frequency:Q"),
        tooltip=[alt.Tooltip("curve:N", title="曲线"),
                 alt.Tooltip("frequency:Q", title="频率 / Hz", format=".1f"),
                 alt.Tooltip("amplitude:Q", title="幅值", format=".4f")],
    )
    layers = []
    if not frame.empty and roi and roi.get("active"):
        band = pd.DataFrame({"start": [float(roi["start"])], "end": [float(roi["end"])]})
        layers.append(alt.Chart(band).mark_rect(color=ROI_COLOR, opacity=0.18, clip=True).encode(
            x=alt.X("start:Q", scale=alt.Scale(domain=domain, nice=False, zero=False)),
            x2="end:Q", y=alt.datum(0), y2=alt.datum(y_max)
        ))
    layers.append(chart)
    return _style(alt.layer(*layers).properties(height=height).interactive(bind_y=False))
