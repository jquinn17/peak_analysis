import streamlit as st
import plotly.graph_objects as go
import numpy as np
from scipy.signal import find_peaks, savgol_filter
import csv
import io

st.set_page_config(page_title="MSU RUS Peak Analysis", layout="wide")
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; }
    .stSlider > div { padding-top: 0.2rem; }
    div[data-testid="stHorizontalBlock"] button { width: 100%; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parameters")

    st.subheader("Data")
    data_type = st.selectbox("Data type", ["mag", "imag", "real"], index=0)
    bad_temps_input = st.text_input("Bad temps (comma-separated)", "")
    bad_temps = tuple(float(t.strip()) for t in bad_temps_input.split(",") if t.strip())

    st.subheader("Frequency range (kHz)")
    min_freq = st.number_input("Min freq", value=290.0, step=1.0)
    max_freq = st.number_input("Max freq", value=400.0, step=1.0)
    factor = st.number_input("Factor", value=1000.0, step=10.0)

    st.subheader("Temperature range (K)")
    min_temp = st.number_input("Min temp", value=80.0, step=1.0)
    max_temp = st.number_input("Max temp", value=300.0, step=1.0)
    temp_interval = st.number_input("Temp interval", value=0.5, step=0.1, format="%.2f")

    st.subheader("Peak detection")
    prominence = st.slider("Prominence", 0.001, 0.5, 0.055, step=0.001, format="%.3f")

    st.subheader("Smoothing (Signal & Peaks)")
    smooth_order = st.slider("Poly fit order", 1, 8, 3)
    smooth_window = st.slider("Window length", 3, 101, 21, step=2)

    st.divider()
    st.subheader("🌊 Waterfall Parameters")
    wf_min_freq = st.number_input("WF Min freq (kHz)", value=220.0, step=1.0)
    wf_max_freq = st.number_input("WF Max freq (kHz)", value=270.0, step=1.0)
    wf_min_temp = st.number_input("WF Min temp (K)", value=0.0, step=1.0)
    wf_max_temp = st.number_input("WF Max temp (K)", value=285.0, step=1.0)
    wf_temp_interval = st.number_input("WF Temp interval", value=7.0, step=0.5, format="%.1f")
    wf_power = st.number_input("WF Power", value=1.0, step=0.5, format="%.1f")
    wf_peak_multiplier = st.number_input("WF Peak multiplier", value=1.0, step=0.1, format="%.2f")
    wf_smooth_order = st.slider("WF Poly fit order", 1, 8, 3, key="wf_ord")
    wf_smooth_window = st.slider("WF Window length", 3, 101, 21, step=2, key="wf_win")
    wf_color_scheme = st.selectbox("WF Color scheme", [
        "Blue → Cyan",
        "Cyan → Blue",
        "Red → Blue",
        "Blue → Red",
        "Red → Yellow",
        "Yellow → Red",
        "Green → Blue",
        "Blue → Green",
        "Purple → Orange",
        "Orange → Purple",
    ])

# ── File upload ───────────────────────────────────────────────────────────────
st.title("MSU RUS Peak Analysis")
uploaded_files = st.file_uploader("Upload .dat files", type=["dat"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more `.dat` files to get started.")
    st.stop()

# ── Shared file read (cached) ─────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_raw_files(file_contents):
    """Parse all files once; return raw temps, freqs, mag/imag/real lists."""
    temperatures, mag_vals, imag_vals, real_vals, frequencies = [], [], [], [], []
    for content in file_contents:
        lines = content.decode("utf-8", errors="ignore").splitlines()
        temp = None
        for line in lines:
            if "Temperature" in line:
                temp = float(line.split(":")[1].strip().split()[0])
                break
        if temp is None:
            continue
        freqs, mags, imags, reals = [], [], [], []
        for line in lines[4:]:
            parts = line.split()
            if len(parts) < 4:
                continue
            try:
                freq, real, imag = float(parts[0]), float(parts[1]), float(parts[2])
                freqs.append(freq)
                imags.append(imag)
                reals.append(real)
                mags.append(np.sqrt(real**2 + imag**2))
            except ValueError:
                continue
        temperatures.append(temp)
        frequencies.append(freqs)
        mag_vals.append(mags)
        imag_vals.append(imags)
        real_vals.append(reals)
    return temperatures, frequencies, mag_vals, imag_vals, real_vals

# ── Signal & Peaks pipeline ───────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_pipeline(file_contents, file_names, data_type, bad_temps,
                 min_freq, max_freq, min_temp, max_temp, temp_interval,
                 prominence, factor,
                 smooth_order, smooth_window):
    from concurrent.futures import ThreadPoolExecutor

    temperatures, frequencies, mag_vals, imag_vals, real_vals = read_raw_files(file_contents)

    # Remove bad temps
    keep = [i for i, t in enumerate(temperatures) if t not in bad_temps]
    temperatures = [temperatures[i] for i in keep]
    frequencies  = [frequencies[i]  for i in keep]
    mag_vals     = [mag_vals[i]      for i in keep]
    imag_vals    = [imag_vals[i]     for i in keep]
    real_vals    = [real_vals[i]     for i in keep]

    values = mag_vals if data_type == "mag" else (imag_vals if data_type == "imag" else real_vals)

    # Filter by frequency — vectorized with numpy
    filtered_freqs, filtered_vals = [], []
    for i in range(len(temperatures)):
        fa = np.array(frequencies[i]) * factor
        mask = (fa > min_freq) & (fa < max_freq)
        filtered_freqs.append(np.array(frequencies[i])[mask].tolist())
        filtered_vals.append(np.array(values[i])[mask].tolist())

    # Filter by temperature interval
    combined = sorted(zip(temperatures, filtered_freqs, filtered_vals), key=lambda x: x[0])
    temps, freqs_out, vals_out = [], [], []
    curr = -1e9
    for temp, ff, fv in combined:
        if min_temp < temp < max_temp and temp > curr + temp_interval:
            temps.append(temp); freqs_out.append(ff); vals_out.append(fv)
            curr = temp

    if not temps:
        return None

    # Smooth + normalize in parallel across temperatures
    def process_one(args):
        i, m, ff = args
        arr = np.array(m)
        if len(arr) >= smooth_window:
            arr = savgol_filter(arr, window_length=smooth_window, polyorder=smooth_order)

        ids, _ = find_peaks(arr, prominence=prominence)
        return i, arr.tolist(), [ff[j] for j in ids], [arr[j] for j in ids]

    n = len(temps)
    smoothed   = [None] * n
    peak_freqs = [None] * n
    peak_mags  = [None] * n
    valid_mask = [True] * n

    with ThreadPoolExecutor() as ex:
        for i, arr, pf, pm in ex.map(process_one, [(i, vals_out[i], freqs_out[i]) for i in range(n)]):
            smoothed[i]   = arr
            peak_freqs[i] = pf
            peak_mags[i]  = pm

    return {"temperatures": temps, "frequencies": freqs_out, "magnitudes": smoothed,
            "peak_frequencies": peak_freqs, "peak_magnitudes": peak_mags}

# ── Waterfall pipeline ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_waterfall_pipeline(file_contents, file_names, data_type, bad_temps,
                           wf_min_freq, wf_max_freq, wf_min_temp, wf_max_temp,
                           wf_temp_interval, wf_power, wf_peak_multiplier,
                           wf_smooth_order, wf_smooth_window, factor):

    temperatures, frequencies, mag_vals, imag_vals, real_vals = read_raw_files(file_contents)

    keep = [i for i, t in enumerate(temperatures) if t not in bad_temps]
    temperatures = [temperatures[i] for i in keep]
    frequencies  = [frequencies[i]  for i in keep]
    mag_vals     = [mag_vals[i]      for i in keep]
    imag_vals    = [imag_vals[i]     for i in keep]
    real_vals    = [real_vals[i]     for i in keep]

    values = mag_vals if data_type == "mag" else (imag_vals if data_type == "imag" else real_vals)

    # Temperature interval filter
    combined = sorted(zip(temperatures, frequencies, values), key=lambda x: x[0])
    temps, freqs_out, vals_out = [], [], []
    curr = -1e9
    for temp, ff, fv in combined:
        if wf_min_temp < temp < wf_max_temp and temp > curr + wf_temp_interval:
            temps.append(temp); freqs_out.append(ff); vals_out.append(fv)
            curr = temp

    if not temps:
        return None

    # Frequency filter — vectorized
    filtered_freqs, filtered_vals = [], []
    for i in range(len(temps)):
        fa = np.array(freqs_out[i]) * factor
        mask = (fa > wf_min_freq) & (fa < wf_max_freq)
        filtered_freqs.append(np.array(freqs_out[i])[mask].tolist())
        filtered_vals.append(np.array(vals_out[i])[mask].tolist())

    # Smooth + normalize in parallel
    from concurrent.futures import ThreadPoolExecutor
    def process_wf(args):
        i, m, ff = args
        arr = np.array(m)
        if len(arr) >= wf_smooth_window:
            arr = savgol_filter(arr, window_length=wf_smooth_window, polyorder=wf_smooth_order)
        if len(arr):
            v = arr ** wf_power
            mx = np.max(np.abs(v))
            if mx > 0:
                arr = (v / mx) * wf_peak_multiplier
        return i, arr.tolist(), ff

    n = len(temps)
    results_wf = [None] * n
    with ThreadPoolExecutor() as ex:
        for i, arr, ff in ex.map(process_wf, [(i, filtered_vals[i], filtered_freqs[i]) for i in range(n)]):
            results_wf[i] = (arr, ff)

    norm = [r[0] for r in results_wf if r[0]]
    nf   = [r[1] for r in results_wf if r[0]]
    nt   = [temps[i] for i in range(n) if results_wf[i][0]]

    return {"temperatures": nt, "frequencies": nf, "magnitudes": norm}

# ── Run pipelines ─────────────────────────────────────────────────────────────
file_contents = tuple(f.read() for f in uploaded_files)
file_names    = tuple(f.name for f in uploaded_files)

with st.spinner("Processing… (only re-runs when parameters change)"):
    result = run_pipeline(
        file_contents, file_names, data_type, bad_temps,
        min_freq, max_freq, min_temp, max_temp, temp_interval,
        prominence, factor,
        smooth_order, smooth_window
    )
    wf_result = run_waterfall_pipeline(
        file_contents, file_names, data_type, bad_temps,
        wf_min_freq, wf_max_freq, wf_min_temp, wf_max_temp,
        wf_temp_interval, wf_power, wf_peak_multiplier,
        wf_smooth_order, wf_smooth_window, factor
    )

if result is None:
    st.error("No Signal & Peaks data after filtering. Adjust the frequency/temperature ranges.")
    st.stop()

temperatures     = result["temperatures"]
frequencies      = result["frequencies"]
magnitudes       = result["magnitudes"]
peak_frequencies = result["peak_frequencies"]
peak_magnitudes  = result["peak_magnitudes"]

# ── Summary metrics ───────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Files loaded", len(uploaded_files))
col2.metric("Temperature points", len(temperatures))
col3.metric("Temp range", f"{min(temperatures):.1f}–{max(temperatures):.1f} K")
col4.metric("Max peaks found", max((len(p) for p in peak_frequencies), default=0))

st.divider()

# ── Session state for slider index ────────────────────────────────────────────
if "temp_idx" not in st.session_state:
    st.session_state.temp_idx = 0
    st.session_state.temp_slider = 0
if st.session_state.temp_idx >= len(temperatures):
    st.session_state.temp_idx = len(temperatures) - 1
    st.session_state.temp_slider = st.session_state.temp_idx

def _slider_changed():
    st.session_state.temp_idx = st.session_state.temp_slider

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Signal & Peaks", "🔵 Peak Tracking", "🌊 Waterfall", "💾 Export CSV"])

PLOTLY_LAYOUT = dict(
    xaxis_title="Frequency (kHz)",
    height=450,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=60, r=20, t=40, b=60),
)

# Tab 1 — Signal & Peaks with slider + arrow buttons
with tab1:
    st.subheader("Signal & peaks by temperature")

    # Buttons BEFORE slider so they update session state before slider renders
    sl_col, dn_col, up_col = st.columns([14, 1, 1])
    with dn_col:
        st.write("")
        if st.button("◀", help="Previous temperature"):
            st.session_state.temp_idx = max(0, st.session_state.temp_idx - 1)
            st.session_state.temp_slider = st.session_state.temp_idx
    with up_col:
        st.write("")
        if st.button("▶", help="Next temperature"):
            st.session_state.temp_idx = min(len(temperatures) - 1, st.session_state.temp_idx + 1)
            st.session_state.temp_slider = st.session_state.temp_idx
    with sl_col:
        st.slider("Temperature step", 0, len(temperatures) - 1,
                  key="temp_slider", on_change=_slider_changed)

    temp_idx = st.session_state.temp_idx
    chosen_temp = temperatures[temp_idx]
    st.caption(f"Temperature: **{chosen_temp:.2f} K**  ·  Peaks found: **{len(peak_frequencies[temp_idx])}**")

    fs = [f * factor for f in frequencies[temp_idx]]
    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=fs, y=magnitudes[temp_idx], mode="markers",
        marker=dict(size=4, color="#378ADD"), name="Signal",
        hovertemplate="Freq: %{x:.3f} kHz<br>Signal: %{y:.6f}<extra></extra>",
    ))
    if peak_frequencies[temp_idx]:
        fig1.add_trace(go.Scatter(
            x=[f * factor for f in peak_frequencies[temp_idx]],
            y=peak_magnitudes[temp_idx], mode="markers",
            marker=dict(size=12, color="#D85A30", symbol="circle"), name="Peaks",
            hovertemplate="Peak — Freq: %{x:.3f} kHz<br>Signal: %{y:.6f}<extra></extra>",
        ))
    fig1.update_layout(title=f"Temperature: {chosen_temp:.2f} K",
                       yaxis_title=f"{data_type.capitalize()} signal", **PLOTLY_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

# Tab 2 — Peak tracking
with tab2:
    st.subheader("Peak frequencies vs temperature")
    fig2 = go.Figure()
    all_temps, all_freqs, all_labels = [], [], []
    for i in range(len(temperatures)):
        for pf in peak_frequencies[i]:
            all_temps.append(temperatures[i])
            all_freqs.append(pf * factor)
            all_labels.append(f"T: {temperatures[i]:.2f} K<br>Freq: {pf * factor:.3f} kHz")
    fig2.add_trace(go.Scatter(
        x=all_temps, y=all_freqs, mode="markers",
        marker=dict(size=6, color="#1D9E75"), name="Peaks",
        hovertemplate="%{text}<extra></extra>", text=all_labels,
    ))
    fig2.update_layout(title="Peak frequencies vs temperature",
                       xaxis_title="Temperature (K)", yaxis_title="Frequency (kHz)",
                       hovermode="closest", height=500, margin=dict(l=60, r=20, t=40, b=60))
    st.plotly_chart(fig2, use_container_width=True)

# Tab 3 — Waterfall
with tab3:
    st.subheader("Waterfall plot")
    if wf_result is None:
        st.warning("No waterfall data after filtering. Adjust the Waterfall Parameters in the sidebar.")
    else:
        wf_temps = wf_result["temperatures"]
        wf_freqs = wf_result["frequencies"]
        wf_mags  = wf_result["magnitudes"]

        # Build gradient colors across all temperature steps
        COLOR_SCHEMES = {
            "Blue → Cyan":     ((0,51,102),   (0,204,255)),
            "Cyan → Blue":     ((0,204,255),  (0,51,102)),
            "Red → Blue":      ((180,0,0),    (0,60,180)),
            "Blue → Red":      ((0,60,180),   (180,0,0)),
            "Red → Yellow":    ((180,0,0),    (220,200,0)),
            "Yellow → Red":    ((220,200,0),  (180,0,0)),
            "Green → Blue":    ((0,130,80),   (0,60,180)),
            "Blue → Green":    ((0,60,180),   (0,130,80)),
            "Purple → Orange": ((120,0,160),  (220,120,0)),
            "Orange → Purple": ((220,120,0),  (120,0,160)),
        }
        c_start, c_end = COLOR_SCHEMES.get(wf_color_scheme, ((0,51,102),(0,204,255)))
        n_steps = max(len(wf_temps), 1)
        def lerp_color(t):
            r = int(c_start[0] + (c_end[0] - c_start[0]) * t)
            g = int(c_start[1] + (c_end[1] - c_start[1]) * t)
            b = int(c_start[2] + (c_end[2] - c_start[2]) * t)
            return f"rgb({r},{g},{b})"

        fig3 = go.Figure()
        for i in range(len(wf_temps)):
            col = lerp_color(i / (n_steps - 1) if n_steps > 1 else 0)
            xs = [f * factor for f in wf_freqs[i]]
            ys = [v + i for v in wf_mags[i]]
            fig3.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(color=col, width=1.5),
                name=f"{wf_temps[i]:.1f} K",
                hovertemplate=f"T: {wf_temps[i]:.2f} K<br>Freq: %{{x:.3f}} kHz<br>Signal: %{{y:.4f}}<extra></extra>",
            ))

        # Y-axis ticks at each temperature
        fig3.update_layout(
            title="Waterfall — Signal vs Frequency by Temperature",
            xaxis_title="Frequency (kHz)",
            yaxis=dict(
                tickvals=list(range(len(wf_temps))),
                ticktext=[f"{t:.1f}" for t in wf_temps],
                title="Temperature (K)",
            ),
            xaxis=dict(range=[wf_min_freq, wf_max_freq]),
            height=max(500, len(wf_temps) * 22),
            hovermode="closest",
            showlegend=False,
            margin=dict(l=80, r=20, t=40, b=60),
        )
        st.plotly_chart(fig3, use_container_width=True)

# Tab 4 — CSV export
with tab4:
    st.subheader("Export peak data")
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    n_peaks = max((len(p) for p in peak_frequencies), default=0)
    writer.writerow(["Temperature (K)"] + [f"Peak {i} freq (kHz)" for i in range(n_peaks)])
    for i in range(len(temperatures)):
        writer.writerow([temperatures[i]] + [pf * factor for pf in peak_frequencies[i]])
    st.download_button(
        "⬇️ Download peak_tracking.csv",
        data=csv_buf.getvalue(),
        file_name="peak_tracking.csv",
        mime="text/csv",
    )
