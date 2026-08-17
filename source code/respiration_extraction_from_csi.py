import csv
import os
import re
import zipfile
from itertools import combinations
from xml.sax.saxutils import escape

os.environ.setdefault("MPLCONFIGDIR", os.path.join("/tmp", "matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, medfilt, savgol_filter, spectrogram, welch

DATA_DIRS = [

]

SAVE_DIR = "RES_figures"
os.makedirs(SAVE_DIR, exist_ok=True)

NUM_ANTENNAS = 4
DEFAULT_FS = 50
fs = DEFAULT_FS
low, high = 0.1, 0.8

corr_window_sec = 5
corr_threshold = 0.1

TOP_K_SUBCARRIERS = 144

PSD_WINDOW_SEC = 20

MIN_RESP_SNR_DB = 3.0
GOOD_RESP_SNR_DB = 10.0
MIN_PEAK_RATIO = 0.10
PAIR_FREQ_TOL_HZ = 0.05
UNRELIABLE_QUALITY = 0.45
RELIABLE_QUALITY = 0.75
SIGNAL_SKIP_THRESHOLD = 0.35

MIN_COMPLETENESS = 0.55
GOOD_COMPLETENESS = 0.95
MIN_RAW_AMP_DB = -55.0
GOOD_RAW_AMP_DB = -40.0

MIN_CYCLES_OBSERVED = 20.0
GOOD_CYCLES_OBSERVED = 40.0

# Dataset-level ground truth for offline evaluation only.
EVAL_GROUND_TRUTH_BREATH_COUNT = 50
GOOD_COUNT_ERROR = 5
BAD_COUNT_ERROR = 8

# All 6 pairs from 4 antennas: (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)
# Geometrically: (0,1),(2,3) = horizontal sides
#                (0,2),(1,3) = vertical sides
#                (0,3),(1,2) = diagonals
ANTENNA_PAIRS = list(combinations(range(NUM_ANTENNAS), 2))
NUM_PAIRS = len(ANTENNA_PAIRS)  # = 6

b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
corr_window = int(corr_window_sec * fs)


def output_stem(data_dir):
    parts = os.path.normpath(data_dir).split(os.sep)
    if "outdoor" in parts:
        parts = parts[parts.index("outdoor") + 1:]
    safe = "_".join(parts) if parts else os.path.basename(os.path.normpath(data_dir))
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", safe)
    return safe.strip("_") or "dataset"


def normalize_signal(signal):
    return (signal - np.mean(signal)) / (np.std(signal) + 1e-9)


def clamp01(value):
    return float(np.clip(value, 0.0, 1.0))


def count_accuracy_score(error_count):
    if not np.isfinite(error_count):
        return 0.0
    return clamp01((BAD_COUNT_ERROR - error_count) / (BAD_COUNT_ERROR - GOOD_COUNT_ERROR))


def odd_window(samples, max_len, min_len=5):
    samples = int(samples)
    if samples % 2 == 0:
        samples += 1
    samples = max(samples, min_len)
    if samples > max_len:
        samples = max_len if max_len % 2 == 1 else max_len - 1
    return max(samples, min_len)

def parabolic_peak_freq(freqs, power, peak_idx):
    if peak_idx <= 0 or peak_idx >= len(freqs) - 1:
        return float(freqs[peak_idx])
    y0, y1, y2 = (np.log(power[peak_idx + d] + 1e-300) for d in (-1, 0, 1))
    denom = y0 - 2.0 * y1 + y2
    if denom == 0:
        return float(freqs[peak_idx])
    delta = np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0)
    df = freqs[peak_idx + 1] - freqs[peak_idx]
    return float(freqs[peak_idx] + delta * df)


def estimate_breath_freq(signal, fs):
    nperseg = min(len(signal), fs * PSD_WINDOW_SEC)
    noverlap = min(nperseg // 2, fs * 10)
    f, Pxx = welch(signal, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=8192)
    mask = (f >= 0.1) & (f <= 0.8)
    band_f, band_p = f[mask], Pxx[mask]
    peak_freq = parabolic_peak_freq(band_f, band_p, int(np.argmax(band_p)))
    return peak_freq, f, Pxx


def psd_quality_metrics(signal, fs):
    sig = normalize_signal(signal)
    nperseg = min(len(sig), fs * PSD_WINDOW_SEC)
    noverlap = nperseg // 2
    f, Pxx = welch(sig, fs=fs, nperseg=nperseg, noverlap=noverlap, nfft=8192)

    resp_mask = (f >= low) & (f <= high)
    noise_mask = ((f >= 0.02) & (f < low)) | ((f > high) & (f <= 2.0))

    if not np.any(resp_mask) or not np.any(noise_mask):
        return {
            "peak_freq": np.nan,
            "snr_db": -np.inf,
            "peak_ratio": 0.0,
            "snr_score": 0.0,
            "peak_score": 0.0,
            "f": f,
            "Pxx": Pxx,
        }

    resp_power = Pxx[resp_mask]
    resp_freqs = f[resp_mask]
    peak_idx = int(np.argmax(resp_power))
    peak_freq = parabolic_peak_freq(resp_freqs, resp_power, peak_idx)
    peak_power = float(resp_power[peak_idx])
    noise_floor = float(np.median(Pxx[noise_mask]))
    band_power = float(np.sum(resp_power))
    peak_band = resp_mask & (np.abs(f - peak_freq) <= 0.03)
    peak_band_power = float(np.sum(Pxx[peak_band]))

    snr_db = 10.0 * np.log10((peak_power + 1e-12) / (noise_floor + 1e-12))
    peak_ratio = peak_band_power / (band_power + 1e-12)

    return {
        "peak_freq": peak_freq,
        "snr_db": float(snr_db),
        "peak_ratio": float(peak_ratio),
        "snr_score": clamp01((snr_db - MIN_RESP_SNR_DB) / (GOOD_RESP_SNR_DB - MIN_RESP_SNR_DB)),
        "peak_score": clamp01((peak_ratio - MIN_PEAK_RATIO) / 0.20),
        "f": f,
        "Pxx": Pxx,
    }


def subcarrier_resp_scores(matrix, fs):
    x = matrix - np.mean(matrix, axis=0, keepdims=True)
    nperseg = min(x.shape[0], fs * 20)
    noverlap = nperseg // 2
    f, Pxx = welch(x, fs=fs, axis=0, nperseg=nperseg, noverlap=noverlap, nfft=2048)

    resp_mask = (f >= low) & (f <= high)
    noise_mask = ((f >= 0.02) & (f < low)) | ((f > high) & (f <= 2.0))
    peak_power = np.max(Pxx[resp_mask, :], axis=0)
    noise_floor = np.median(Pxx[noise_mask, :], axis=0) + 1e-12
    return peak_power / noise_floor


def amp_phase_corr_score(amp, phase):
    amp_norm = normalize_signal(amp)
    phase_norm = normalize_signal(phase)
    if len(amp_norm) < corr_window:
        corr = np.corrcoef(amp_norm, phase_norm)[0, 1]
        return float(np.nan_to_num(abs(corr)))

    half_w = corr_window // 2
    corr_values = []
    for i in range(half_w, len(amp_norm) - half_w, max(1, half_w)):
        a_seg = amp_norm[i - half_w: i + half_w]
        p_seg = phase_norm[i - half_w: i + half_w]
        corr_values.append(abs(np.corrcoef(a_seg, p_seg)[0, 1]))
    return float(np.nan_to_num(np.median(corr_values), nan=0.0))


def pair_quality_metrics(pair_amp, pair_phase):
    metrics = []
    for amp, phase in zip(pair_amp, pair_phase):
        amp_metrics = psd_quality_metrics(amp, fs)
        phase_metrics = psd_quality_metrics(phase, fs)
        best_metrics = amp_metrics if amp_metrics["snr_db"] >= phase_metrics["snr_db"] else phase_metrics
        corr_score = amp_phase_corr_score(amp, phase)
        quality = (
            0.65 * best_metrics["snr_score"]
            + 0.25 * best_metrics["peak_score"]
            + 0.10 * corr_score
        )
        metrics.append({
            **best_metrics,
            "corr_score": corr_score,
            "quality": float(quality),
            "source": "amp" if best_metrics is amp_metrics else "phase",
        })
    return metrics


def robust_global_frequency(pair_metrics):
    good = [
        m for m in pair_metrics
        if m["snr_db"] >= MIN_RESP_SNR_DB and m["peak_ratio"] >= MIN_PEAK_RATIO
    ]
    candidates = good if good else pair_metrics
    weights = np.array([max(m["quality"], 1e-3) for m in candidates])
    freqs = np.array([m["peak_freq"] for m in candidates])
    order = np.argsort(freqs)
    freqs = freqs[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    f_est = float(freqs[np.searchsorted(cdf, 0.5)])

    inlier_count = sum(abs(m["peak_freq"] - f_est) <= PAIR_FREQ_TOL_HZ for m in pair_metrics)
    consistency_score = inlier_count / NUM_PAIRS
    return f_est, consistency_score, inlier_count


def temporal_stability_score(signal, fs):
    window = fs * 20
    step = fs * 10
    if len(signal) < window:
        return 0.5, np.nan, 1.0

    freqs = []
    valid_windows = 0
    for start in range(0, len(signal) - window + 1, step):
        seg = signal[start:start + window]
        metrics = psd_quality_metrics(seg, fs)
        if metrics["snr_db"] >= MIN_RESP_SNR_DB and metrics["peak_ratio"] >= MIN_PEAK_RATIO:
            valid_windows += 1
            freqs.append(metrics["peak_freq"])

    total_windows = max(1, 1 + (len(signal) - window) // step)
    valid_ratio = valid_windows / total_windows
    if len(freqs) < 2:
        freq_std_bpm = np.nan
        stability = 0.35 * valid_ratio
    else:
        freq_std_bpm = float(np.std(freqs) * 60.0)
        stability = valid_ratio * clamp01(1.0 - freq_std_bpm / 6.0)
    return float(stability), freq_std_bpm, float(valid_ratio)


def attention_from_quality(pair_metrics):
    scores = np.array([m.get("effective_quality", m["quality"]) for m in pair_metrics], dtype=float)
    if np.all(scores <= 1e-6):
        return np.ones(NUM_PAIRS) / NUM_PAIRS
    scores = 4.0 * scores
    weights = np.exp(scores - scores.max())
    weights /= weights.sum()
    return weights


def reliability_label(quality):
    if quality >= RELIABLE_QUALITY:
        return "reliable"
    if quality >= UNRELIABLE_QUALITY:
        return "weak"
    return "unreliable"


def excel_col_name(index):
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def xlsx_cell_xml(row_idx, col_idx, value, style_id=None):
    ref = f"{excel_col_name(col_idx)}{row_idx}"
    style = f' s="{style_id}"' if style_id is not None else ""

    if value is None:
        return f'<c r="{ref}"{style}/>'

    if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(value):
        return f'<c r="{ref}"{style}><v>{float(value):.12g}</v></c>'

    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"{style}><is><t>{text}</t></is></c>'


def write_styled_xlsx(rows, fieldnames, path):
    sheet_rows = []

    header_cells = [
        xlsx_cell_xml(1, col_idx, fieldname, style_id=2)
        for col_idx, fieldname in enumerate(fieldnames)
    ]
    sheet_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_offset, row in enumerate(rows, start=2):
        style_id = 1 if row.get("reliability") == "weak" else None
        cells = [
            xlsx_cell_xml(row_offset, col_idx, row.get(fieldname, ""), style_id=style_id)
            for col_idx, fieldname in enumerate(fieldnames)
        ]
        sheet_rows.append(f'<row r="{row_offset}">{"".join(cells)}</row>')

    last_col = excel_col_name(len(fieldnames) - 1)
    dimension = f"A1:{last_col}{max(1, len(rows) + 1)}"
    cols = "".join(
        f'<col min="{idx + 1}" max="{idx + 1}" width="{max(10, min(28, len(fieldname) + 2))}" customWidth="1"/>'
        for idx, fieldname in enumerate(fieldnames)
    )
    worksheet = f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <dimension ref="{dimension}"/>
  <sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/><selection pane="bottomLeft"/></sheetView></sheetViews>
  <cols>{cols}</cols>
  <sheetData>{"".join(sheet_rows)}</sheetData>
  <autoFilter ref="{dimension}"/>
</worksheet>'''

    workbook = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="summary" sheetId="1" r:id="rId1"/></sheets>
</workbook>'''
    workbook_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>'''
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>'''
    styles = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2"><font><sz val="11"/><name val="Calibri"/></font><font><b/><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="3"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill><fill><patternFill patternType="solid"><fgColor rgb="FFD9D9D9"/><bgColor indexed="64"/></patternFill></fill></fills>
  <borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="3"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/><xf numFmtId="0" fontId="0" fillId="2" borderId="0" xfId="0" applyFill="1"/><xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>
  <cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles>
</styleSheet>'''

    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as xlsx:
        xlsx.writestr("[Content_Types].xml", content_types)
        xlsx.writestr("_rels/.rels", root_rels)
        xlsx.writestr("xl/workbook.xml", workbook)
        xlsx.writestr("xl/_rels/workbook.xml.rels", workbook_rels)
        xlsx.writestr("xl/worksheets/sheet1.xml", worksheet)
        xlsx.writestr("xl/styles.xml", styles)

# extract amp+phase respiration signal for one antenna pair
def extract_pair_signal(H_i, H_j):
    ratio = np.abs(H_i) / (np.abs(H_j) + 1e-6)

    amp_scores = subcarrier_resp_scores(ratio, fs)
    num_good_amp = min(TOP_K_SUBCARRIERS, ratio.shape[1])
    good_idx_amp = np.argsort(amp_scores)[-num_good_amp:]
    ratio = ratio[:, good_idx_amp]

    ratio_mean = np.mean(ratio, axis=1)
    ratio_mean = medfilt(ratio_mean, 9)
    amp_resp = filtfilt(b, a, ratio_mean)

    phase_diff = np.unwrap(np.angle(H_i) - np.angle(H_j), axis=0)
    phase_scores = subcarrier_resp_scores(phase_diff, fs)
    num_good_phase = min(TOP_K_SUBCARRIERS, phase_diff.shape[1])
    good_idx_phase = np.argsort(phase_scores)[-num_good_phase:]
    phase_diff = phase_diff[:, good_idx_phase]

    phase_mean = np.mean(phase_diff, axis=1)
    phase_mean -= np.mean(phase_mean)
    phase_mean = medfilt(phase_mean, 9)
    phase_resp = filtfilt(b, a, phase_mean)

    return amp_resp, phase_resp

# fuse amp + phase for a single pair using sliding correlation
def fuse_amp_phase(amp, phase, T_period, fs):
    amp_norm = (amp - np.mean(amp)) / (np.std(amp) + 1e-9)
    phase_norm = (phase - np.mean(phase)) / (np.std(phase) + 1e-9)

    ma_window = odd_window(0.4 * T_period * fs, len(amp_norm))

    amp_s = savgol_filter(amp_norm, ma_window, 3)
    phase_s = savgol_filter(phase_norm, ma_window, 3)

    half_w = corr_window // 2
    corr_series = np.zeros(len(amp_s))

    for i in range(half_w, len(amp_s) - half_w):
        a_seg = amp_s[i - half_w: i + half_w]
        p_seg = phase_s[i - half_w: i + half_w]
        corr_series[i] = np.corrcoef(a_seg, p_seg)[0, 1]

    fusion = np.zeros(len(amp_s))
    eps = 1e-6

    for i in range(len(amp_s)):
        corr_w = abs(corr_series[i])
        if corr_w > corr_threshold:
            Ea = abs(amp_s[i])
            Ep = abs(phase_s[i])
            total = Ea + Ep + eps
            fusion[i] = corr_w * ((Ea / total) * amp_s[i] + (Ep / total) * phase_s[i])
        else:
            fusion[i] = 0

    fusion = savgol_filter(fusion, ma_window, 3)
    fusion = (fusion - np.mean(fusion)) / (np.std(fusion) + 1e-9)
    return fusion


# spatial attention weights over pairs
def spatial_attention(pair_signals, pair_metrics=None):
    if pair_metrics is not None:
        return attention_from_quality(pair_metrics)

    scores = np.array([np.var(sig) for sig in pair_signals])  # [NUM_PAIRS]
    weights = np.exp(scores - scores.max())                   # softmax (numerically stable)
    weights /= weights.sum()
    return weights


def analyze_raw_recording(data_dir):
    per_file_fs = []
    amp_sum = 0.0
    amp_sq_sum = 0.0
    amp_count = 0
    n_frames_total = 0
    all_min_ts = []
    all_max_ts = []

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue
        data = np.load(os.path.join(data_dir, fname), allow_pickle=True)
        ts = data["timestamp"]
        csi = data["csi"]

        n_frames = len(ts) // NUM_ANTENNAS
        if n_frames < 2:
            continue

        tmin, tmax = float(ts.min()), float(ts.max())
        file_span_sec = (tmax - tmin) / 1e9
        if file_span_sec > 0:
            per_file_fs.append(n_frames / file_span_sec)
        all_min_ts.append(tmin)
        all_max_ts.append(tmax)
        n_frames_total += n_frames

        mag = np.abs(csi).astype(np.float64)
        amp_sum += mag.sum()
        amp_sq_sum += np.square(mag).sum()
        amp_count += mag.size

    if not per_file_fs or amp_count == 0:
        raise RuntimeError(f"No usable npz files found in {data_dir}")

    real_fs = float(np.median(per_file_fs))
    span_sec = float((max(all_max_ts) - min(all_min_ts)) / 1e9)
    expected_frames = span_sec * real_fs
    completeness = clamp01(n_frames_total / expected_frames) if expected_frames > 0 else 0.0

    mean_amp = amp_sum / amp_count
    mean_sq_amp = amp_sq_sum / amp_count
    raw_amp_db = float(20.0 * np.log10(mean_amp + 1e-12))
    raw_amp_cv = float(np.sqrt(max(mean_sq_amp - mean_amp ** 2, 0.0)) / (mean_amp + 1e-12))

    return {
        "real_fs": float(real_fs),
        "completeness": float(completeness),
        "raw_amp_db": raw_amp_db,
        "raw_amp_cv": raw_amp_cv,
        "span_sec": span_sec,
        "n_frames": n_frames_total,
    }


def load_pair_amp_phase(data_dir):
    all_pair_amp = [[] for _ in range(NUM_PAIRS)]
    all_pair_phase = [[] for _ in range(NUM_PAIRS)]

    for fname in sorted(os.listdir(data_dir)):
        if not fname.endswith(".npz"):
            continue

        path = os.path.join(data_dir, fname)
        data = np.load(path, allow_pickle=True)
        csi = data["csi"]

        num_packets, num_subcarriers = csi.shape
        num_frames = num_packets // NUM_ANTENNAS

        if num_frames < fs * 2:
            continue

        csi_reshaped = np.zeros((num_frames, NUM_ANTENNAS, num_subcarriers), dtype=np.complex64)
        for t in range(num_frames):
            for ant in range(NUM_ANTENNAS):
                csi_reshaped[t, ant, :] = csi[t * NUM_ANTENNAS + ant]

        for k, (i, j) in enumerate(ANTENNA_PAIRS):
            H_i = csi_reshaped[:, i, :]
            H_j = csi_reshaped[:, j, :]
            amp_resp, phase_resp = extract_pair_signal(H_i, H_j)
            all_pair_amp[k].append(amp_resp)
            all_pair_phase[k].append(phase_resp)

    if any(not all_pair_amp[k] or not all_pair_phase[k] for k in range(NUM_PAIRS)):
        raise RuntimeError(f"No usable npz files found in {data_dir}")

    pair_amp = [np.concatenate(all_pair_amp[k]) for k in range(NUM_PAIRS)]
    pair_phase = [np.concatenate(all_pair_phase[k]) for k in range(NUM_PAIRS)]

    return pair_amp, pair_phase


def plot_result(
    save_path,
    stem,
    time_axis,
    pair_fused_trimmed,
    attn_weights,
    baseline_signal,
    fusion_final,
    peaks,
    f_psd,
    P_psd,
    f_est,
    f_fusion,
    P_fusion,
    f_spec,
    t_spec,
    Sxx,
    signal_quality_score,
    confidence_score,
    reliability,
):
    plt.rcParams.update({"font.size": 13, "lines.linewidth": 1.8})

    pair_labels = [f"({i},{j})" for i, j in ANTENNA_PAIRS]

    fig = plt.figure(figsize=(20, 24))
    fig.suptitle(
        f"Multi-antenna 5G CSI Respiration Sensing | {stem} | "
        f"signal={signal_quality_score:.2f}, confidence={confidence_score:.2f} ({reliability})",
        fontsize=15,
        y=0.98,
    )

    for k in range(NUM_PAIRS):
        ax = fig.add_subplot(5, 3, k + 1)
        ax.plot(time_axis, pair_fused_trimmed[k], linewidth=1.2)
        ax.set_title(f"Pair {pair_labels[k]}  (w={attn_weights[k]:.3f})", fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.grid(True, alpha=0.4)

    ax_attn = fig.add_subplot(5, 1, 3)
    colors = plt.cm.viridis(attn_weights / attn_weights.max())
    bars = ax_attn.bar(pair_labels, attn_weights, color=colors)
    ax_attn.set_title("Spatial attention weights over antenna pairs")
    ax_attn.set_ylabel("Weight (softmax)")
    ax_attn.grid(True, axis="y", alpha=0.4)
    for bar, w in zip(bars, attn_weights):
        ax_attn.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.002,
            f"{w:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    ax_cmp = fig.add_subplot(5, 1, 4)
    ax_cmp.plot(time_axis, baseline_signal, label="Baseline: pair (0,1) only", alpha=0.6, linewidth=1.2)
    ax_cmp.plot(time_axis, fusion_final, label="Spatial attention fusion (6 pairs)", linewidth=1.8)
    ax_cmp.plot(time_axis[peaks], fusion_final[peaks], "rx", markersize=8, label="Detected peaks")
    ax_cmp.set_title("Baseline vs. spatial attention fusion")
    ax_cmp.set_xlabel("Time (s)")
    ax_cmp.legend()
    ax_cmp.grid(True, alpha=0.4)

    ax_psd = fig.add_subplot(5, 2, 9)
    ax_psd.plot(f_psd, P_psd, label="Pair (0,1) ref", alpha=0.7)
    ax_psd.plot(f_fusion, P_fusion, label="Fused signal", linewidth=1.8)
    ax_psd.axvline(f_est, color="r", linestyle="--", label=f"f_est={f_est * 60:.1f} bpm")
    ax_psd.set_xlim(0, 1)
    ax_psd.set_title("PSD comparison")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.legend(fontsize=10)
    ax_psd.grid(True, alpha=0.4)

    ax_spec = fig.add_subplot(5, 2, 10)
    im = ax_spec.pcolormesh(t_spec, f_spec, 10 * np.log10(Sxx + 1e-12), shading="gouraud", cmap="inferno")
    ax_spec.set_ylim(0, 1)
    ax_spec.set_title("Spectrogram (fused signal, dB)")
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=ax_spec, label="Power (dB)")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def process_one_dir(data_dir):
    global fs, b, a, corr_window

    print(f"\n[Processing] {data_dir}", flush=True)
    stem = output_stem(data_dir)

    raw_info = analyze_raw_recording(data_dir)

    fs = int(round(raw_info["real_fs"]))
    b, a = butter(4, [low / (fs / 2), high / (fs / 2)], btype="band")
    corr_window = int(corr_window_sec * fs)

    completeness_score = clamp01(
        (raw_info["completeness"] - MIN_COMPLETENESS) / (GOOD_COMPLETENESS - MIN_COMPLETENESS)
    )
    raw_amp_score = clamp01(
        (raw_info["raw_amp_db"] - MIN_RAW_AMP_DB) / (GOOD_RAW_AMP_DB - MIN_RAW_AMP_DB)
    )

    print(
        f"Raw recording: real_fs={raw_info['real_fs']:.1f} Hz (used fs={fs}), "
        f"completeness={raw_info['completeness']:.3f}, "
        f"raw_amp_db={raw_info['raw_amp_db']:.1f}, "
        f"span={raw_info['span_sec']:.1f}s, n_frames={raw_info['n_frames']}",
        flush=True,
    )

    pair_amp, pair_phase = load_pair_amp_phase(data_dir)

    # Quality gate before peak counting
 
    pair_metrics = pair_quality_metrics(pair_amp, pair_phase)
    f_est, consistency_score, consistency_count = robust_global_frequency(pair_metrics)
    T_period = 1.0 / max(f_est, 1e-6)
    for m in pair_metrics:
        freq_match = abs(m["peak_freq"] - f_est) <= PAIR_FREQ_TOL_HZ
        m["freq_match"] = float(freq_match)
        m["effective_quality"] = m["quality"] * (1.0 if freq_match else 0.40)

    amp_norm_ref = (pair_amp[0] - np.mean(pair_amp[0])) / (np.std(pair_amp[0]) + 1e-9)
    _, f_psd, P_psd = estimate_breath_freq(amp_norm_ref, fs)

    print(f"Estimated breathing frequency: {f_est * 60:.1f} bpm", flush=True)
    print(f"Estimated breathing period:    {T_period:.2f} sec", flush=True)
    print(f"Antenna pairs used: {ANTENNA_PAIRS}", flush=True)
    print("Pre-extraction pair quality:", flush=True)
    for k, (i, j) in enumerate(ANTENNA_PAIRS):
        m = pair_metrics[k]
        print(
            f"  Pair ({i},{j}) [{m['source']:>5}]: "
            f"f={m['peak_freq'] * 60:5.1f} bpm, "
            f"SNR={m['snr_db']:5.1f} dB, "
            f"peak_ratio={m['peak_ratio']:.3f}, "
            f"corr={m['corr_score']:.3f}, "
            f"q={m['quality']:.3f}, "
            f"match={int(m['freq_match'])}",
            flush=True,
        )
    print(
        f"Pair consistency: {consistency_count}/{NUM_PAIRS} within "
        f"{PAIR_FREQ_TOL_HZ * 60:.1f} bpm",
        flush=True,
    )

    pair_fused = []
    pair_variances = []
    for k, (i, j) in enumerate(ANTENNA_PAIRS):
        fused_k = fuse_amp_phase(pair_amp[k], pair_phase[k], T_period, fs)
        pair_fused.append(fused_k)
        pair_variances.append(np.var(fused_k))
        print(f"  Pair ({i},{j}) signal variance: {np.var(fused_k):.4f}", flush=True)

    attn_weights = spatial_attention(pair_fused, pair_metrics)

    print("Spatial attention weights:", flush=True)
    for k, (i, j) in enumerate(ANTENNA_PAIRS):
        baseline = "side" if abs(i - j) == 1 or {i, j} in ({0, 2}, {1, 3}) else "diagonal"
        print(f"  Pair ({i},{j}) [{baseline:>8}]: {attn_weights[k]:.4f}", flush=True)

    min_len = min(len(sig) for sig in pair_fused)
    pair_fused_trimmed = [sig[:min_len] for sig in pair_fused]

    fusion_final = sum(w * sig for w, sig in zip(attn_weights, pair_fused_trimmed))
    fusion_final = (fusion_final - np.mean(fusion_final)) / (np.std(fusion_final) + 1e-9)

    time_axis = np.arange(min_len) / fs
    duration_sec = min_len / fs

    fusion_metrics = psd_quality_metrics(fusion_final, fs)
    temporal_score, temporal_freq_std_bpm, valid_window_ratio = temporal_stability_score(fusion_final, fs)
    pair_quality_score = float(np.mean([m["quality"] for m in pair_metrics]))

    cycles_observed = duration_sec * f_est
    cycles_score = clamp01((cycles_observed - MIN_CYCLES_OBSERVED) / (GOOD_CYCLES_OBSERVED - MIN_CYCLES_OBSERVED))

    signal_quality_score = (
        0.15 * fusion_metrics["snr_score"]
        + 0.10 * fusion_metrics["peak_score"]
        + 0.10 * consistency_score
        + 0.10 * temporal_score
        + 0.05 * pair_quality_score
        + 0.10 * completeness_score
        + 0.05 * raw_amp_score
        + 0.35 * cycles_score
    )

    baseline_signal = pair_fused_trimmed[0]  # pair (0,1)

    min_distance = int(0.5 * T_period * fs)

    if signal_quality_score < SIGNAL_SKIP_THRESHOLD:
        peaks = np.array([], dtype=int)
        peak_properties = {}
        print("Quality gate: very low signal quality, skip peak counting.", flush=True)
    else:
        peaks, peak_properties = find_peaks(
            fusion_final,
            distance=min_distance,
            prominence=0.3,
            width=int(0.2 * T_period * fs),
        )

    breath_count = len(peaks)
    breath_rate = breath_count / duration_sec * 60 if breath_count > 0 else np.nan
    spectral_count = fusion_metrics["peak_freq"] * duration_sec
    count_spectral_error = breath_count - spectral_count
    count_spectral_score = count_accuracy_score(abs(count_spectral_error))
    if breath_count >= 3:
        peak_intervals = np.diff(peaks) / fs
        interval_cv = float(np.std(peak_intervals) / (np.mean(peak_intervals) + 1e-9))
        interval_score = clamp01(1.0 - interval_cv / 0.35)
    else:
        interval_cv = np.nan
        interval_score = 0.0
    if "prominences" in peak_properties and len(peak_properties["prominences"]) > 0:
        median_prominence = float(np.median(peak_properties["prominences"]))
        prominence_score = clamp01((median_prominence - 0.25) / 0.55)
    else:
        median_prominence = np.nan
        prominence_score = 0.0

    confidence_score = (
        0.65 * signal_quality_score
        + 0.15 * count_spectral_score
        + 0.12 * interval_score
        + 0.08 * prominence_score
    )
    reliability = reliability_label(confidence_score)

    eval_gt_expected_bpm = EVAL_GROUND_TRUTH_BREATH_COUNT / duration_sec * 60.0
    error_bpm = abs(breath_rate - eval_gt_expected_bpm)
    eval_psd_count_error = spectral_count - EVAL_GROUND_TRUTH_BREATH_COUNT
    eval_gt_count_error = breath_count - EVAL_GROUND_TRUTH_BREATH_COUNT
    eval_gt_count_score = count_accuracy_score(abs(eval_gt_count_error))

    print("========== Multi-antenna fusion results ==========", flush=True)
    print(f"Duration:           {duration_sec:.1f} sec", flush=True)
    print(f"Fusion SNR:         {fusion_metrics['snr_db']:.1f} dB", flush=True)
    print(f"Fusion peak ratio:  {fusion_metrics['peak_ratio']:.3f}", flush=True)
    print(f"Valid win ratio:    {valid_window_ratio:.2f}", flush=True)
    print(f"Temporal std:       {temporal_freq_std_bpm:.2f} bpm", flush=True)
    print(f"Cycles needed:      min={MIN_CYCLES_OBSERVED:.1f}, good={GOOD_CYCLES_OBSERVED:.1f} "
          f"(observed={cycles_observed:.1f})", flush=True)
    print(f"Spectral count:     {spectral_count:.1f}", flush=True)
    print(f"Peak/PSD error:     {count_spectral_error:.1f}", flush=True)
    print(f"Peak interval CV:   {interval_cv:.3f}", flush=True)
    print(f"Median prominence:  {median_prominence:.3f}", flush=True)
    print(f"Signal quality:     {signal_quality_score:.3f}", flush=True)
    print(f"Confidence score:   {confidence_score:.3f} ({reliability})", flush=True)
    print(f"Eval GT count err:  {eval_gt_count_error}", flush=True)
    print(f"Breath count:       {breath_count}", flush=True)
    print(f"Estimated BPM:      {breath_rate:.1f}", flush=True)
    print(f"Error BPM:          {error_bpm:.1f}", flush=True)

    f_fusion, P_fusion = welch(fusion_final, fs, nperseg=fs * 10)
    f_spec, t_spec, Sxx = spectrogram(fusion_final, fs=fs, nperseg=fs * 10)

    save_path = os.path.join(SAVE_DIR, f"multi_antenna_fusion_{stem}.png")
    plot_result(
        save_path,
        stem,
        time_axis,
        pair_fused_trimmed,
        attn_weights,
        baseline_signal,
        fusion_final,
        peaks,
        f_psd,
        P_psd,
        f_est,
        f_fusion,
        P_fusion,
        f_spec,
        t_spec,
        Sxx,
        signal_quality_score,
        confidence_score,
        reliability,
    )

    print(f"Figure saved to {save_path}", flush=True)

    return {
        "stem": stem,
        "duration_sec": duration_sec,
        "f_est_bpm": f_est * 60,
        "T_period_sec": T_period,
        "breath_count": breath_count,
        "estimated_bpm": breath_rate,
        "error_bpm": error_bpm,
        "signal_quality_score": signal_quality_score,
        "confidence_score": confidence_score,
        "reliability": reliability,
        "spectral_count": spectral_count,
        "count_spectral_error": count_spectral_error,
        "count_spectral_score": count_spectral_score,
        "peak_interval_cv": interval_cv,
        "peak_interval_score": interval_score,
        "median_prominence": median_prominence,
        "prominence_score": prominence_score,
        "eval_gt_expected_bpm": eval_gt_expected_bpm,
        "eval_psd_count_error": eval_psd_count_error,
        "eval_gt_count_error": eval_gt_count_error,
        "eval_gt_count_score": eval_gt_count_score,
        "fusion_snr_db": fusion_metrics["snr_db"],
        "fusion_peak_ratio": fusion_metrics["peak_ratio"],
        "pair_consistency": consistency_score,
        "valid_window_ratio": valid_window_ratio,
        "temporal_freq_std_bpm": temporal_freq_std_bpm,
        "real_fs_hz": raw_info["real_fs"],
        "used_fs_hz": fs,
        "completeness": raw_info["completeness"],
        "completeness_score": completeness_score,
        "raw_amp_db": raw_info["raw_amp_db"],
        "raw_amp_score": raw_amp_score,
        "cycles_observed": cycles_observed,
        "cycles_score": cycles_score,
        "min_cycles_observed": MIN_CYCLES_OBSERVED,
        "good_cycles_observed": GOOD_CYCLES_OBSERVED,
        # "attention_weights": ";".join(f"{w:.6f}" for w in attn_weights),
        # "pair_variances": ";".join(f"{v:.6f}" for v in pair_variances),
        # "pair_quality": ";".join(f"{m['quality']:.6f}" for m in pair_metrics),
        # "pair_effective_quality": ";".join(f"{m['effective_quality']:.6f}" for m in pair_metrics),
        # "pair_snr_db": ";".join(f"{m['snr_db']:.6f}" for m in pair_metrics),
        # "pair_peak_bpm": ";".join(f"{m['peak_freq'] * 60:.6f}" for m in pair_metrics),
    }


def write_summary(rows):
    path = os.path.join(SAVE_DIR, "respiration_combine_batch_summary.csv")
    xlsx_path = os.path.join(SAVE_DIR, "respiration_combine_batch_summary.xlsx")
    fieldnames = [
        "stem",
        "duration_sec",
        "f_est_bpm",
        "T_period_sec",
        "breath_count",
        "estimated_bpm",
        "error_bpm",
        "signal_quality_score",
        "confidence_score",
        "reliability",
        "spectral_count",
        "count_spectral_error",
        "count_spectral_score",
        "peak_interval_cv",
        "peak_interval_score",
        "median_prominence",
        "prominence_score",
        "eval_gt_expected_bpm",
        "eval_psd_count_error",
        "eval_gt_count_error",
        "eval_gt_count_score",
        "fusion_snr_db",
        "fusion_peak_ratio",
        "pair_consistency",
        "valid_window_ratio",
        "temporal_freq_std_bpm",
        "real_fs_hz",
        "used_fs_hz",
        "completeness",
        "completeness_score",
        "raw_amp_db",
        "raw_amp_score",
        "cycles_observed",
        "cycles_score",
        "min_cycles_observed",
        "good_cycles_observed",
        # "attention_weights",
        # "pair_variances",
        # "pair_quality",
        # "pair_effective_quality",
        # "pair_snr_db",
        # "pair_peak_bpm",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_styled_xlsx(rows, fieldnames, xlsx_path)
    return path, xlsx_path


def main():
    rows = []
    fail_count = 0

    print("========== Respiration Extraction ==========", flush=True)
    print(f"Output directory: {SAVE_DIR}", flush=True)

    for data_dir in DATA_DIRS:
        try:
            rows.append(process_one_dir(data_dir))
        except Exception as exc:
            fail_count += 1
            print(f"\n[Skipped] {data_dir}", flush=True)
            print(f"Reason: {exc}", flush=True)

    summary_path, summary_xlsx_path = write_summary(rows)

    print("\n========== complete ==========", flush=True)
    print(f"Succeeded: {len(rows)}", flush=True)
    print(f"Failed/skipped: {fail_count}", flush=True)
    print(f"summary: {summary_path}", flush=True)
    print(f"summary xlsx: {summary_xlsx_path}", flush=True)
    print(f"Output directory: {SAVE_DIR}", flush=True)


if __name__ == "__main__":
    main()
