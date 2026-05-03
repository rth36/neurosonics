#!/usr/bin/env python3
"""
Neurosonics — EEG Entrainment Analysis

Compares resting-state EEG (baseline) against EEG recorded while listening
to a song, and checks whether the brain's band-power shifts mirror the
song's own amplitude-envelope fingerprint (neural entrainment).

Workflow:
    Step 1 — brainwave_analysis.py  →  song_fingerprint.csv  (run once per song)
    Step 2 — Record EEG             →  baseline_user1.csv, listening_user1.csv
    Step 3 — eeg_entrainment.py     →  entrainment plot

Usage:
    python eeg_entrainment.py \\
        --song-csv  song_fingerprint.csv \\
        --baseline  baseline_user1.csv \\
        --listening listening_user1.csv

    # Alternatively, pass --song <YouTube URL> to skip the CSV and download on the fly.

    # Optional flags:
    --sr        125                  # OpenBCI sample rate (Cyton-Daisy = 125 Hz)
    --window    2.0                  # sliding-window size in seconds
    --channel   all                  # 'all' to average, or 0-indexed channel number
    --out       entrainment_result.png

IMPORTANT — recording protocol:
    baseline_userN.csv  : EEG recorded during silence (2 min recommended).
    listening_userN.csv : EEG recorded starting exactly when the song starts,
                          ending when the song ends. Trim in OpenBCI GUI if needed.

Dependencies:
    pip install yt-dlp librosa scipy matplotlib numpy
"""

import argparse
import csv
import glob
import subprocess
import tempfile
import os
import sys
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

# ── Constants ────────────────────────────────────────────────────────────────

BANDS = {
    "Delta": (0.5, 4,   "#4B0082"),
    "Theta": (4,   8,   "#1E90FF"),
    "Alpha": (8,   12,  "#228B22"),
    "Beta":  (12,  30,  "#FF8C00"),
    "Gamma": (30,  100, "#DC143C"),
}

ENVELOPE_SR = 500   # Hz — envelope sample rate (covers gamma with margin)


# ── OpenBCI CSV loader ────────────────────────────────────────────────────────

def load_openbci_csv(path: str, sr: int = 250):
    """
    Parse an OpenBCI GUI CSV export.
    Returns (eeg_array [samples × channels], sample_rate).
    """
    rows = []
    header_found = False
    n_eeg_cols = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%"):
                # Detect sample rate from header comments
                if "Sample Rate" in line or "sample rate" in line:
                    for token in line.split():
                        try:
                            sr = int(token)
                        except ValueError:
                            pass
                continue

            parts = [p.strip() for p in line.split(",")]

            # First non-comment row = column headers
            if not header_found:
                header_found = True
                # Count EEG channel columns (named "EEG Channel N" or just numbers)
                n_eeg_cols = sum(
                    1 for p in parts
                    if "eeg" in p.lower() or "channel" in p.lower()
                )
                if n_eeg_cols == 0:
                    # Fall back: columns 1..N-2 are EEG (skip sample index and timestamp)
                    n_eeg_cols = len(parts) - 2
                continue

            try:
                vals = [float(p) for p in parts[1: 1 + n_eeg_cols]]
                if len(vals) == n_eeg_cols:
                    rows.append(vals)
            except ValueError:
                continue

    if not rows:
        raise ValueError(f"No EEG data found in {path}")

    return np.array(rows), sr   # shape: [samples, channels]


# ── Signal processing helpers ─────────────────────────────────────────────────

def notch_filter(data: np.ndarray, sr: float, freq: float = 60.0) -> np.ndarray:
    """Remove power-line noise (60 Hz + harmonics)."""
    out = data.copy()
    for f in [freq, freq * 2, freq * 3]:
        if f < sr / 2:
            b, a = signal.iirnotch(f, Q=30, fs=sr)
            out = signal.filtfilt(b, a, out, axis=0)
    return out


def bandpass(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    low_n  = max(low  / nyq, 1e-4)
    high_n = min(high / nyq, 1 - 1e-4)
    b, a = signal.butter(order, [low_n, high_n], btype="band")
    return signal.filtfilt(b, a, data, axis=0)


def band_power_timeseries(
    data: np.ndarray,   # [samples, channels] or [samples]
    sr: float,
    low: float,
    high: float,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Sliding-window band power (mean squared amplitude of bandpass signal).
    Returns (times_sec, power_array).
    """
    if data.ndim == 2:
        chan_data = data.mean(axis=1)   # average across channels
    else:
        chan_data = data

    filtered = bandpass(chan_data.reshape(-1, 1), low, high, sr).ravel()

    win  = int(window_sec * sr)
    step = int(step_sec * sr)
    n_samples = len(filtered)

    times, powers = [], []
    for start in range(0, n_samples - win + 1, step):
        seg = filtered[start: start + win]
        powers.append(np.mean(seg ** 2))
        times.append((start + win / 2) / sr)

    return np.array(times), np.array(powers)


# ── Song fingerprint ──────────────────────────────────────────────────────────

def load_song_csv(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Load a song_fingerprint.csv produced by brainwave_analysis.py.
    Returns {band_name: (times_array, power_array)}.
    """
    times = []
    band_cols: dict[str, list] = {b: [] for b in BANDS}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row["time_sec"]))
            for b in BANDS:
                band_cols[b].append(float(row[b]))
    t = np.array(times)
    return {b: (t, np.array(band_cols[b])) for b in BANDS}


def download_audio(url: str, out_dir: str) -> str:
    out_template = os.path.join(out_dir, "audio.%(ext)s")
    subprocess.run(
        ["yt-dlp", "--no-playlist", "-x", "--audio-format", "wav",
         "--audio-quality", "0", "-o", out_template, url],
        check=True,
    )
    matches = glob.glob(os.path.join(out_dir, "audio.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp produced no output file.")
    return matches[0]


def song_band_timeseries(
    url: str,
    low: float,
    high: float,
    window_sec: float,
    step_sec: float,
    target_duration_sec: float | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Download song, compute amplitude envelope, return band-power time series.
    Returns (times_sec, power_array, song_duration_sec).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        audio_path = download_audio(url, tmpdir)
        audio, sr = librosa.load(audio_path, sr=None, mono=True)

    duration = len(audio) / sr
    envelope = np.abs(signal.hilbert(audio))
    envelope_ds = librosa.resample(envelope, orig_sr=sr, target_sr=ENVELOPE_SR)

    # Trim envelope to match EEG listening duration if provided
    if target_duration_sec is not None:
        n_trim = int(target_duration_sec * ENVELOPE_SR)
        envelope_ds = envelope_ds[:n_trim]

    times, powers = band_power_timeseries(
        envelope_ds, ENVELOPE_SR, low, high, window_sec, step_sec
    )
    return times, powers, duration


# ── Entrainment score (cross-correlation peak) ────────────────────────────────

def entrainment_score(song_power: np.ndarray, eeg_power: np.ndarray) -> tuple[float, int]:
    """
    Pearson cross-correlation in a ±5-window lag range.
    Returns (peak_r, lag_in_windows).
    """
    n = min(len(song_power), len(eeg_power))
    s = (song_power[:n] - song_power[:n].mean()) / (song_power[:n].std() + 1e-12)
    e = (eeg_power[:n]  - eeg_power[:n].mean())  / (eeg_power[:n].std()  + 1e-12)

    max_lag = 5
    best_r, best_lag = -999.0, 0
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            r = np.corrcoef(s[lag:], e[:n - lag])[0, 1] if n - lag > 2 else 0.0
        else:
            r = np.corrcoef(s[:n + lag], e[-lag:])[0, 1] if n + lag > 2 else 0.0
        if r > best_r:
            best_r, best_lag = r, lag

    return float(best_r), best_lag


# ── Main analysis & plotting ──────────────────────────────────────────────────

def run(args):
    window_sec = args.window
    step_sec   = window_sec / 2

    # ── Load EEG files ───────────────────────────────────────────────────────
    print("Loading baseline EEG …")
    base_data, sr = load_openbci_csv(args.baseline, sr=args.sr)

    print("Loading listening EEG …")
    listen_data, _ = load_openbci_csv(args.listening, sr=args.sr)

    # Select channel(s)
    if args.channel != "all":
        ch = int(args.channel)
        base_data   = base_data[:, ch:ch+1]
        listen_data = listen_data[:, ch:ch+1]

    # ── Preprocess ───────────────────────────────────────────────────────────
    print("Preprocessing (notch filter) …")
    base_data   = notch_filter(base_data,   sr)
    listen_data = notch_filter(listen_data, sr)

    listen_duration = len(listen_data) / sr
    print(f"  Baseline:  {len(base_data)/sr:.1f} s")
    print(f"  Listening: {listen_duration:.1f} s")

    # ── Baseline mean power per band (scalar reference) ──────────────────────
    print("Computing baseline band power …")
    baseline_means = {}
    for band, (low, high, _) in BANDS.items():
        _, bp = band_power_timeseries(base_data, sr, low, high, window_sec, step_sec)
        baseline_means[band] = bp.mean()

    # ── Listening band power time series ─────────────────────────────────────
    print("Computing listening-EEG band power …")
    listen_ts = {}
    for band, (low, high, _) in BANDS.items():
        times, bp = band_power_timeseries(listen_data, sr, low, high, window_sec, step_sec)
        listen_ts[band] = (times, bp)

    # ── Song fingerprint ─────────────────────────────────────────────────────
    if args.song_csv:
        print(f"Loading song fingerprint from CSV: {args.song_csv} …")
        song_ts = load_song_csv(args.song_csv)
        song_label = args.song_csv
    else:
        print("Downloading song and computing fingerprint …")
        song_ts = {}
        for band, (low, high, _) in BANDS.items():
            t, bp, song_dur = song_band_timeseries(
                args.song, low, high, window_sec, step_sec, listen_duration
            )
            song_ts[band] = (t, bp)
        print(f"  Song duration: {song_dur:.1f} s")
        song_label = args.song

    # ── Build figure ─────────────────────────────────────────────────────────
    n_bands = len(BANDS)
    fig = plt.figure(figsize=(18, 3.4 * (n_bands + 1)))
    gs  = gridspec.GridSpec(n_bands + 1, 3, hspace=0.65, wspace=0.35, figure=fig)

    col_labels = ["Song Fingerprint", "EEG During Listening", "Δ Power vs Baseline  (%)"]

    for col, label in enumerate(col_labels):
        fig.text(
            (col + 0.5) / 3, 1.002, label,
            ha="center", va="bottom", fontsize=12, fontweight="bold",
            transform=fig.transFigure,
        )

    entrainment_scores = {}

    for row, (band, (low, high, color)) in enumerate(BANDS.items()):
        s_times, s_power = song_ts[band]
        l_times, l_power = listen_ts[band]
        baseline_mean    = baseline_means[band]

        # Normalise song power for visual comparison (z-score)
        s_norm = (s_power - s_power.mean()) / (s_power.std() + 1e-12)
        l_norm = (l_power - l_power.mean()) / (l_power.std() + 1e-12)

        # Percent change from baseline
        pct_change = 100.0 * (l_power - baseline_mean) / (baseline_mean + 1e-12)

        # Entrainment
        peak_r, lag = entrainment_score(s_power, l_power)
        entrainment_scores[band] = peak_r

        band_label = f"{band}  ({low}–{high} Hz)"

        # Col 0 — song fingerprint
        ax0 = fig.add_subplot(gs[row, 0])
        ax0.plot(s_times, s_norm, color=color, lw=0.9)
        ax0.set_title(band_label, color=color, fontsize=9, fontweight="bold")
        ax0.set_ylabel("Norm. power")
        ax0.margins(x=0)
        ax0.axhline(0, color="#aaa", lw=0.5, ls="--")

        # Col 1 — EEG listening (z-scored for same scale as song)
        ax1 = fig.add_subplot(gs[row, 1])
        ax1.plot(l_times, l_norm, color=color, lw=0.9, alpha=0.85)
        ax1.set_title(f"r = {peak_r:.2f}  (lag {lag*step_sec:+.0f}s)",
                      fontsize=8, color="gray")
        ax1.margins(x=0)
        ax1.axhline(0, color="#aaa", lw=0.5, ls="--")

        # Col 2 — % change from baseline
        ax2 = fig.add_subplot(gs[row, 2])
        ax2.fill_between(l_times, pct_change, 0,
                         where=pct_change > 0, color=color, alpha=0.45, label="↑ above baseline")
        ax2.fill_between(l_times, pct_change, 0,
                         where=pct_change < 0, color="#888", alpha=0.35, label="↓ below baseline")
        ax2.axhline(0, color="black", lw=0.8)
        ax2.set_ylabel("% change")
        ax2.margins(x=0)

        if row == n_bands - 1:
            for ax in (ax0, ax1, ax2):
                ax.set_xlabel("Time (s)")

    # ── Summary bar chart — entrainment scores ────────────────────────────────
    ax_sum = fig.add_subplot(gs[n_bands, :])
    bands_list  = list(entrainment_scores.keys())
    scores      = [entrainment_scores[b] for b in bands_list]
    bar_colors  = [BANDS[b][2] for b in bands_list]
    bars = ax_sum.bar(bands_list, scores, color=bar_colors, alpha=0.75, edgecolor="black", lw=0.6)
    ax_sum.axhline(0, color="black", lw=0.8)
    ax_sum.axhline(0.3,  color="green", lw=0.8, ls="--", alpha=0.6, label="weak entrainment (r=0.3)")
    ax_sum.axhline(0.5,  color="blue",  lw=0.8, ls="--", alpha=0.6, label="moderate (r=0.5)")
    for bar, score in zip(bars, scores):
        ax_sum.text(bar.get_x() + bar.get_width() / 2, score + 0.01,
                    f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    ax_sum.set_title("Neural Entrainment Score per Band  (peak cross-correlation with song fingerprint)",
                     fontweight="bold")
    ax_sum.set_ylabel("Pearson r")
    ax_sum.set_ylim(-0.2, 1.0)
    ax_sum.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"Q1: Neural Entrainment Analysis\nSong: {song_label}",
        fontsize=11, y=1.03,
    )

    plt.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nSaved → {args.out}")

    # ── Print text summary ────────────────────────────────────────────────────
    print("\n── Entrainment scores ──────────────────────────────────")
    for band, r in entrainment_scores.items():
        low, high, _ = BANDS[band]
        label = "strong" if r > 0.5 else "moderate" if r > 0.3 else "weak"
        pct = 100 * (listen_ts[band][1].mean() - baseline_means[band]) / (baseline_means[band] + 1e-12)
        print(f"  {band:6s} ({low:4.1f}–{high:4.0f} Hz):  r={r:+.2f} [{label:8s}]   "
              f"mean power vs baseline: {pct:+.1f}%")


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EEG Q1 — Neural entrainment analysis")
    song_src = p.add_mutually_exclusive_group(required=True)
    song_src.add_argument("--song-csv", dest="song_csv", help="song_fingerprint.csv from brainwave_analysis.py")
    song_src.add_argument("--song",                      help="YouTube URL (downloads on the fly)")
    p.add_argument("--baseline",  required=True, help="OpenBCI CSV — resting/silence recording")
    p.add_argument("--listening", required=True, help="OpenBCI CSV — recorded while listening")
    p.add_argument("--sr",        type=int,   default=125,       help="EEG sample rate (Hz) — Cyton-Daisy default is 125")
    p.add_argument("--window",    type=float, default=2.0,       help="Sliding window size (s)")
    p.add_argument("--channel",   default="all",                 help="Channel index or 'all'")
    p.add_argument("--out",       default="q1_result.png",       help="Output filename")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
