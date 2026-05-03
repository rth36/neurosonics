#!/usr/bin/env python3
"""
Brainwave-band analysis of a YouTube song.

Downloads audio, computes its amplitude envelope via Hilbert transform,
then bandpass-filters the envelope into delta/theta/alpha/beta/gamma bands.
Also exports a CSV of sliding-window band power — used as input to eeg_q1_analysis.py.

Usage:
    python brainwave_analysis.py --song "https://www.youtube.com/watch?v=XXXX"

    # Optional:
    --out-png  brainwave_bands.png   # plot output
    --out-csv  song_fingerprint.csv  # band power CSV for eeg_q1_analysis.py
    --window   2.0                   # sliding window size in seconds

Dependencies:
    pip install yt-dlp librosa scipy matplotlib numpy
"""

import argparse
import csv
import glob
import subprocess
import tempfile
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

BANDS = {
    "Delta": (0.5, 4,   "#4B0082"),
    "Theta": (4,   8,   "#1E90FF"),
    "Alpha": (8,   12,  "#228B22"),
    "Beta":  (12,  30,  "#FF8C00"),
    "Gamma": (30,  100, "#DC143C"),
}

ENVELOPE_SR = 500   # Hz — covers gamma (100 Hz) with 2.5× margin


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


def bandpass(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, data)


def sliding_band_power(
    envelope: np.ndarray,
    low: float,
    high: float,
    fs: float,
    window_sec: float,
    step_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass-filter envelope then compute sliding-window RMS power."""
    filtered = bandpass(envelope, low, high, fs)
    win  = int(window_sec * fs)
    step = int(step_sec * fs)
    times, powers = [], []
    for start in range(0, len(filtered) - win + 1, step):
        seg = filtered[start: start + win]
        powers.append(float(np.mean(seg ** 2)))
        times.append((start + win / 2) / fs)
    return np.array(times), np.array(powers)


def analyze(args) -> None:
    window_sec = args.window
    step_sec   = window_sec / 2

    with tempfile.TemporaryDirectory() as tmpdir:
        print("Downloading audio …")
        audio_path = download_audio(args.song, tmpdir)

        print("Loading audio …")
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(audio) / sr
        print(f"  Duration: {duration:.1f} s  |  Sample rate: {sr} Hz")

        print("Computing amplitude envelope …")
        envelope = np.abs(signal.hilbert(audio))
        envelope = librosa.resample(envelope, orig_sr=sr, target_sr=ENVELOPE_SR)
        t_full   = np.linspace(0, duration, len(envelope))

        # ── Sliding-window band power per band ───────────────────────────────
        print("Computing band power time series …")
        band_times  = {}
        band_powers = {}
        for band, (low, high, _) in BANDS.items():
            t, p = sliding_band_power(envelope, low, high, ENVELOPE_SR, window_sec, step_sec)
            band_times[band]  = t
            band_powers[band] = p

        # ── Save CSV ─────────────────────────────────────────────────────────
        ref_times = band_times["Delta"]
        with open(args.out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["time_sec"] + list(BANDS.keys()))
            for i, t in enumerate(ref_times):
                row = [f"{t:.4f}"] + [
                    f"{band_powers[b][i]:.8f}" if i < len(band_powers[b]) else "0"
                    for b in BANDS
                ]
                writer.writerow(row)
        print(f"  Band power CSV saved → {args.out_csv}")

        # ── Build figure ─────────────────────────────────────────────────────
        n_rows = len(BANDS) + 2
        fig = plt.figure(figsize=(16, 3.2 * n_rows))
        gs  = gridspec.GridSpec(n_rows, 1, hspace=0.60, figure=fig)

        # Row 0 — raw waveform
        ax0   = fig.add_subplot(gs[0])
        step  = max(1, len(audio) // 20_000)
        t_aud = np.linspace(0, duration, len(audio))
        ax0.plot(t_aud[::step], audio[::step], color="#333", lw=0.4, alpha=0.85)
        ax0.set_title("Audio Waveform", fontweight="bold")
        ax0.set_ylabel("Amplitude")
        ax0.margins(x=0)

        # Row 1 — PSD of envelope
        ax1 = fig.add_subplot(gs[1])
        freqs, psd = signal.welch(envelope, fs=ENVELOPE_SR, nperseg=2048)
        mask = freqs <= 120
        ax1.semilogy(freqs[mask], psd[mask], color="#222", lw=1.0)
        for band, (low, high, color) in BANDS.items():
            bm = mask & (freqs >= low) & (freqs <= high)
            ax1.fill_between(freqs[bm], psd[bm], alpha=0.40, color=color,
                             label=f"{band} ({low}–{high} Hz)")
        ax1.set_title("Envelope Power Spectral Density (0–120 Hz)", fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Power (log scale)")
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax1.margins(x=0)

        # Rows 2+ — filtered envelope per band (continuous, for visual detail)
        for row, (band, (low, high, color)) in enumerate(BANDS.items(), start=2):
            ax = fig.add_subplot(gs[row])
            filtered = bandpass(envelope, low, high, ENVELOPE_SR)
            ax.plot(t_full, filtered, color=color, lw=0.8)
            ax.set_title(f"{band} Band  ({low}–{high} Hz)", fontweight="bold", color=color)
            ax.set_ylabel("Amplitude")
            ax.margins(x=0)
            if row == n_rows - 1:
                ax.set_xlabel("Time (seconds)")

        fig.suptitle(
            f"Brainwave-Band Analysis of Audio Envelope\n{args.song}",
            fontsize=11, y=1.005,
        )
        plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Plot saved → {args.out_png}")


def parse_args():
    p = argparse.ArgumentParser(description="Song brainwave-band fingerprint")
    p.add_argument("--song",    required=True, help="YouTube URL")
    p.add_argument("--out-png", default="brainwave_bands.png",   help="Plot output path")
    p.add_argument("--out-csv", default="song_fingerprint.csv",  help="Band power CSV output")
    p.add_argument("--window",  type=float, default=2.0,         help="Sliding window (s)")
    return p.parse_args()


if __name__ == "__main__":
    analyze(parse_args())
