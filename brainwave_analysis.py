#!/usr/bin/env python3
"""
Brainwave-band analysis of a YouTube song.

Downloads audio, computes its amplitude envelope via Hilbert transform,
then bandpass-filters the envelope into delta/theta/alpha/beta/gamma bands.
This works because the *slow modulations* of the audio signal are what
entrain neural oscillations — not the carrier audio frequencies themselves.

Dependencies:
    pip install yt-dlp librosa scipy matplotlib numpy
"""

import sys
import glob
import subprocess
import tempfile
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import signal

# Brainwave bands: name → (low_hz, high_hz, plot_color)
BANDS = {
    "Delta": (0.5, 4,   "#4B0082"),
    "Theta": (4,   8,   "#1E90FF"),
    "Alpha": (8,   12,  "#228B22"),
    "Beta":  (12,  30,  "#FF8C00"),
    "Gamma": (30,  100, "#DC143C"),
}

# Envelope sample rate — 500 Hz covers gamma (100 Hz) with 2.5× margin
ENVELOPE_SR = 500


def download_audio(url: str, out_dir: str) -> str:
    out_template = os.path.join(out_dir, "audio.%(ext)s")
    subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-x",
            "--audio-format", "wav",
            "--audio-quality", "0",
            "-o", out_template,
            url,
        ],
        check=True,
    )
    matches = glob.glob(os.path.join(out_dir, "audio.*"))
    if not matches:
        raise FileNotFoundError("yt-dlp did not produce an output file.")
    return matches[0]


def bandpass(data: np.ndarray, low: float, high: float, fs: float, order: int = 4) -> np.ndarray:
    nyq = fs / 2.0
    b, a = signal.butter(order, [low / nyq, high / nyq], btype="band")
    return signal.filtfilt(b, a, data)


def analyze(url: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Download ────────────────────────────────────────────────────────
        print("Downloading audio …")
        audio_path = download_audio(url, tmpdir)

        # ── Load ────────────────────────────────────────────────────────────
        print("Loading audio …")
        audio, sr = librosa.load(audio_path, sr=None, mono=True)
        duration = len(audio) / sr
        print(f"  Duration: {duration:.1f} s  |  Sample rate: {sr} Hz")

        # ── Amplitude envelope via Hilbert transform ─────────────────────────
        print("Computing amplitude envelope …")
        envelope_full = np.abs(signal.hilbert(audio))
        envelope = librosa.resample(envelope_full, orig_sr=sr, target_sr=ENVELOPE_SR)
        t = np.linspace(0, duration, len(envelope))

        # ── Build figure ─────────────────────────────────────────────────────
        n_rows = len(BANDS) + 2  # waveform + PSD + one row per band
        fig = plt.figure(figsize=(16, 3.2 * n_rows))
        gs = gridspec.GridSpec(n_rows, 1, hspace=0.60, figure=fig)

        # Row 0 — raw waveform
        ax0 = fig.add_subplot(gs[0])
        step = max(1, len(audio) // 20_000)
        t_audio = np.linspace(0, duration, len(audio))
        ax0.plot(t_audio[::step], audio[::step], color="#333", lw=0.4, alpha=0.85)
        ax0.set_title("Audio Waveform", fontweight="bold")
        ax0.set_ylabel("Amplitude")
        ax0.margins(x=0)

        # Row 1 — PSD of envelope (shows where rhythm energy sits)
        ax1 = fig.add_subplot(gs[1])
        freqs, psd = signal.welch(envelope, fs=ENVELOPE_SR, nperseg=2048)
        mask120 = freqs <= 120
        ax1.semilogy(freqs[mask120], psd[mask120], color="#222", lw=1.0)
        for band, (low, high, color) in BANDS.items():
            band_mask = mask120 & (freqs >= low) & (freqs <= high)
            ax1.fill_between(freqs[band_mask], psd[band_mask],
                             alpha=0.40, color=color, label=f"{band} ({low}–{high} Hz)")
        ax1.set_title("Envelope Power Spectral Density (0–120 Hz)", fontweight="bold")
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Power (log scale)")
        ax1.legend(loc="upper right", fontsize=8, framealpha=0.7)
        ax1.margins(x=0)

        # Rows 2+ — filtered envelope per band
        for row, (band, (low, high, color)) in enumerate(BANDS.items(), start=2):
            ax = fig.add_subplot(gs[row])
            filtered = bandpass(envelope, low, high, ENVELOPE_SR)
            ax.plot(t, filtered, color=color, lw=0.8)
            ax.set_title(f"{band} Band  ({low}–{high} Hz)", fontweight="bold", color=color)
            ax.set_ylabel("Amplitude")
            ax.margins(x=0)
            if row == n_rows - 1:
                ax.set_xlabel("Time (seconds)")

        fig.suptitle(
            f"Brainwave-Band Analysis of Audio Envelope\n{url}",
            fontsize=11,
            y=1.005,
        )

        out_path = "brainwave_bands.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python brainwave_analysis.py <youtube_url>")
        sys.exit(1)
    analyze(sys.argv[1])
