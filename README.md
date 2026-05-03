# Neurosonics

> Does music entrain your brain?

Neurosonics is an EEG research pipeline that measures whether — and to what degree — a song's rhythmic structure synchronizes with a listener's brainwaves. It uses OpenBCI hardware to record neural activity and compares resting-state EEG against EEG captured during music playback across five classical brainwave bands.

---

## The Research Question

We know music affects mood. But can we *measure* it in the brain?

**Q1 — Neural Entrainment:** Does listening to a song shift your brainwave activity, and does the pattern of that shift mirror the song's own rhythmic fingerprint?

Entrainment is the phenomenon where brain oscillations synchronize to the rhythmic structure of an external stimulus. If a song's amplitude envelope pulses at theta frequencies (4–8 Hz), does the listener's theta power rise and fall in kind?

---

## How It Works

Audio frequencies (20–20,000 Hz) don't overlap with brainwave frequencies (0.5–100 Hz). The bridge is the **amplitude envelope** — the slow rise and fall of a song's loudness over time. Research in auditory neuroscience shows that these low-frequency modulations in music are what drive neural entrainment in listeners.

**Brainwave bands analyzed:**

| Band  | Frequency     | Associated with |
|-------|---------------|-----------------|
| Delta | 0.5 – 4 Hz    | Slow pulse, fundamental beat |
| Theta | 4 – 8 Hz      | Rhythmic grouping, emotional memory |
| Alpha | 8 – 12 Hz     | Relaxation, engagement |
| Beta  | 12 – 30 Hz    | Attention, cognitive processing |
| Gamma | 30 – 100 Hz   | Complex auditory binding |

---

## Pipeline

```
Step 1 — Song fingerprint
    brainwave_analysis.py  ←  YouTube URL
                           →  song_fingerprint.csv  (band power over time)
                           →  brainwave_bands.png   (visual plot)

Step 2 — EEG recording  (OpenBCI hardware)
    Record 2 min silence  →  baseline_user1.csv
    Record full song      →  listening_user1.csv

Step 3 — Entrainment analysis
    eeg_entrainment.py  ←  song_fingerprint.csv
                        ←  baseline_user1.csv
                        ←  listening_user1.csv
                        →  entrainment_result.png
```

---

## Installation

```bash
pip install yt-dlp librosa scipy matplotlib numpy
```

OpenBCI GUI is required for EEG recording — download at [openbci.com](https://openbci.com).

---

## Usage

### Step 1 — Analyze the song

```bash
python brainwave_analysis.py --song "https://www.youtube.com/watch?v=XXXX"
```

Outputs:
- `song_fingerprint.csv` — sliding-window band power for each of the 5 bands
- `brainwave_bands.png` — visual plot of the song's brainwave-band fingerprint

### Step 2 — Record EEG

Using OpenBCI GUI:
1. Fit the headset and confirm signal quality
2. **Baseline:** Record 2 minutes of silence (eyes closed, no music) → export as `baseline_user1.csv`
3. **Listening:** Start the song and the recording simultaneously. Stop both when the song ends → export as `listening_user1.csv`

### Step 3 — Run entrainment analysis

```bash
python eeg_entrainment.py \
    --song-csv  song_fingerprint.csv \
    --baseline  baseline_user1.csv \
    --listening listening_user1.csv \
    --out       entrainment_user1.png
```

---

## Output

The entrainment analysis produces a figure with three columns across five brainwave bands:

| Column | Description |
|--------|-------------|
| **Song Fingerprint** | The song's band power over time (from Step 1) |
| **EEG During Listening** | The listener's brain power in the same band, with entrainment score r |
| **Δ vs Baseline (%)** | How much each band rose or fell relative to resting state |

A summary bar chart at the bottom shows the **neural entrainment score** (Pearson r) per band — how closely the brain's fluctuations tracked the song's. r > 0.5 indicates meaningful entrainment.

---

## Project Structure

```
Neurosonics/
├── brainwave_analysis.py   # Step 1 — song fingerprint from YouTube
└── eeg_entrainment.py      # Step 3 — EEG comparison and entrainment analysis
```

---

## Recording Protocol

| File | Duration | Conditions |
|------|----------|------------|
| `baseline_userN.csv` | ~2 min | Silence, eyes closed, relaxed |
| `listening_userN.csv` | Song duration | Recording starts with the song, ends with the song |

Name files consistently — `baseline_user1.csv`, `listening_user1.csv`, etc. — to keep multi-participant sessions organized.

---

## Roadmap

- **Q1 (current):** Does the song entrain a single listener's brain?
- **Q2 (planned):** Does preference shape the response? Compare EEG between listeners who love vs. dislike the same song.

---

## Hardware

- [OpenBCI Cyton](https://openbci.com/cyton) (8-channel, 250 Hz) or Ganglion (4-channel, 125 Hz)
- Standard 10-20 electrode placement

---

*Neurosonics — measuring what music does to the brain.*
