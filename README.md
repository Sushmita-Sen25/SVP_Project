# SVP Lab: Speech & Video Processing

**Live site:** https://svp-project-nine.vercel.app

A course project with three working systems, all running **entirely in the browser** (recordings and videos never leave the device):

| Module | Title | What it does |
|---|---|---|
| 1a | **Sanskrit Pronunciation Using MFCC and Spectral Features** | Compares a student's pronunciation of a word (e.g. कृष्ण, *kṛṣṇa* vs. *krishna / kishna / krisna*) with a reference. Outputs a similarity score and breaks it down into vowel, consonant, duration, spectral and per-MFCC differences, with error localisation and error-type identification. |
| 1b | **HMM-Based Recognition of Bhagavad Gita Shlokas Using MFCC Features** | Recognises which of 10 verses was recited (e.g. *Verse 9.26*) using one left-to-right Gaussian HMM per verse, with Viterbi alignment and in-browser Baum–Welch retraining. |
| 2 | **Long-Video Analysis and Summarisation** | For a video of any length (e.g. a 30-minute lecture): (a) important frames, (b) scene boundaries, (c) motion map, (d) tracked objects and human movement, (e) a short summary video. |

## Results (bundled corpus)

| Experiment | Result |
|---|---|
| Verse recognition, 60 held-out recitations | **100%** |
| Verse recognition, 2 speakers never seen in training | **100%** |
| Verse recognition, leave-one-speaker-out (6 folds) | **100%** |
| Verse recognition with white noise at 20 / 10 / 5 dB SNR | 90% / 50% / 32% (clean-trained models) |
| Verse recognition from the first 30% / 50% of a verse | 100% / 97% (any end state) |
| Pronunciation error-type identification, leave-one-speaker-out | **64.6%** (4-way, chance 25%) |
| Scene boundaries on the demo video (7 true boundaries) | 7/7 found, F1 ≈ 93% |

A single reference-vs-student similarity score mostly measures voice differences when the two speakers differ. On this corpus a correct word from another speaker scores about the same as a wrong one (median 80% vs 79%). Comparing against several reference speakers (error identification) is what separates the forms. The site states this limitation next to the score.

## How it works

**Speech front end** (`assets/js/dsp.js`): pre-emphasis → 25 ms Hamming frames / 10 ms hop → 512-pt FFT → 26 mel filters → log → DCT → 13 liftered MFCCs → Δ, ΔΔ → CMVN. Plus energy, zero-crossing rate, spectral centroid, roll-off, flatness and YIN pitch.

**Pronunciation** (`assets/js/pronounce.js`): MFCC+Δ+ΔΔ projected with a speaker-robust **LDA** (trained on 86k frames labelled with 359 HMM states) → **DTW** (symmetric-2, Sakoe–Chiba band) → calibrated logistic scores. Frames are labelled vowel-like or consonant-like for the segmental scores. A long-term average spectrum and fricative centroid separate s / ś / ṣ. Multi-template DTW against all forms from several speakers identifies the error type.

**Verse recognition** (`assets/js/hmm.js`): Bakis HMM (stay / next / skip), diagonal Gaussians, N ≈ frames/20 states. Flat start → 4 Viterbi (segmental k-means) iterations → 6 Baum–Welch iterations, all in the log domain. Recognition by the forward algorithm, with an optional any-end-state mode for partial recitations.

**Video** (`assets/js/video-core.js`): seek-based frame sampling (auto 4 → 0.5 fps by length, at most 3,600 samples). 16×4×4 HSV histograms → Bhattacharyya + pixel dissimilarity → adaptive-threshold cuts and twin-comparison gradual transitions. Medoid key frames, motion energy image, global-shift camera-motion detection, per-scene median background subtraction + connected components, or COCO-SSD (TensorFlow.js) person detection. Greedy constant-velocity tracker. Importance-ranked summary clips that play in place or export to WebM.

## Project structure

```
index.html, pronunciation.html, verse.html, video.html   pages
assets/css/style.css                                     design system (light/dark)
assets/js/dsp.js, dtw.js, hmm.js, linalg.js              signal processing & models (also used by Node)
assets/js/pronounce.js, video-core.js                    analysis logic
assets/js/*-page.js, plot.js, audio.js, ui.js            browser UI
data/                                                    word/verse metadata, trained HMMs, LDA, evaluations
audio/words, audio/gita/{train,test}                     TTS speech corpus (372 clips)
media/demo.mp4, media/demo_truth.json                    synthetic demo video + ground truth
scripts/                                                 corpus generation, training, calibration
```

## Run locally

Any static server with HTTP range support (needed for video seeking):

```bash
npm run dev    # http://localhost:5174
```

## Reproduce data and models

```bash
pip install edge-tts opencv-python numpy imageio-ffmpeg
python scripts/generate_audio.py
python scripts/make_demo_video.py
export FFMPEG=$(python -c "import imageio_ffmpeg as f; print(f.get_ffmpeg_exe())")
node scripts/train_gita_hmm.mjs
node scripts/train_lda.mjs
node scripts/calibrate_pronunciation.mjs
```

## Notes and limitations

- The speech corpus is synthesised with neural TTS voices (Hindi, Marathi, Nepali), not recorded from native Vedic reciters. For real use, record a teacher as the reference. The page supports recording and uploading both sides.
- The verse HMMs are trained on clean speech and degrade under noise. Multi-condition training or spectral subtraction are natural extensions.
- The demo video is procedurally generated. COCO-SSD person tracking needs real footage; the demo uses moving-blob tracking.
