#!/usr/bin/env python3
"""Beat analysis for promo tracks: spectral-flux onset envelope +
comb-grid tempo scan (scipy/numpy), the method documented in
experiments/batch_music/SUMMARY.md. The music itself is never edited.

NOTE: the beat maps shipped in experiments/batch_music/ are the
canonical frozen inputs for the current tracks — they were produced
with this method when the tracks were first analysed. This script
regenerates an equivalent map for a NEW track (or re-derives one);
expect small numeric differences from the shipped maps, never use it
to overwrite them silently.

Usage: analyze_beats.py <audio.m4a> <out_beatmap.json>
"""
import json, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from scipy import signal as sig

SR = 22050
WIN, HOP = 2048, 512


def load_mono(path):
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        wav = f.name
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(path),
                    '-ac', '1', '-ar', str(SR), wav], check=True)
    import wave
    w = wave.open(wav)
    n = w.getnframes()
    x = np.frombuffer(w.readframes(n), dtype='<i2').astype(np.float64) / 32768.0
    w.close()
    Path(wav).unlink(missing_ok=True)
    return x


def onset_envelope(x):
    win = np.hanning(WIN)
    n_fr = 1 + max(0, (len(x) - WIN) // HOP)
    mags = np.empty((WIN // 2 + 1, n_fr))
    for i in range(n_fr):
        seg = x[i * HOP: i * HOP + WIN] * win
        mags[:, i] = np.abs(np.fft.rfft(seg))
    flux = np.zeros(n_fr)
    d = np.diff(mags, axis=1)
    flux[1:] = np.maximum(d, 0).sum(axis=0)
    flux /= (flux.max() + 1e-12)
    t = np.arange(n_fr) * HOP / SR
    return t, flux


def pick_peaks(t, env, min_dist_s=0.12, thresh=0.12):
    pk, props = sig.find_peaks(env, height=thresh,
                               distance=max(1, int(min_dist_s * SR / HOP)))
    return [(float(t[i]), float(env[i])) for i in pk]


def _comb_score(t, env, period):
    n_beats = int(t[-1] / period)
    if n_beats < 4:
        return 0.0, 0.0
    best, best_ph = 0.0, 0.0
    for ph in np.arange(0.0, period, period / 32):
        idx = np.clip(((ph + period * np.arange(n_beats)) / (HOP / SR)).astype(int),
                      0, len(env) - 1)
        s = env[idx].mean()
        if s > best:
            best, best_ph = s, ph
    return best, best_ph


def tempo_scan(t, env):
    """Comb-grid scan: for each candidate period, fold the envelope
    onto a beat grid and score the best phase alignment. Octave rule:
    if the half-tempo grid scores >=85% of the best, prefer it (the
    shipped analyses favour the slower pulse)."""
    best = (0.0, -1.0, 0.0)
    for bpm in np.arange(40, 200.5, 0.5):
        s, ph = _comb_score(t, env, 60.0 / bpm)
        if s > best[1]:
            best = (float(bpm), float(s), float(ph))
    bpm, score, phase = best
    while bpm / 2 >= 40:
        s2, ph2 = _comb_score(t, env, 120.0 / bpm)
        if s2 >= 0.85 * score:
            bpm, score, phase = bpm / 2, s2, ph2
        else:
            break
    return bpm, phase


def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    x = load_mono(src)
    dur = len(x) / SR
    t, env = onset_envelope(x)
    beats = pick_peaks(t, env, min_dist_s=0.12, thresh=0.10)
    bpm, phase = tempo_scan(t, env)
    # accents: strong peaks that sit on the fitted tempo grid
    period = 60.0 / bpm
    acc = []
    for time_, s in beats:
        if s < 0.75:
            continue
        k = round((time_ - phase) / period)
        if abs(time_ - (phase + k * period)) <= 0.18:
            acc.append({'time': round(time_, 2), 'strength': round(s, 2)})
    loud_i = int(np.argmax(sig.resample_poly(np.abs(x), 1, 512)))
    head = float(t[np.argmax(env > 0.02)]) if (env > 0.02).any() else 0.0
    d = {
        'audio_file': str(src),
        'duration_s': round(dur, 2),
        'music_start_s': round(head, 2),
        'music_end_s': round(dur, 2),
        'tempo_bpm': round(bpm, 1),
        'beat_period_s': round(60.0 / bpm, 3) if bpm else None,
        'beats': [{'time': round(a, 2), 'strength': round(b, 2)} for a, b in beats],
        'accents': acc,
        'structure': {'loudest_s': round(loud_i * 512 / SR, 2)},
        'method': 'spectral-flux onset envelope + comb-grid tempo scan '
                  '(analyze_beats.py, frozen promo pipeline)',
        'notes': 'Regenerated map; shipped maps in experiments/batch_music/ '
                 'remain canonical for the current tracks.',
    }
    out.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f'{src.name}: dur {dur:.2f}s bpm {bpm:.1f} beats {len(beats)} '
          f'accents {len(acc)} -> {out}')


if __name__ == '__main__':
    main()
