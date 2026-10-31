#!/usr/bin/env python3
"""Batch-track promos (frozen v3 pipeline, batch_lib3 template:
lit photos, segmented hold/travel camera, solidification swell).
T2 sentimental opening kept exactly as approved (warm grade +
opening overlay in the driver, sentimental schedule flag). 720p30,
encode crf23 / maxrate 5M / bufsize 2.5M / faststart, audio
stream-copied. Usage: render_track.py 3,1 [outdir] | still"""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
WORK = HERE / 'work'
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import batch_lib as BL
import batch_lib3 as L
import batch_b as B
from overlays import warm_grade, opening_overlay

BM = HERE.parent / 'experiments' / 'batch_music'
OUTD = HERE / 'output'
OUTD.mkdir(parents=True, exist_ok=True)
FPS = 30
SRC = {1: 'track1_24.m4a', 2: 'track2_21.m4a', 3: 'track3_22.m4a', 4: 'track4_23.m4a'}


def accents(n):
    d = json.loads((BM / f'track{n}_beatmap.json').read_text())
    return [(a['time'], a['strength']) for a in d['accents']], d['duration_s']


def build_tracks():
    T = {}
    acc, dur = accents(1)
    pins = BL.pick_pins(acc, 3.9, 23.0, gap=1.5)
    T[1] = dict(dur=dur, A=24.0, drop_accent=29.87, sentimental=False,
                sched=L.build_schedule3(
                    24.0, 10, 3.4, 21.9, pins, mont0=3.4, card_start=0.5,
                    card_end=2.8, route_win=(0.13 * 24.0, 0.31 * 24.0)))
    acc, dur = accents(2)
    pins = BL.pick_pins(acc, 4.4, 13.6, gap=1.5)
    T[2] = dict(dur=dur, A=14.5, drop_accent=18.07, sentimental=True,
                sched=L.build_schedule3(
                    14.5, 6, 4.6, 12.4, pins, mont0=4.4, card_start=3.4,
                    card_end=5.0, route_win=(4.0, 6.6), sentimental=True))
    acc, dur = accents(3)
    pins = BL.pick_pins(acc, 3.0, 13.5, gap=1.2)
    T[3] = dict(dur=dur, A=14.4, drop_accent=19.39, sentimental=False,
                sched=L.build_schedule3(
                    14.4, 7, 2.6, 12.2, pins, mont0=3.0, card_start=0.5,
                    card_end=2.2, route_win=(0.13 * 14.4, 0.31 * 14.4)))
    acc, dur = accents(4)
    pins = BL.pick_pins(acc, 2.8, 13.9, gap=1.5)
    T[4] = dict(dur=dur, A=14.6, drop_accent=18.94, sentimental=False,
                sched=L.build_schedule3(
                    14.6, 7, 2.4, 13.0, pins, mont0=2.8, card_start=0.5,
                    card_end=2.0, route_win=(0.13 * 14.6, 0.31 * 14.6)))
    return T


TRACKS = build_tracks()
for n, tr in TRACKS.items():
    L._cam_keys(tr['sched'])
    print(f"[t{n}] heroes:",
          [(round(m['accent'], 2), round(m['slot'], 2), m['is_pin'])
           for m in tr['sched']['frame_meta']], flush=True)
    print(f"[t{n}] groups:",
          [(round(g['hold_start'], 2), round(g['hold_end'], 2))
           for g in tr['sched']['groups']], flush=True)


def render_track(n):
    tr = TRACKS[n]
    sched = tr['sched']
    total = round(tr['dur'] * FPS)
    a_frames = round(tr['A'] * FPS)
    b_frames = total - a_frames
    A_sec = a_frames / FPS
    b_dur = b_frames / FPS
    drop = tr['drop_accent'] - A_sec
    Image.fromarray(L.render_a_frame3(sched, 0, still=True)).save(
        L.WORK / 'scene_still.png')  # B segment reads it from assets/
    print(f'[t{n}] total {total}f A {a_frames}f B {b_frames}f drop_local {drop:.2f}',
          flush=True)
    silent = WORK / f'sample_t{n}_silent.mp4'
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{L.W}x{L.H}', '-r', str(FPS), '-i', '-',
           '-vf', 'scale=1280:720:flags=lanczos',
           '-c:v', 'libx264', '-profile:v', 'high', '-crf', '23',
           '-maxrate', '5M', '-bufsize', '2.5M', '-preset', 'medium',
           '-pix_fmt', 'yuv420p', '-r', str(FPS), '-an', str(silent)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a_frames):
        t = i / FPS
        img = L.render_a_frame3(sched, t)
        if tr['sentimental']:
            img = warm_grade(img)
            if t < 4.6:
                img = opening_overlay(img, t)
        p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        if i % 60 == 0:
            print(f'[t{n}] A {i}/{a_frames} {time.time()-t0:.0f}s', flush=True)
    for i in range(b_frames):
        img = B.render_b_frame(i / FPS, b_dur, drop)
        p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        if i % 120 == 0:
            print(f'[t{n}] B {i}/{b_frames} {time.time()-t0:.0f}s', flush=True)
    p.stdin.close(); p.wait()
    assert p.returncode == 0, (n, p.returncode)
    out = OUT_DIR / f'echo_promo_t{n}_sample.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(silent),
                    '-i', str(BM / SRC[n]),
                    '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy',
                    '-movflags', '+faststart', str(out)], check=True)
    print(f'[t{n}] shipped -> {out} ({time.time()-t0:.0f}s)', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'still':
        img = L.render_a_frame3(TRACKS[3]['sched'], 0, still=True)
        Image.fromarray(img).save(L.WORK / 'scene_still.png')  # assets/
        print('scene_still regenerated', flush=True)
    else:
        OUT_DIR = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTD
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for n in [int(x) for x in sys.argv[1].split(',')]:
            render_track(n)
        print('ALL DONE', flush=True)
