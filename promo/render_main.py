#!/usr/bin/env python3
"""MAIN promo (frozen v3 pipeline): operator pipeline (batch_lib3 +
batch_b). Lit true-colour photos (never black cards), segmented
hold/travel camera synced to landings, solidification swell.
720p30, encode crf23 / maxrate 5M / bufsize 2.5M / faststart,
music stream-copied unedited. A 0->17.22s, B ->26.56s, crush pins
4.32/7.62/10.02/13.62/14.81, B drop accent 22.01.
Usage: render_main.py stills | full [outdir]"""
import subprocess, sys, time
from pathlib import Path
import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
WORK = HERE / 'work'
WORK.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(HERE))
import batch_lib3 as L
import batch_b as B

EXP = HERE.parent / 'experiments'
OUTD = HERE / 'output'
OUTD.mkdir(parents=True, exist_ok=True)
FPS = 30
A_DUR, TOTAL = 17.22, 26.56
PINS = [4.32, 7.62, 10.02, 13.62, 14.81]
DROP_ACCENT = 22.01

sched = L.build_schedule3(
    A_DUR, 8, 2.9, 14.9, PINS, mont0=2.9, card_start=0.5,
    card_end=2.2, route_win=(0.13 * A_DUR, 0.31 * A_DUR), sentimental=False)
L._cam_keys(sched)
print('heroes (accent, photo_t, t1, pin):',
      [(round(m['accent'], 2), round(m['slot'], 2), round(m['points_t1'], 2), m['is_pin'])
       for m in sched['frame_meta']], flush=True)
print('groups (hold_start, hold_end):',
      [(round(g['hold_start'], 2), round(g['hold_end'], 2)) for g in sched['groups']],
      flush=True)

total = round(TOTAL * FPS)
a_frames = round(A_DUR * FPS)
b_frames = total - a_frames
A_sec = a_frames / FPS
b_dur = b_frames / FPS
drop = DROP_ACCENT - A_sec


def stills():
    out = WORK / 'stills_main'
    out.mkdir(exist_ok=True)
    # regenerate the B-segment scene still with the v3 solid look
    img = L.render_a_frame3(sched, 0, still=True)
    Image.fromarray(img).save(L.WORK / 'scene_still.png')  # assets/, where batch_b reads it
    print('scene_still regenerated', flush=True)
    first_photo = sched['frame_meta'][0]['slot']
    for name, t in [('cards_open', 1.5), ('entrance', first_photo + 0.10),
                    ('hold_crush', 4.32 + 0.35), ('swell_mid', 10.6),
                    ('finale', 16.9)]:
        img = L.render_a_frame3(sched, t)
        Image.fromarray(img).resize((1280, 720), Image.LANCZOS).save(out / f'{name}.png')
        print('still', name, t, flush=True)
    img = B.render_b_frame(6.2, b_dur, drop)
    Image.fromarray(img).resize((1280, 720), Image.LANCZOS).save(out / 'b_closing.png')
    print('still b_closing', flush=True)


def full():
    silent = WORK / 'main_sample_v3_silent.mp4'
    cmd = ['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', f'{L.W}x{L.H}', '-r', str(FPS), '-i', '-',
           '-vf', 'scale=1280:720:flags=lanczos',
           '-c:v', 'libx264', '-profile:v', 'high', '-crf', '23',
           '-maxrate', '5M', '-bufsize', '2.5M', '-preset', 'medium',
           '-pix_fmt', 'yuv420p', '-r', str(FPS), '-an', str(silent)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    t0 = time.time()
    for i in range(a_frames):
        img = L.render_a_frame3(sched, i / FPS)
        p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        if i % 60 == 0:
            print(f'A frame {i}/{a_frames} {time.time()-t0:.0f}s', flush=True)
    # B segment reads the solid-look scene still from assets/
    Image.fromarray(L.render_a_frame3(sched, 0, still=True)).save(
        L.WORK / 'scene_still.png')
    for i in range(b_frames):
        img = B.render_b_frame(i / FPS, b_dur, drop)
        p.stdin.write(np.ascontiguousarray(img, dtype=np.uint8).tobytes())
        if i % 60 == 0:
            print(f'B frame {i}/{b_frames} {time.time()-t0:.0f}s', flush=True)
    p.stdin.close(); p.wait()
    assert p.returncode == 0, p.returncode
    print(f'render+encode done {time.time()-t0:.0f}s', flush=True)
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else OUTD
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / 'echo_promo_main_sample.mp4'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(silent),
                    '-i', str(EXP / 'promo_music.m4a'),
                    '-map', '0:v:0', '-map', '1:a:0', '-c', 'copy',
                    '-movflags', '+faststart', str(out)], check=True)
    print(f'shipped -> {out}', flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'stills':
        stills()
    else:
        full()
