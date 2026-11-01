#!/usr/bin/env python3
"""Verify a shipped sample: full decode, duration, audio MD5 vs source,
per-second video bitrate peak."""
import json, subprocess, sys
from collections import defaultdict


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def audio_md5(path):
    r = run(['ffmpeg', '-v', 'error', '-i', str(path), '-map', '0:a:0',
             '-c', 'copy', '-f', 'md5', '-'])
    return r.stdout.strip()


def check(mp4, src_audio, expect_dur):
    mp4, src_audio = str(mp4), str(src_audio)
    dec = run(['ffmpeg', '-v', 'error', '-xerror', '-i', mp4, '-f', 'null', '-'])
    pr = run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
              '-of', 'json', mp4])
    dur = float(json.loads(pr.stdout)['format']['duration'])
    pk = run(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
              '-show_entries', 'packet=pts_time,size', '-of', 'csv=p=0', mp4])
    per = defaultdict(int)
    for line in pk.stdout.splitlines():
        t, s = line.split(',')
        per[int(float(t))] += int(s)
    peak = max(per.values()) * 8 / 1e6
    a_out, a_src = audio_md5(mp4), audio_md5(src_audio)
    print(f'{mp4}')
    print(f'  decode: {"OK" if dec.returncode == 0 and not dec.stderr.strip() else "FAIL " + dec.stderr[:200]}')
    print(f'  duration: {dur:.2f}s (track {expect_dur:.2f}s, diff {dur - expect_dur:+.2f}s)')
    print(f'  audio md5: out {a_out} | src {a_src} | {"MATCH" if a_out == a_src else "MISMATCH"}')
    print(f'  peak 1s video bitrate: {peak:.2f} Mbps')


if __name__ == '__main__':
    check(sys.argv[1], sys.argv[2], float(sys.argv[3]))
