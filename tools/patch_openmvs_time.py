"""Backport only the upstream formatTime remaining-buffer fix to OpenMVS 2.3.

Source: cdcseacave/openMVS@58117204, libs/Common/Util.h; issues #1136/#1241.
No geometry setting or glibc fortify protection is changed.
"""
import hashlib
import json
from pathlib import Path
import sys


def patch(text):
    start = text.index('static String formatTime(')
    end = text.index('\n\tstatic String toString(', start)
    body = text[start:end]
    old = '_sntprintf(buf+len, 128,'
    if body.count(old) != 5:
        raise ValueError('Not the pinned OpenMVS 2.3 formatTime implementation')
    return text[:start]+body.replace(old,'_sntprintf(buf+len, 128-len,')+text[end:]


if __name__ == '__main__':
    path = Path(sys.argv[1])/'libs/Common/Util.h'
    before = path.read_text()
    after = patch(before)
    path.write_text(after)
    print(json.dumps({'patch':'upstream-formatTime-remaining-buffer',
                      'before_sha256':hashlib.sha256(before.encode()).hexdigest(),
                      'after_sha256':hashlib.sha256(after.encode()).hexdigest(),
                      'geometry_changed':False, 'fortify_disabled':False}))
