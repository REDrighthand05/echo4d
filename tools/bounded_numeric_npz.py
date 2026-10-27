"""Read numeric research assets only after ZIP and NPY allocation preflight."""
from __future__ import annotations

import math
from pathlib import Path
import zipfile

import numpy as np

MAX_BYTES = 160 * 1024**2


def read_numeric_npz(path: Path, schema: dict) -> dict[str, np.ndarray]:
    """schema maps names to (minimum rows, maximum rows, columns, dtype kinds).

    Inspect every member/header before materializing any array. Use the same
    open archive for validation and reads, reject aliases and object dtypes.
    """
    if path.stat().st_size > MAX_BYTES:
        raise ValueError('Numeric NPZ compressed size exceeds budget')
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            expected = {name + '.npy' for name in schema}
            if (len(members) != len(expected) or
                    {member.filename for member in members} != expected):
                raise ValueError('Numeric NPZ has missing, duplicate or extra members')
            if any(m.file_size > MAX_BYTES or m.flag_bits & 1 for m in members) or sum(
                    m.file_size for m in members) > MAX_BYTES:
                raise ValueError('Numeric NPZ expanded size exceeds budget')
            for member in members:
                minimum, maximum, columns, kinds = schema[member.filename[:-4]]
                with archive.open(member) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, _, dtype = np.lib.format.read_array_header_1_0(stream)
                    elif version == (2, 0):
                        shape, _, dtype = np.lib.format.read_array_header_2_0(stream)
                    else:
                        raise ValueError('Unsupported numeric NPY header version')
                    if (len(shape) != 2 or shape[1] != columns or
                            not minimum <= shape[0] <= maximum or
                            dtype.hasobject or dtype.kind not in kinds or dtype.itemsize > 8):
                        raise ValueError('Numeric NPY shape/dtype exceeds schema budget')
                    payload = math.prod(shape) * dtype.itemsize
                    if payload > MAX_BYTES or stream.tell() + payload != member.file_size:
                        raise ValueError('Numeric NPY payload size differs from header')
            result = {}
            for member in members:
                with archive.open(member) as stream:
                    result[member.filename[:-4]] = np.load(stream, allow_pickle=False)
            return result
    except (zipfile.BadZipFile, EOFError, OSError, KeyError) as exc:
        raise ValueError('Invalid numeric NPZ archive') from exc
