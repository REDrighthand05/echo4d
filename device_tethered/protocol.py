"""ECHO-T1 wire protocol: ESP32-S3 sensor head -> PC, over USB serial.

One binary frame per sensor event. The MCU sends; the PC mostly
listens (one optional downlink byte for the record LED, see README).

Frame layout (all integers little-endian):

    +------+------+-------+-----------+----------+===========+--------+
    | 0xA5 | 0x5A | type  | tick u32  | len u16  | payload   | crc16  |
    | 1B   | 1B   | 1B    | 4B        | 2B       | len bytes | 2B     |
    +------+------+-------+-----------+----------+===========+--------+

  * tick: the MCU's microsecond clock (Arduino micros()), modulo
    2**32 -- it wraps every ~71.6 minutes. The parser unwraps it into
    a monotone int64 by forward-delta accumulation; that is exact as
    long as consecutive frames are less than half a wrap apart in
    true time (at the slowest stream's 15 Hz, the gap is 67 ms --
    five orders of magnitude inside the bound).
  * len: payload length in bytes; the parser rejects len > 4096 as a
    sync error.
  * crc16: CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF, no
    reflection) over type..payload (everything after the sync pair,
    before the CRC itself). A frame whose CRC fails is dropped and
    the parser resynchronises on the next sync pair.

Frame types and payloads:

  * 0x01 IMU    40 bytes: 10 x float32 -- qw,qx,qy,qz (fused
                orientation, world-from-device, scalar-first, the
                record format's convention), ax,ay,az in m/s^2
                (device frame, incl. gravity), gx,gy,gz in rad/s.
                Emitted at 100 Hz, once per BNO085 rotation-vector
                report, carrying the freshest accel/gyro values.
  * 0x02 TOF    128 bytes: 64 x int16 -- zone ranges in whole mm,
                row-major from the imager's top-left, -1 = no target
                (mirrors the record format's tof.csv exactly).
                Emitted per VL53L5CX ranging, 15 Hz at 8x8.
  * 0x03 BUTTON 1 byte: 0x01 = button pressed (debounced on the MCU).
                The PC toggles recording on each press frame.
  * 0x04 HELLO  UTF-8 text, sent once at boot after both sensors
                initialise, e.g. "ECHO-T1 fw=0.1 proto=1 imu=100
                tof=15". Informational; the recorder logs it.

Data rate: IMU 51 B x 100 Hz + TOF 139 B x 15 Hz ~= 7.2 KB/s -- any
USB CDC link carries it with huge margin.

This module is the protocol's reference implementation: the
simulator encodes with it, the recorder parses with it, the tests
lock it, and firmware/echo_collector implements the same bytes in
C++. If the protocol ever changes, change it here and in the README
together (bump the version in the HELLO string).
"""
import struct
from dataclasses import dataclass, field

import numpy as np

SYNC = b"\xA5\x5A"
MAX_PAYLOAD = 4096
TICK_MODULUS = 1 << 32

TYPE_IMU = 0x01
TYPE_TOF = 0x02
TYPE_BUTTON = 0x03
TYPE_HELLO = 0x04

BUTTON_PRESS = 0x01

_HEADER = struct.Struct("<BIH")          # type, tick, len
_CRC = struct.Struct("<H")
IMU_PAYLOAD = struct.Struct("<10f")
TOF_PAYLOAD = struct.Struct("<64h")

# The firmware's HELLO text; the recorder only logs it, tests use it
# to check the simulator behaves like the firmware.
HELLO_TEXT = "ECHO-T1 fw=0.1 proto=1 imu=100 tof=15"


def crc16_ccitt(data: bytes) -> int:
    """CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) & 0xFFFF if crc & 0x8000 \
                else (crc << 1) & 0xFFFF
    return crc


def encode_frame(ftype: int, tick_raw: int, payload: bytes) -> bytes:
    """One wire frame. tick_raw is taken modulo 2**32 (the MCU clock
    wraps; encoding a pre-wrap tick must not raise)."""
    body = _HEADER.pack(ftype, tick_raw % TICK_MODULUS, len(payload)) \
        + payload
    return SYNC + body + _CRC.pack(crc16_ccitt(body))


def encode_imu(tick_raw: int, quat_wxyz, accel, gyro) -> bytes:
    payload = IMU_PAYLOAD.pack(*quat_wxyz, *accel, *gyro)
    return encode_frame(TYPE_IMU, tick_raw, payload)


def encode_tof(tick_raw: int, zones_mm) -> bytes:
    payload = TOF_PAYLOAD.pack(*[int(v) for v in zones_mm])
    return encode_frame(TYPE_TOF, tick_raw, payload)


def encode_button(tick_raw: int, event: int = BUTTON_PRESS) -> bytes:
    return encode_frame(TYPE_BUTTON, tick_raw, bytes([event]))


def encode_hello(tick_raw: int, text: str = HELLO_TEXT) -> bytes:
    return encode_frame(TYPE_HELLO, tick_raw, text.encode("utf-8"))


@dataclass
class Frame:
    """One parsed frame. tick_us is the UNWRAPPED MCU tick (int64
    microseconds since boot, wrap-free); tick_raw is what was on the
    wire (modulo 2**32) and is what the recorder's sidecar stores."""
    ftype: int
    tick_raw: int
    tick_us: int
    payload: bytes = field(repr=False)

    def imu(self):
        """-> (quat_wxyz (4,), accel (3,), gyro (3,)) float64 arrays."""
        v = np.array(IMU_PAYLOAD.unpack(self.payload), dtype=np.float64)
        return v[0:4], v[4:7], v[7:10]

    def tof(self) -> np.ndarray:
        """-> (64,) int16 zone ranges in mm, -1 = no target."""
        return np.array(TOF_PAYLOAD.unpack(self.payload), dtype=np.int16)

    def button_event(self) -> int:
        return self.payload[0] if self.payload else 0

    def hello_text(self) -> str:
        return self.payload.decode("utf-8", errors="replace")


class ProtocolParser:
    """Incremental parser: feed() arbitrary byte chunks, get back the
    frames completed by them. Robust to partial frames at chunk
    boundaries, garbage between frames, and CRC failures (the bad
    frame is dropped, parsing resynchronises on the next sync pair).
    `dropped_bytes` / `bad_frames` count what was skipped, so a noisy
    link is measurable, not silent."""

    def __init__(self):
        self._buf = bytearray()
        self._prev_raw = None
        self._unwrapped = 0
        self.dropped_bytes = 0
        self.bad_frames = 0

    def _unwrap(self, raw: int) -> int:
        if self._prev_raw is None:
            self._unwrapped = raw
        else:
            self._unwrapped += (raw - self._prev_raw) % TICK_MODULUS
        self._prev_raw = raw
        return self._unwrapped

    def feed(self, data: bytes) -> list:
        self._buf += data
        frames = []
        buf = self._buf
        while True:
            start = buf.find(SYNC)
            if start < 0:
                # keep a trailing lone 0xA5: it may be half a sync pair
                keep = 1 if buf.endswith(SYNC[:1]) else 0
                self.dropped_bytes += len(buf) - keep
                del buf[:len(buf) - keep]
                break
            if start:
                self.dropped_bytes += start
                del buf[:start]
            if len(buf) < 2 + _HEADER.size:
                break  # partial header: wait for more bytes
            ftype, tick_raw, length = _HEADER.unpack_from(buf, 2)
            if length > MAX_PAYLOAD:
                # implausible length: this sync pair was a coincidence
                self.bad_frames += 1
                del buf[:1]
                continue
            total = 2 + _HEADER.size + length + _CRC.size
            if len(buf) < total:
                break  # partial frame: wait for more bytes
            body = bytes(buf[2:2 + _HEADER.size + length])
            (crc_seen,) = _CRC.unpack_from(buf, 2 + _HEADER.size + length)
            if crc16_ccitt(body) != crc_seen:
                self.bad_frames += 1
                del buf[:1]  # drop one byte and resynchronise
                continue
            payload = bytes(buf[2 + _HEADER.size:2 + _HEADER.size + length])
            del buf[:total]
            frames.append(Frame(ftype, tick_raw,
                                self._unwrap(tick_raw), payload))
        return frames
