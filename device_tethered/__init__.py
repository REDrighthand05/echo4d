"""ECHO Collector, tethered build: ESP32-S3 sensor head + PC recorder.

The handheld head is only sensors (BNO085 IMU + VL53L5CX ToF on an
ESP32-S3, plus a UVC camera straight into the PC); the PC does all
compute, storage and display. This package holds the wire protocol
(protocol.py), the PC recorder (recorder.py) and the hardware-free
loopback simulator (simulator.py). The capture folders it produces are
exactly the format echo_scene/sensor_record.py defines -- that format,
and the importer/payoff built on it, did not change when the Pi design
was superseded by this one.
"""
