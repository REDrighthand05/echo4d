"""ECHO: known-pose RGB-D surface reconstruction."""
__version__ = "0.1.0"
from .model import Camera, Frame, Recording, DataError, load_recording
from .fusion import FusionConfig, TSDFVolume
