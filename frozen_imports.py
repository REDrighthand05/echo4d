"""Build-time dependency magnet.

The shipped exe's entry (launcher.py) imports this module so
PyInstaller's analysis pulls the whole third-party stack into the
frozen base: the product's own code ships loose under ``app/`` and
imports these libraries at runtime, so they must already be inside
the archive. Keep this list in step with requirements-app.txt --
the Windows workflow's bundle guards (open3d, scipy) exist because
a missing entry here fails silently until a user hits the stage
that needs it.

Not imported: torch / transformers. The discovery module's model
nomination half needs them, they are deliberately NOT in the
bundle (they would add ~900MB), and discovery degrades to its
measured adjudication half without them.
"""
import cv2  # noqa: F401
import customtkinter  # noqa: F401
import numba  # noqa: F401
import numpy  # noqa: F401
import onnxruntime  # noqa: F401
import open3d  # noqa: F401
import PIL  # noqa: F401
import pycolmap  # noqa: F401
import pystray  # noqa: F401
import scipy  # noqa: F401
import tkinterdnd2  # noqa: F401
import webview  # noqa: F401

try:
    import winotify  # noqa: F401  (Windows-only toast library)
except Exception:
    pass
