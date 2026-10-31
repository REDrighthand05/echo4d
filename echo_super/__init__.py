"""echo_super — permanent infrastructure, independent of the live pipeline.

Layers:
  core      uniform object (PyObject-like): kind + data + t + frame + source
  registry  strongly-bounded kernels declare consumes/produces; routes
            are found by type, Make/blackboard style
  plan      integration plan as data (Lisp layer): savable, replayable,
            rewritable S-expressions
  super_fn  one general entry: variable inputs in, target out
"""
from .core import EchoObject, infer_kind, wrap
from .plan import Plan, parse_sexp, to_sexp
from .registry import Kernel, NoPath, Registry
from .super_fn import make_super

__all__ = ["EchoObject", "infer_kind", "wrap", "Plan", "parse_sexp",
           "to_sexp", "Kernel", "NoPath", "Registry", "make_super"]
