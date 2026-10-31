"""The super-function.

    scene = super_call(anything, ..., target="scene")

- Variable inputs: any number, any mix — paths, payloads,
  (kind, data) tuples, EchoObjects. Each is wrapped uniformly.
- Routing: the registry finds the fitting path from the kinds on
  hand to the target. Steps already satisfied are skipped.
- Restart: if a kernel fails, its name is excluded and a new path
  is searched (Lisp-style restart, simplified) before giving up.
- The Plan (Lisp layer) is always produced and attached to the
  result's meta, executed or not, so every output carries the
  record of how it was made and can be replayed.
"""
from __future__ import annotations

from .core import EchoObject, wrap
from .plan import Plan
from .registry import NoPath, Registry


def make_super(registry: Registry):
    def super_call(*inputs, target: str = "scene", execute: bool = True,
                   **wrap_kw) -> EchoObject:
        objects: dict[str, EchoObject] = {}
        for raw in inputs:
            obj = wrap(raw, **wrap_kw) if not isinstance(raw, EchoObject) else raw
            objects[obj.kind] = obj
        if not execute:
            path = registry.find_path(objects.keys(), target)
            plan = Plan(steps=[(k.name, k.consumes, k.produces) for k in path],
                        target=target, inputs=sorted(objects))
            return EchoObject(kind=target, data=None, source="super:dry-run",
                              meta={"plan": plan.as_data(), "sexp": plan.to_sexp()})

        excluded, last_err = set(), None
        while True:
            try:
                path = registry.find_path(objects.keys(), target, exclude=frozenset(excluded))
            except NoPath:
                if last_err is not None:
                    raise last_err
                raise
            plan = Plan(steps=[(k.name, k.consumes, k.produces) for k in path],
                        target=target, inputs=sorted(objects))
            try:
                for k in path:
                    args = [objects[c] for c in k.consumes]
                    out = k.fn(*args)
                    obj = out if isinstance(out, EchoObject) else \
                        args[0].derive(k.produces, out, source=f"kernel:{k.name}")
                    if obj.kind != k.produces:
                        obj = EchoObject(kind=k.produces, data=obj.data, t=obj.t,
                                          frame=obj.frame, source=obj.source, meta=obj.meta)
                    objects[k.produces] = obj
                result = objects[target]
                result.meta["plan"] = plan.as_data()
                result.meta["sexp"] = plan.to_sexp()
                return result
            except Exception as e:  # restart: drop the failing kernel, re-route
                failing = k.name
                excluded.add(failing)
                last_err = e
                # keep objects gathered so far; the new path reuses them
                continue

    return super_call
