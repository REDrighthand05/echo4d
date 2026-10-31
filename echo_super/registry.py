"""Kernel registry + type-driven path finding (the Make / blackboard layer).

A kernel is a strongly-bounded small function — the kind DS Flash
produces cheaply and reliably. It declares exactly:

    consumes : tuple of EchoObject kinds it needs
    produces : the single kind it returns
    fn       : callable(*EchoObject) -> payload | EchoObject

The registry finds a route from whatever kinds are on hand to a
target kind by BFS over kind-sets, like `make` over a dependency
graph. Kernels never know about each other; they only know kinds.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Callable


@dataclass
class Kernel:
    name: str
    consumes: tuple[str, ...]
    produces: str
    fn: Callable
    cost: float = 1.0  # relative; cheaper routes win ties


class NoPath(Exception):
    pass


class Registry:
    def __init__(self):
        self.kernels: dict[str, Kernel] = {}

    def register(self, name, consumes, produces, cost=1.0):
        def deco(fn):
            self.kernels[name] = Kernel(name, tuple(consumes), produces, fn, cost)
            return fn
        return deco

    def add(self, kernel: Kernel):
        self.kernels[kernel.name] = kernel

    def find_path(self, have, target, exclude=frozenset()):
        """Shortest (then cheapest) kernel sequence turning `have` kinds
        into one that includes `target`. Returns [] if already there."""
        have = frozenset(have)
        if target in have:
            return []
        queue = deque([(have, [])])
        # A kind-set can be reached by different kernels (plain vs
        # IMU-stabilized trajectory); keep the best (length, cost)
        # known per state and allow a cheaper route to revisit it.
        def score(path):
            return (len(path), sum(x.cost for x in path))

        best_to = {have: (0, 0.0)}
        best = None
        while queue:
            state, path = queue.popleft()
            if best is not None and len(path) >= len(best):
                continue
            if score(path) > best_to.get(state, (1 << 30, float("inf"))):
                continue  # stale entry; a better route to this state exists
            for k in self.kernels.values():
                if k.name in exclude or k.produces in state:
                    continue
                if set(k.consumes) <= state:
                    new = state | {k.produces}
                    npath = path + [k]
                    if target in new:
                        if best is None or score(npath) < score(best):
                            best = npath
                        continue
                    if score(npath) < best_to.get(new, (1 << 30, float("inf"))):
                        best_to[new] = score(npath)
                        queue.append((new, npath))
        if best is None:
            raise NoPath(f"no route from {sorted(have)} to {target!r}; "
                         f"registered: {sorted(self.kernels)}")
        return best
