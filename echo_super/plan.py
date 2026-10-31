"""Plan-as-data — the Lisp layer.

An integration plan is a plain nested list (an S-expression in
Python clothing). It can be executed, saved to a scene's record,
replayed later, diffed, and generated or rewritten by a program —
including by DS Flash or by the router itself — without touching
any kernel's code.

    ["super",
      ["step", "extract", ["video"], "frames"],
      ["step", "depth",   ["frames"], "depth"],
      ...,
      ["target", "scene"]]

to_sexp / parse_sexp round-trip it through the classic textual
form:  (super (step extract (video) frames) ... (target scene))
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class Plan:
    steps: list = field(default_factory=list)  # [(name, consumes, produces)]
    target: str = "scene"
    inputs: list = field(default_factory=list)  # starting kinds

    def as_data(self):
        return ["super",
                *[["step", n, list(c), p] for n, c, p in self.steps],
                ["target", self.target]]

    def to_json(self) -> str:
        return json.dumps({"inputs": self.inputs, "plan": self.as_data()},
                          ensure_ascii=False)

    @staticmethod
    def from_json(text: str) -> "Plan":
        d = json.loads(text)
        steps = [(s[1], tuple(s[2]), s[3]) for s in d["plan"][1:-1]]
        return Plan(steps=steps, target=d["plan"][-1][1], inputs=d.get("inputs", []))

    def to_sexp(self) -> str:
        return to_sexp(self.as_data())


def to_sexp(node) -> str:
    if isinstance(node, (list, tuple)):
        return "(" + " ".join(to_sexp(x) for x in node) + ")"
    return str(node)


def parse_sexp(text: str):
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def parse():
        nonlocal pos
        tok = tokens[pos]; pos += 1
        if tok == "(":
            out = []
            while tokens[pos] != ")":
                out.append(parse())
            pos += 1  # consume )
            return out
        return tok

    result = parse()
    if pos != len(tokens):
        raise ValueError("trailing tokens in sexp")
    return result
