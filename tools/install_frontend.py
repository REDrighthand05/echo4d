#!/usr/bin/env python3
"""Safely provision the canonical Echo4D frontend assets in a branch worktree.

Does not commit, push, change unrelated files, or rewrite frozen experiment refs.
For branch preview CI, --force only changes the ephemeral checked-out workspace.
"""
import argparse
import hashlib
import pathlib
import subprocess
import sys

ASSETS = ("dist/quality.html", "dist/quality.css", "dist/quality.mjs")

def git(*args):
    return subprocess.run(["git", *args], check=True, capture_output=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="origin/main", help="trusted local/fetched Git ref")
    parser.add_argument("--check", action="store_true", help="verify files are identical; do not modify")
    parser.add_argument("--force", action="store_true", help="replace divergent assets ONLY in current worktree")
    options = parser.parse_args()
    root = pathlib.Path.cwd()
    try:
        worktree = pathlib.Path(git("rev-parse", "--show-toplevel").stdout.decode().strip()).resolve()
    except (subprocess.CalledProcessError, OSError):
        parser.error("run inside a git working tree")
    if root.resolve() != worktree:
        parser.error("run from repository root")
    failed = False
    for relative in ASSETS:
        try:
            canonical = git("show", options.source + ":" + relative).stdout
        except subprocess.CalledProcessError:
            print("MISSING SOURCE " + options.source + ":" + relative, file=sys.stderr)
            failed = True
            continue
        destination = root / relative
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            print("BLOCKED unsafe target: " + relative, file=sys.stderr)
            failed = True
            continue
        local = destination.read_bytes() if destination.is_file() else None
        if local == canonical:
            print("CURRENT " + relative)
            continue
        if options.check:
            print("OUTDATED " + relative)
            failed = True
            continue
        if local is not None and not options.force:
            print("CONFLICT (unchanged): " + relative, file=sys.stderr)
            failed = True
            continue
        # An explicit force cannot silently overwrite uncommitted local changes.
        if local is not None:
            dirty = git("status", "--porcelain", "--", relative).stdout
            if dirty:
                print("BLOCKED uncommitted local edits: " + relative, file=sys.stderr)
                failed = True
                continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(canonical)
        sha = hashlib.sha256(canonical).hexdigest()
        print("INSTALLED " + relative + " sha256=" + sha)
    return 1 if failed else 0

if __name__ == "__main__":
    sys.exit(main())
