# WO batch-6 — quick-line release kernels (for DS Flash)

Pure, stdlib-only, no I/O, no globals mutated. One module:
`echo_super/ds_batch6.py` with exactly these three public functions.
Deterministic; raise ValueError on malformed input shapes.

## 1. merge_manifest(base, overrides, version) -> dict

- `base`: {"version": str, "files": {path: {"sha256": str, "size": int}}}
- `overrides`: {path: {"sha256": str, "size": int} | None}
  (None deletes the entry; a dict adds or replaces it)
- Returns a NEW manifest dict: version replaced by `version`,
  overrides applied. Neither input is mutated (deep copy semantics).
- Validation (ValueError): any path that is empty, absolute,
  contains "\\", or has a ".." part; any sha256 that is not a
  64-char lowercase hex string; any size that is not an int >= 0;
  base/overrides not of the shapes above.

## 2. looks_text(data: bytes) -> bool; align_crlf(data: bytes) -> bytes

- looks_text: True iff there is no NUL byte in the first 8192 bytes
  AND the whole data decodes as UTF-8 (strict). Empty data is text.
- align_crlf: normalize every line ending to CRLF: collapse CRLF
  and lone CR to LF first, then convert LF to CRLF. Must be
  idempotent: align_crlf(align_crlf(x)) == align_crlf(x).

## 3. select_changes(base_files, candidates) -> dict

- `base_files`: {path: {"sha256": str, "size": int}} (one release's
  manifest file map; keys are install-relative POSIX paths, payload
  files start with "app/")
- `candidates`: {path: {"text": bool,
    "lf": {"sha256": str, "size": int},
    "crlf": {"sha256": str, "size": int}}}
  describing every file in the local payload tree (keys all start
  with "app/")
- Returns {"changed": [...], "deleted": [...], "ship": {...}}:
  - A path present in both is UNCHANGED if the base entry equals
    the candidate's "lf" entry or its "crlf" entry (both sha256
    and size); otherwise it is changed.
  - A path only in candidates is changed (new file).
  - A path only in base_files AND starting with "app/" is deleted.
    Base paths outside "app/" are ignored entirely.
  - "ship" maps each changed path to "crlf" when its candidate
    "text" is true, else "raw".
  - "changed" and "deleted" are sorted lists; every changed path
    appears in "ship" and nothing else does.
