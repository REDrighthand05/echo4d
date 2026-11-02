# WO-update-kernels: 热更新纯函数内核（DS batch 5）
stdlib only，纯函数，无 IO，无随机，确定性。同一个文件 ds_batch5.py 四个函数。

manifest 结构（贯穿全部函数）：
{"version": "0.4.0", "files": {"相对路径": {"sha256": "<64位小写hex>", "size": 123}}}
路径一律 POSIX 风格相对路径；出现绝对路径、".." 段、空路径、反斜杠 → ValueError。
manifest 缺 version/files 键、files 值形状不对 → ValueError。

1) manifest_digest(manifest) -> str
canonical = json.dumps({"version": ..., "files": ...}, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
返回 hashlib.sha256(canonical.encode("utf-8")).hexdigest()。

2) plan_update(old, new) -> dict
{"from_version", "to_version",
 "to_change": [{"path", "sha256", "size"}...] 按 path 排序（new 里新增的、或 sha256/size 与 old 不同的）,
 "to_delete": [path...] 排序（old 有 new 无）,
 "unchanged": int（两边完全一致的文件数）,
 "total_bytes": to_change 的 size 总和}

3) verify_listing(manifest, listing) -> dict
listing = 实际磁盘 {path: {"sha256", "size"}}。
{"ok": bool, "missing": [...], "mismatched": [...], "extra": [...]} 均排序。
ok = 无 missing 且无 mismatched；extra（清单外多出的文件）不算失败（用户文件可以住在安装目录里）。

4) order_apply_steps(plan) -> list[dict]
把 plan 变成扳手执行顺序：先全部 {"op": "replace", "path": ...}（按 path 排序），
但路径以 "installer/version.inc" 结尾的 replace 必须排在所有 replace 的最后一位（版本号最后才翻，
半途失败不会谎报新版本）；然后全部 {"op": "delete", "path": ...}（按 path 排序）。
plan 形状不对 → ValueError。
