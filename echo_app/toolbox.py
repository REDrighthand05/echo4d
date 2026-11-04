"""Toolbox page data: the super-integration registry, as the app shows it.

The toolbox page (web_ui) must never drift from the code, so nothing
here is a hand-kept list of kernels. The payload is built at runtime
from the same sources of truth the rest of the project already
guards:

  * the kernel set, versions, consumes/produces -- the echo_super
    registry itself, assembled with the same register_* functions
    tests/test_versions_ledger.py uses (plus a sweep for any other
    register_* function, so a future batch lands on the page on its
    own, under a "new" group, instead of silently missing);
  * Chinese names + ledger status -- parsed out of VERSIONS.md's
    machine-checkable KERNEL-TABLE (the single ledger of record);
  * evidence grade + review status -- the KERNEL_META dicts the
    kernels already carry (adapters + the kernel packages);
  * the one-line introductions -- the BLURBS table below, the one
    hand-written part: one honest product-language line per kernel,
    keyed by kernel name. Coverage is gated by
    tests/test_toolbox.py (registry set == BLURBS set, both ways),
    so a kernel registered without its line fails the suite instead
    of silently getting a generated sentence;
  * deployed （已列装） -- decided mechanically: a kernel counts as
    deployed when shipped product code (echo_scene/, echo_app/,
    web_app.py, app.py, launcher.py) references it by name. One
    documented exception lives in MODULE_HINTS below: the MUS
    renderer, whose kernel and product module are the same body in
    two forms (VERSIONS.md board B says so outright), so the check
    looks for the module the kernel wraps.

Grouping follows the register_* function that brought each kernel
in -- the batches ARE the lines the kernels were built in (sensor
batch, hot-update batch, release batch, ...), which matches the
line tags VERSIONS.md itself annotates.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# EchoObject kind -> product-language Chinese, for the I/O line.
# Presentation vocabulary only (a closed set of kinds); an unknown
# future kind falls back to its raw name rather than breaking.
KIND_ZH = {
    "altitude": "高度", "bytes": "文件内容", "candidates": "候选文件",
    "change_set": "变更集", "digest": "摘要指纹", "env": "环境读数",
    "flag": "判定结果", "frame": "画面", "gaussians": "高斯泼溅",
    "heading": "航向", "job": "采集任务", "keyframe_track": "关键帧轨道",
    "known_size": "已知尺寸", "listing": "文件清单",
    "mag_accel": "磁场与加速度", "manifest": "更新清单",
    "manifest_patch": "清单补丁", "mesh": "网格模型", "normals": "法向",
    "objects": "物品清单", "path": "路径", "plan": "更新计划",
    "points": "点云", "poses": "相机位姿", "report": "核对报告",
    "scale": "真尺度", "sound_speed": "声速", "spectrum": "频谱",
    "steps": "执行步骤", "streams": "多路数据流",
    "sync_report": "同步报告", "tof_frame": "测距帧",
    "track_value": "轨道读数",
}

# Kernels whose product face is a module rather than the kernel's
# own name: kernel name -> module string product code imports.
MODULE_HINTS = {
    # VERSIONS.md board B: 场景预览渲染 MUS is echo_render_frame's
    # same body; replay_walk / live_preview import the module.
    "echo_render_frame": "echo_super.kernels.echo_render",
}

_GROUP_FALLBACK = ("new", "新加入")


def _group_plan():
    """(group id, title, [register functions]) in display order."""
    from echo_super import adapters
    from echo_super import kernels as kernel_pkg
    return [
        ("geometry", "几何链",
         [adapters.register_real_kernels]),
        ("scale-replay", "尺度 · 回放 · 体素",
         [adapters.register_ds_batch1]),
        ("path-points", "路径与选取",
         [adapters.register_ds_batch2, adapters.register_ds_batch4]),
        ("sensors", "传感器线",
         [adapters.register_ds_batch3]),
        ("hotupdate", "热更新线",
         [adapters.register_ds_batch5]),
        ("release", "发版快线",
         [adapters.register_ds_batch6]),
        ("render-sense", "渲染 · 接入 · 识物",
         [kernel_pkg.register_echo_render_kernels,
          kernel_pkg.register_gaussian_kernels,
          kernel_pkg.register_photo_discovery_kernels]),
    ]


def _sweep_extra_register_fns(used):
    """Any other register_* callables in adapters / kernels, so a
    future batch's kernels still reach the page (fallback group)."""
    from echo_super import adapters
    from echo_super import kernels as kernel_pkg
    extra = []
    for mod in (adapters, kernel_pkg):
        for name in sorted(dir(mod)):
            if name.startswith("register_"):
                fn = getattr(mod, name)
                if callable(fn) and fn not in used:
                    extra.append(fn)
    return extra


def _assemble_grouped():
    """[(group_id, title, [Kernel, ...]), ...] from the live registry."""
    from echo_super.registry import Registry

    reg = Registry()
    groups = []
    used = []
    for gid, title, fns in _group_plan():
        before = set(reg.kernels)
        for fn in fns:
            fn(reg)
            used.append(fn)
        groups.append((gid, title,
                        [reg.kernels[n] for n in reg.kernels
                         if n not in before]))
    extra = _sweep_extra_register_fns(used)
    if extra:
        before = set(reg.kernels)
        for fn in extra:
            fn(reg)
        new = [reg.kernels[n] for n in reg.kernels if n not in before]
        if new:
            groups.append((_GROUP_FALLBACK[0], _GROUP_FALLBACK[1], new))
    return groups


def _kernel_meta():
    from echo_super import adapters
    from echo_super import kernels as kernel_pkg
    from echo_super.kernels import gaussian_splat, photo_discovery
    meta = {}
    for source in (adapters.KERNEL_META, kernel_pkg.KERNEL_META,
                   gaussian_splat.KERNEL_META, photo_discovery.KERNEL_META):
        meta.update(source)
    return meta


_LEDGER_ROW = re.compile(
    r"\|\s*`([a-z0-9_]+)`\s*([^|]*?)\s*\|\s*v?([0-9]+\.[0-9]+)\s*\|"
    r"\s*([^|]*?)\s*\|")


def _ledger_rows():
    """kernel name -> (Chinese descriptor, ledger status) from
    VERSIONS.md's KERNEL-TABLE block."""
    text = (ROOT / "VERSIONS.md").read_text(encoding="utf-8")
    m = re.search(r"<!-- KERNEL-TABLE-START -->(.*?)"
                  r"<!-- KERNEL-TABLE-END -->", text, re.S)
    rows = {}
    if m:
        for name, desc, _ver, status in _LEDGER_ROW.findall(m.group(1)):
            rows[name] = (desc.strip(), status.strip())
    return rows


def _product_text():
    """All shipped product sources the deployment scan reads."""
    parts = []
    for base in (ROOT / "echo_scene", ROOT / "echo_app"):
        for path in sorted(base.glob("*.py")):
            if path.name == Path(__file__).name:
                continue  # never let the scan fulfil itself
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    for name in ("web_app.py", "app.py", "launcher.py"):
        path = ROOT / name
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts)


def _kind_zh(kind):
    return KIND_ZH.get(kind, kind)


def _display_consumes(kernel):
    """Consumes for display: a kernel eating the same kind twice
    (plan_update takes the old and the new manifest) reads as the
    kind once."""
    seen = []
    for c in kernel.consumes:
        if c not in seen:
            seen.append(c)
    return seen


# One hand-written product-language line per kernel: what the thing
# actually does, in the user's terms. Written from each kernel's own
# implementation (echo_super adapters / ds_batch modules / kernel
# packages), never from its consumes/produces -- a generated "X in,
# Y out" sentence says nothing and is banned here. Adding a kernel
# to the registry without adding its line fails tests/test_toolbox.py.
BLURBS = {
    # 几何链
    "orient_normals": "按每个点的拍摄方向逐个翻正法向，保证表面朝向一致。",
    "fit_plane": "给点云拟合一个整体最贴合的平面，方程和残差都记下来。",
    "poisson_reconstruct": "把带法向的点云重建成连续的网格表面，稀疏边角自动剪除。",
    # 尺度 · 回放 · 体素
    "estimate_scale": "量参照物在点云中的大小，反推整个场景的真实比例。",
    "build_replay_path": "把相机位姿连成平滑的回放路线，每步都算好镜头朝向。",
    "surface_voxel_indices": "挑出体素点云表面那一层的点——挨着空体素的才算。",
    # 路径与选取
    "resample_path_even": "按固定弧长给路线重新取点，让路径上的点疏密均匀。",
    "catmull_rom_path": "用向心样条把折线路径磨成平滑曲线，转弯不甩出去。",
    "frustum_select_points": "按一台相机的取景框选点：投影回画面，框外的不要。",
    "plane_fit_lsq": "拟合最小二乘平面，并给出每点到平面的有符号距离。",
    "voxel_select_representative": "每个体素格只留离格心最近的一个点，给点云均匀瘦身。",
    "sample_track": "按关键帧与缓动曲线，算出任意时刻轨道上该有的值。",
    # 传感器线
    "tof_grid_points": "把 ToF 的 8×8 测距栅格按视场角展开成三维点云。",
    "tilt_compensated_heading": "先用加速度扳平倾斜，再读磁力计，手机歪着拿航向也准。",
    "pressure_altitude_m": "由当前气压和参考气压换算高度，气压计当高度计用。",
    "speed_of_sound_mps": "按气温算出当时的声速，给声学测距备好换算系数。",
    "stream_sync_stats": "对齐几路传感器流的时间戳：各自多密、差主时钟多少。",
    "spectral_ratios": "把光谱 11 通道的读数按总光强归一，亮度变了比值不变。",
    # 热更新线
    "manifest_digest": "给更新清单算规范的 SHA-256 指纹，动没动过一比就知。",
    "plan_update": "对比新旧两份清单，算出这次要换哪些、删哪些文件。",
    "verify_listing": "把磁盘上的实际文件与清单对账，缺失、多余、被改都列出。",
    "order_apply_steps": "把更新计划排成确定顺序：先替换、后删除，步步不错位。",
    # 发版快线
    "merge_manifest": "把补丁条目覆写进基础清单并换上新版本号，得完整新清单。",
    "select_changes": "在候选文件里圈出真正变了的和该删的，凑齐这一版变更集。",
    "align_crlf": "把文件换行统一成 CRLF，发版哈希不再被换行符坑。",
    "looks_text": "先判定一段内容是不是正经 UTF-8 文本，再决定怎么待它。",
    # 渲染 · 接入 · 识物
    "echo_render_frame": "自研 MUS 渲染器把点云直接画成画面，预览回放都靠它。",
    "gaussian_to_points": "把 3DGS 泼溅拆成带颜色的点云，能直接进点云固化。",
    "photo_discovery": "按预置目录在画面里认家具，抬进三维、两帧印证才算数。",
}


def _blurb(kernel):
    """The kernel's hand-written line. Missing means the registry
    grew without its introduction: fail loudly (the bridge turns it
    into the page's error note), never fall back to a template."""
    try:
        return BLURBS[kernel.name]
    except KeyError:
        raise ValueError(
            f"toolbox: kernel {kernel.name} has no hand-written "
            f"blurb in echo_app/toolbox.py BLURBS -- add one; "
            f"tests/test_toolbox.py gates the coverage") from None


def build_toolbox():
    """The full toolbox payload for Bridge.get_toolbox()."""
    meta = _kernel_meta()
    ledger = _ledger_rows()
    product = _product_text()
    groups_out = []
    count = 0
    deployed_count = 0
    for gid, title, kernels in _assemble_grouped():
        cards = []
        for k in kernels:
            km = meta.get(k.name, {})
            zh, _ledger_status = ledger.get(k.name, (k.name, ""))
            deployed = bool(
                re.search(r"\b" + re.escape(k.name) + r"\b", product)
                or (k.name in MODULE_HINTS
                    and MODULE_HINTS[k.name] in product))
            count += 1
            deployed_count += 1 if deployed else 0
            cards.append({
                "id": k.name,
                "name": zh,
                "blurb": _blurb(k),
                "version": k.version,
                "consumes": list(k.consumes),
                "produces": k.produces,
                "io": " ＋ ".join(_kind_zh(c)
                                  for c in _display_consumes(k))
                      + " → " + _kind_zh(k.produces),
                "evidence": km.get("evidence", ""),
                "review": km.get("status", ""),
                "deployed": deployed,
            })
        groups_out.append({"id": gid, "title": title, "kernels": cards})
    return {"count": count, "deployedCount": deployed_count,
            "groups": groups_out}
