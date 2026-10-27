"""Input evidence matrix and user-directed reconstruction route selection.

This module DOES NOT reconstruct a scene or verify declarations of sensor
quality. It exposes all compatible and incompatible strategies, including
paths not implemented in ECHO, so agents cannot silently narrow the roadmap
to the pipeline they most recently worked on.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import DataError

PROFILE_VERSION = "echo-strategy-profile-1"
REPORT_VERSION = "echo-strategy-matrix-1"
STATUSES = frozenset({"absent", "present", "checked"})
GOALS = frozenset({"reconstruct_scene", "investigate_limits", "inspect_existing_assets"})

INPUT_CATALOG: tuple[dict[str, str], ...] = (
    {"id": "rgb_video", "label": "原始彩色视频", "check": "原始分辨率、帧率/VFR PTS、拍摄覆盖、图像质量、逐源哈希"},
    {"id": "rgb_images", "label": "多视角照片", "check": "视角重叠、原图分辨率、拍摄时序、逐图哈希；单张需明说"},
    {"id": "imu", "label": "IMU惯性观测", "check": "采样率、坐标轴、陀螺/加计零偏、外参、时间偏移"},
    {"id": "camera_calibration", "label": "相机内参/畸变", "check": "对应设备/变焦模式、成像坐标、畸变/去畸变及分辨率"},
    {"id": "camera_poses", "label": "已有相机位姿", "check": "世界系、world-to-camera或反向、坐标手性、每帧对应时间"},
    {"id": "time_sync", "label": "传感器时间/帧同步", "check": "真实时钟/PTS、同步误差及漂移、每个观测的匹配"},
    {"id": "sensor_depth", "label": "实测深度/RGB-D", "check": "深度类型camera-Z或ray、单位、像素配准、有效区域和来源"},
    {"id": "metric_anchor", "label": "独立实测尺度锚点", "check": "测量方法、单位、位姿/深度共坐标；绝非事后GT拟合"},
    {"id": "existing_mesh", "label": "已有Mesh/场景资产", "check": "来源、空间范围、物理尺度、时间版本、孔洞和合法授权"},
    {"id": "mesh_alignment", "label": "Mesh与观测的已验证对齐", "check": "同世界坐标、绝对/相对位姿、尺度、时间有效性、残差"},
    {"id": "lidar_points", "label": "激光扫描/点云", "check": "坐标标定、扫描密度、时间漂移、覆盖和单位"},
    {"id": "semantic_masks", "label": "物体掩码/语义标注", "check": "跨帧同实例来源、动态/静态类别、人工核对"},
    {"id": "independent_truth", "label": "独立几何测量/真值", "check": "没有参与优化、时间/坐标配准、独立测量协议"},
    {"id": "audio", "label": "音频/事件声学", "check": "同步与许可；不能独自当成可见物体的几何真值"},
    {"id": "gps", "label": "GPS/GNSS位置", "check": "室内外精度、地理坐标与场景系变换、时间同步"},
)
INPUT_IDS = frozenset(entry["id"] for entry in INPUT_CATALOG)

# Every requirement group is OR: one source within each group must be present.
# Routes are catalogued from information-rich combinations to sparse inputs.
# This is NOT a scalar difficulty ranking and NEVER an automatic recommendation.
ROUTES: tuple[dict[str, Any], ...] = (
    {
        "id": "rich_video_imu_depth_mesh",
        "name": "已有Mesh + 视频/IMU + 实测深度多源对齐与增量融合",
        "tier": "information_rich",
        "requires": (("rgb_video",), ("imu",), ("existing_mesh",),
                     ("camera_calibration",), ("sensor_depth",), ("time_sync",)),
        "optional": ("mesh_alignment", "camera_poses", "metric_anchor",
                     "lidar_points", "semantic_masks", "independent_truth", "gps"),
        "implementation": "not_integrated",
        "evidence": "E0: 成熟技术可组合的候选，ECHO尚无此完整集成实跑",
        "components": "视觉惯性优化/外部VIO或SLAM + Mesh配准 + 深度融合；仅将现有TSDF当子模块",
        "output": "在已配准、观测覆盖范围内复用/更新空间结构，保留旧Mesh版本",
        "unknown": "原Mesh未拍到部分、时间变化和被遮挡的真实结构不能自动证实",
        "checks": ("Mesh是否和视频处于同世界系、同年代且有观测支持",
                   "IMU外参/时间同步、RGB-D配准和测深单位",
                   "对已有Mesh的保留/替换/冲突区域建立来源账本"),
        "links": ("#4", "#15", "#74"),
    },
    {
        "id": "video_imu_mesh_no_depth",
        "name": "视频 + IMU + 既有Mesh（不要求密集测深）",
        "tier": "information_rich",
        "requires": (("rgb_video",), ("imu",), ("existing_mesh",),
                     ("camera_calibration",), ("time_sync",)),
        "optional": ("mesh_alignment", "camera_poses", "metric_anchor",
                     "semantic_masks", "independent_truth"),
        "implementation": "not_integrated",
        "evidence": "E0: 候选集成，未完成真实图像到现有Mesh的位姿/尺度注册",
        "components": "视觉惯性里程计 + image-to-model定位/图像-已知Mesh配准",
        "output": "已有结构上的视频定位与被观测到的差异标记；必要时保留未解区域",
        "unknown": "不能凭IMU或旧Mesh推断所有新出现、遮挡或缺失表面",
        "checks": ("检测旧Mesh是静态结构还是另一时间的资产",
                   "实际完成图像到Mesh尺度、坐标与遮挡一致性的注册",
                   "用未参与拟合的机位/独立测量检验位置"),
        "links": ("#4", "#74"),
    },
    {
        "id": "video_mesh_alignment",
        "name": "视频 + 已有Mesh，进行图像到模型配准",
        "tier": "information_rich",
        "requires": (("rgb_video",), ("existing_mesh",), ("camera_calibration",)),
        "optional": ("imu", "time_sync", "camera_poses", "mesh_alignment",
                     "independent_truth"),
        "implementation": "not_integrated",
        "evidence": "E0: 组合候选，ECHO尚未证明原视频和现成Mesh可以稳定配准",
        "components": "SfM/局部特征定位 + 已知网格投影/对齐核验",
        "output": "已有Mesh可见部分的机位对齐和可审查空间关系",
        "unknown": "旧Mesh不包含的真实部分仍不确定；无IMU时退化运动可能失效",
        "checks": ("Mesh的尺度、姿态、拍摄时间、覆盖与来源",
                   "处理遮挡/纹理重复/新旧场景变化，逐视角保留FAIL"),
        "links": ("#4", "#74"),
    },
    {
        "id": "known_pose_measured_depth_tsdf",
        "name": "已知位姿 + 实测RGB-D → TSDF Mesh",
        "tier": "measured_geometry",
        "requires": (("rgb_video", "rgb_images"), ("camera_poses",),
                     ("sensor_depth",), ("camera_calibration",), ("time_sync",)),
        "optional": ("existing_mesh", "imu", "metric_anchor",
                     "semantic_masks", "independent_truth"),
        "implementation": "integrated_experiment",
        "evidence": "E3: ECHO现有已知相机位姿+测深TSDF已有TUM真实数据实验",
        "components": "echo4d import-tum / import-colmap / reconstruct；现有Mesh不会自动参与融合",
        "output": "有已测量深度支持的可观察静态Mesh；不是保证完整房间",
        "unknown": "未观察区域、错误标定、移动物体与独立表面真值",
        "checks": ("确认相机位姿与实测深度同时间、同像素、同尺度",
                   "输出原分辨率、三原机位/两自由机位和未融合视角验收",
                   "明确已有Mesh与IMU若提供，目前本路线不会使用它们"),
        "links": ("#3", "#15", "#74"),
    },
    {
        "id": "video_sensor_scaled_tsdf",
        "name": "视频估计位姿 + 实测传感器尺度/深度 → Mesh",
        "tier": "measured_geometry",
        "requires": (("rgb_video",), ("sensor_depth",),
                     ("camera_calibration",), ("time_sync",)),
        "optional": ("metric_anchor", "imu", "existing_mesh", "independent_truth"),
        "implementation": "integrated_experiment",
        "evidence": "E3: TUM条件下COLMAP+独立传感器尺度+实测Z融合已实跑；desk冻结评分FAIL",
        "components": "ECHO video_sfm / metric_anchor / metric_rgbd / TSDF / metric_holdout",
        "output": "在指定测量条件下产生有来源的静态Mesh与未融合深度分数",
        "unknown": "不能以同设备测深当独立表面真值；没有传感器锚点不能得米制",
        "checks": ("尺度必须来自另行锁定的实测锚点，不能用GT Sim(3)",
                   "模型可辨认、留出几何、原分辨率展示仍是独立门禁"),
        "links": ("#15", "#74"),
    },
    {
        "id": "lidar_video_registration",
        "name": "实测点云 + 视频 → 带来源的几何对齐",
        "tier": "measured_geometry",
        "requires": (("rgb_video",), ("lidar_points",),
                     ("camera_calibration",), ("time_sync",)),
        "optional": ("imu", "camera_poses", "existing_mesh", "independent_truth"),
        "implementation": "not_integrated",
        "evidence": "E0: ECHO没有已实跑的LiDAR/视频联合配准工作流",
        "components": "点云-图像外参估计、时序对齐、已测几何融合（待接入）",
        "output": "已扫描到的表面可得到有测距支持的3D资产",
        "unknown": "稀疏扫描之外、运动造成的伪影与资产时效性",
        "checks": ("标定/同步/点云单位和传感器视野重叠",
                   "不得把LiDAR扫描结果等同于此前未知区域已观测"),
        "links": ("#3", "#4"),
    },
    {
        "id": "video_imu_no_depth",
        "name": "视频 + IMU → 惯性辅助位姿，几何另求",
        "tier": "geometry_sparse",
        "requires": (("rgb_video",), ("imu",), ("camera_calibration",), ("time_sync",)),
        "optional": ("metric_anchor", "existing_mesh", "semantic_masks",
                     "independent_truth"),
        "implementation": "not_integrated",
        "evidence": "E0: VIO/VI-SLAM尚未集成，不能声称ECHO已利用IMU",
        "components": "可评估外部VIO/VI-SLAM，另选多视角/预测深度几何模块",
        "output": "候选相机运动约束；是否有绝对尺度须看可观测性和校准",
        "unknown": "IMU不测密集几何；弱纹理和长时间漂移需实测检验",
        "checks": ("真实IMU与相机外参、时钟、噪声偏置",
                   "独立验证尺度和可见/不可见表面的分层"),
        "links": ("#4", "#15"),
    },
    {
        "id": "rgb_video_sfm_only",
        "name": "只有标定视频 → RGB SfM稀疏位姿",
        "tier": "rgb_only",
        "requires": (("rgb_video",), ("camera_calibration",)),
        "optional": ("metric_anchor", "existing_mesh", "independent_truth"),
        "implementation": "partial",
        "evidence": "E3: ECHO RGB-only COLMAP稀疏相机轨迹有实跑；完整表面/米制未知",
        "components": "video_sfm/COLMAP；MVS/预测深度是另一段尚未验证的实现",
        "output": "相对相机轨迹与稀疏观测；非完整米制房间",
        "unknown": "绝对尺度、遮挡表面、可靠稠密结构和历史状态",
        "checks": ("检查失败/未注册帧和退化运动，不以事后真值尺度作为产品尺度",
                   "若扩展稠密几何须另起可审计的实验与可见区域验证"),
        "links": ("#11", "#15", "#74"),
    },
    {
        "id": "multi_view_images",
        "name": "多视角照片 → 特征几何 / 深度预测候选",
        "tier": "rgb_only",
        "requires": (("rgb_images",),),
        "optional": ("camera_calibration", "camera_poses", "metric_anchor",
                     "independent_truth"),
        "implementation": "partial",
        "evidence": "E1: 现有定位支持带时间戳图片；多视角完整Mesh候选尚未真实验收",
        "components": "图像SfM、可选MVS或外部预测模型（需各自做实测）",
        "output": "在有视角重叠处估计可见几何，视角数量不足时仅为假设",
        "unknown": "真实尺度、未拍到的物体背面及没有视差的深度",
        "checks": ("不少于两个足够有视差的实拍视角才评价多视角几何",
                   "单张图片必须标注为单视角推测，不能冒充真实可旋转重建"),
        "links": ("#1", "#4", "#74"),
    },
    {
        "id": "prior_mesh_review_only",
        "name": "只有现成Mesh → 审阅来源/显示（并非新重建）",
        "tier": "asset_only",
        "requires": (("existing_mesh",),),
        "optional": ("rgb_video", "rgb_images", "mesh_alignment", "independent_truth"),
        "implementation": "partial",
        "evidence": "E1/E2: ECHO能读部分限定格式Mesh，未保证任意资产直接可视化",
        "components": "现有几何IO/检查台，仅可验证格式和可用观测支持",
        "output": "查看和审计资产本身，不能由此确认它属于当前实际场景",
        "unknown": "现实场景是否仍与模型一致、未扫描区域和来源可信度",
        "checks": ("必须核对原始资产来源、尺寸单位、授权与时间版本",),
        "links": ("#4", "#74"),
    },
)

_ROUTE_IDS = frozenset(route["id"] for route in ROUTES)


def _load_json_profile(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 128 * 1024:
        raise DataError("Missing or oversized strategy profile (max 128KiB)")
    def no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise DataError("Duplicate field in strategy profile JSON")
            result[key] = value
        return result
    try:
        payload = json.loads(path.read_text(encoding="utf-8"),
                             object_pairs_hook=no_duplicate_keys)
    except (ValueError, UnicodeError) as exc:
        raise DataError("Invalid strategy profile JSON") from exc
    if not isinstance(payload, dict):
        raise DataError("Strategy profile must be a JSON object")
    return payload


def validate_profile(payload: dict[str, Any]) -> dict[str, Any]:
    """Strict user declarations; 'checked' is a CLAIM, not tool verification."""
    if not isinstance(payload, dict) or payload.get("version") != PROFILE_VERSION:
        raise DataError("Unsupported strategy profile version")
    if set(payload) - {"version", "goal", "inputs", "choice"}:
        raise DataError("Unrecognized strategy profile fields")
    goal = payload.get("goal", "investigate_limits")
    if not isinstance(goal, str) or goal not in GOALS:
        raise DataError("Unknown strategy goal")
    given = payload.get("inputs")
    if not isinstance(given, dict) or not given:
        raise DataError("Strategy profile needs a nonempty inputs map")
    if set(given) - INPUT_IDS:
        raise DataError("Unrecognized input modality; update the strategy catalog first")
    states = {key: {"status": "absent"} for key in INPUT_IDS}
    for key, value in given.items():
        if not isinstance(value, dict) or set(value) - {"status", "evidence", "notes"}:
            raise DataError(f"Invalid input declaration: {key}")
        status = value.get("status")
        if not isinstance(status, str) or status not in STATUSES:
            raise DataError(f"Input {key} must be absent, present, or checked")
        evidence = value.get("evidence", "")
        notes = value.get("notes", "")
        if not isinstance(evidence, str) or not isinstance(notes, str):
            raise DataError(f"Invalid evidence/notes for {key}")
        if len(evidence) > 2048 or len(notes) > 2048:
            raise DataError("Input evidence/notes are too long")
        if status == "checked" and not evidence.strip():
            raise DataError(f"Checked input {key} requires an evidence pointer")
        states[key] = {"status": status}
        if evidence.strip():
            states[key]["evidence"] = evidence.strip()
        if notes.strip():
            states[key]["notes"] = notes.strip()
    choice = payload.get("choice", {})
    if not isinstance(choice, dict) or set(choice) - {"primary", "backup"}:
        raise DataError("choice supports only primary and backup")
    if any(key in choice and (not isinstance(choice[key], str) or choice[key] not in _ROUTE_IDS)
           for key in ("primary", "backup")):
        raise DataError("User selected an unknown strategy route")
    if choice.get("primary") and choice.get("primary") == choice.get("backup"):
        raise DataError("Primary and backup routes must differ")
    return {"version": PROFILE_VERSION, "goal": goal, "inputs": states,
            "choice": {k: choice[k] for k in ("primary", "backup") if k in choice}}


def _route_result(route: dict[str, Any], states: dict[str, Any]) -> dict[str, Any]:
    present = {k for k, v in states.items() if v["status"] != "absent"}
    missing = [list(group) for group in route["requires"] if not present.intersection(group)]
    required = {k for group in route["requires"] for k in group}
    unused = sorted(present - required - set(route["optional"]))
    return {
        "id": route["id"], "name": route["name"], "tier": route["tier"],
        "input_match": "candidate" if not missing else "missing_inputs",
        "missing_required_groups": missing,
        "implementation": route["implementation"],
        "evidence": route["evidence"],
        "components": route["components"],
        "cost_on_this_input": "not_measured",
        "external_dependency_license_review": "not_performed_for_chosen_versions",
        "possible_output": route["output"],
        "still_unknown": route["unknown"],
        "checks_not_performed": list(route["checks"]),
        "declared_inputs_unused_in_this_route": unused,
        "links": list(route["links"]),
    }


def strategy_plan(payload: dict[str, Any]) -> dict[str, Any]:
    """No ranking, automatic selection, performance prediction, or reconstruction."""
    profile = validate_profile(payload)
    states = profile["inputs"]
    rows = [_route_result(route, states) for route in ROUTES]
    available = {key for key, data in states.items() if data["status"] != "absent"}
    ablations = []
    for key in sorted(available):
        ablated = dict(states)
        ablated[key] = {"status": "absent"}
        now_lost = [route["id"] for route, row in zip(ROUTES, rows)
                    if row["input_match"] == "candidate" and
                    _route_result(route, ablated)["input_match"] == "missing_inputs"]
        ablations.append({"removed_input": key, "routes_which_lose_required_input": now_lost,
                          "note": "仅输入依赖消融；不代表真实运行后的精度/难度变化"})
    choice = profile["choice"]
    decisions = []
    for role in ("primary", "backup"):
        if role in choice:
            chosen = next(row for row in rows if row["id"] == choice[role])
            decisions.append({"role": role, "route": chosen["id"],
                              "input_match": chosen["input_match"],
                              "missing_required_groups": chosen["missing_required_groups"],
                              "requires_user_approval_to_run": True})
    return {
        "version": REPORT_VERSION,
        "profile": profile,
        "route_catalog_order_note": "从高信息到稀疏输入展示，不是单维难度/质量排序",
        "catalog_review": {"date": "2026-10-27",
                           "main_sha": "00e1167b9406aa7d6d8425f6283f4f1cd5772843",
                           "update_required_if_repo_changed": True},
        "no_auto_selection": not bool(choice),
        "user_choices": decisions,
        "routes": rows,
        "input_ablation": ablations,
        "assessment": {
            "scene_reconstructed": "not_evaluated",
            "image_view_and_free_camera_visual_gate": "not_evaluated",
            "independent_geometry_gate": "not_evaluated",
            "quality_claims": "none",
            "input_evidence_audited_by_tool": False,
            "notes": [
                "checked只表示输入者宣称已核查及其提供的证据指针，本工具不验签、不访问源文件",
                "候选路线不表示ECHO已实现；not_integrated必须另立真实集成任务",
                "任何路线仍须经过#74清晰原图/多视角/未知区域检查及独立几何评价",
                "不能把已有Mesh当成当时现场的真值，也不能把IMU当成稠密深度",
            ],
        },
    }


def strategy_catalog() -> dict[str, Any]:
    return {"version": REPORT_VERSION,
            "catalog_review": {"date": "2026-10-27",
                               "main_sha": "00e1167b9406aa7d6d8425f6283f4f1cd5772843",
                               "update_required_if_repo_changed": True},
            "input_catalog": list(INPUT_CATALOG),
            "routes": [{**route,
                        "cost_on_this_input": "not_measured",
                        "external_dependency_license_review": "not_performed_for_chosen_versions",
                        "requires": [list(g) for g in route["requires"]],
                        "optional": list(route["optional"]),
                        "checks": list(route["checks"]),
                        "links": list(route["links"])}
                       for route in ROUTES],
            "note": "目录说明候选与现有证据，不自动挑路线、重建或确认传感器质量"}


def render_strategy_markdown(result: dict[str, Any]) -> str:
    """Human-readable version of the exact same machine route catalog."""
    if result.get("version") != REPORT_VERSION:
        raise DataError("Not an ECHO strategy result")
    if "profile" not in result:
        lines = ["# ECHO 输入信息与策略候选矩阵", "",
                 "候选顺序不是单维难度排序；实现成熟度是**本仓库**状态，不是外部算法能力。", "",
                 "| 策略 | 必需输入（每组任选一个） | 仓库支持 | 成本/外部许可 | 能得到什么 / 不能证明什么 |",
                 "|---|---|---|---|---|"]
        for route in result["routes"]:
            req = " + ".join("/".join(g) for g in route["requires"])
            lines.append(f"| {route['name']} | {req} | {route['implementation']} ({route['evidence']}) | "
                         f"本输入未测 / 依赖许可待核 | {route['output']}；未知：{route['unknown']} |")
        lines.extend(["", "## 输入类型与核验", "",
                      "| 类型 | 需要核对 |", "|---|---|"])
        for item in result["input_catalog"]:
            lines.append(f"| {item['label']} (\`{item['id']}\`) | {item['check']} |")
        return "\n".join(lines) + "\n"
    lines = ["# ECHO 选择路线前的输入情况", "",
             "这是**声明式路线候选分析**，未实际读取传感器、运行算法或通过视觉验收。",
             "即便 checked 也仅表示声明者提供证据，不是此程序验过的真实数据。", "",
             "| 路线 | 输入条件 | 在仓库的状态 | 成本和许可证 | 缺少的输入组 |",
             "|---|---|---|---|---|"]
    for row in result["routes"]:
        missing = "；".join("/".join(g) for g in row["missing_required_groups"]) or "无（仍须核验质量/配准）"
        lines.append(f"| {row['name']} | {row['input_match']} | {row['implementation']} | "
                     f"本输入未测/依赖许可待核 | {missing} |")
    lines.extend(["", "## 用户选择（不替用户决定）", ""])
    if not result["user_choices"]:
        lines.append("尚未由用户指定主路线和备选路线。")
    for decision in result["user_choices"]:
        lines.append(f"- {decision['role']}: {decision['route']}；输入条件 {decision['input_match']}")
    lines.extend(["", "## 撤去一类信息的实验设计（仅依赖分析）", ""])
    for entry in result["input_ablation"]:
        names = ", ".join(entry["routes_which_lose_required_input"]) or "当前候选没有因缺此项而失去最低输入资格"
        lines.append(f"- 移除 {entry['removed_input']}：{names}")
    lines.extend(["", "## 尚未验收", "",
                  "原始照片清晰度、已恢复的场景结构、自由第三人称视角、未见帧和独立几何测量，全部 **not_evaluated**；详见 #74。"])
    return "\n".join(lines) + "\n"


def strategy_from_file(path: Path) -> dict[str, Any]:
    return strategy_plan(_load_json_profile(path))
