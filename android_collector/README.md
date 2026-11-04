# ECHO Collector — Android 采集端

**状态：v0.2，UNTESTED ON HARDWARE（从未在真机上运行过）。**
v0.2 是「第一次真机采集之前」的完善版：位姿转正、流式落盘抗杀、前台服务防杀、
逐帧相机参数、物理超广角选择、内参入册、一键打包回传（详见下文 v0.2 一节）。
本机（无 GPU 的 Linux VM）只能做到：JVM 单元测试全过 + 桌面 harness 用应用自身的
转换/写盘代码从模拟数据生成完整采集文件夹、被 PC 端 Python 读取器与原 payoff
原样验证通过 + APK 编译打包签名验证通过。真机行为见文末「无法在此验证的边界」。

这是 ECHO 采集端的第三个实现（前两个：树莓派方案、ESP32 系留方案，均已被取代），
**记录格式不变**：产出的文件夹与 `device_tethered` 的 PC 录像端字节级同构，
直接喂 `echo_scene/sensor_record.py` 的读取器和上层重建管线，一个字不用改。

## 它做什么

打开 App → 顶部一行能力清单（机型 + 选中的摄像头/FoV + 深度后端）→ 一个大录制按钮。
录制中显示：已录时长、各流实时计数、剩余空间。停止后在 App 私有目录生成一个采集
文件夹（`Android/data/echo.collector/files/captures/capture_yyyyMMdd_HHmmss/`），
拷回电脑即可用现有的导入/payoff 验证。第一次打开时还会写一份完整的
`capabilities.json`（传感器清单与最高频率、每个摄像头的特性、深度可用性）在同一目录
—— 这就是我们了解这台一加到底有什么硬件的途径。

## 流与后端（能力自适应）

| 流 | 来源 | 落盘 |
|---|---|---|
| 视频 + 逐帧时刻 | Camera2，最宽 FoV 的镜头，1080p，H.264 | `video.mp4` + `frames.csv` |
| IMU | 旋转矢量（融合姿态）+ 加速度 + 陀螺，最高稳定频率 | `imu.csv` |
| 深度/ToF | Camera2 DEPTH16，或 ARCore Depth | `tof.csv`（8×8 分区） |
| 位姿 | ARCore 6DoF（米制轨迹） | `poses.csv`（含逐行跟踪状态 TRACKING/LIMITED/STOPPED；桌面读取器已认作正式流，有轨迹长度与可信段统计） |
| 逐帧相机状态 | Camera2 捕获结果 | `frame_meta.csv`（曝光/ISO/对焦距离/AE/AF；桌面端有曝光报告：中位快门、慢于 1/60 的帧占比） |
| 附加 | 磁力计 / 气压计 / GPS(1Hz)，有则录 | `mag.csv` / `baro.csv` / `gps.csv` |

后端优先级（v0.2 起位姿优先）：Camera2+DEPTH16（实测深度最贵）→ ARCore
（只要设备支持就用：视频+位姿，深度有则加；ARCore 独占相机，位姿流只能
由它产生，v0.1 把它绑在「ARCore 深度可用」上，导致位姿可用但深度不可用
的机器——正如目标一加——根本录不到位姿，v0.2 已解耦）→ 纯 Camera2。
**核心功能不依赖 root**；root 不被使用（所有核心流都是公开 API）。

镜头选择（v0.2）：候选 = 每个后置逻辑相机 + 逻辑多摄里的每个**物理相机**
（一加 12 的超广角只以物理相机形式存在，v0.1 只枚举逻辑相机，选中了
72° 主摄）。取 FoV 最宽者；物理镜头通过会话输出配置定向，设备拒绝时
自动退回逻辑流并在 meta 里如实记录。录制相机的内参（焦距/主点，含厂家
标定数组若有）写入 meta.json 的 `calibration.camera`（v0.1 恒为 null）。

## 时间与坐标转换（格式的关键，都有测试锁定）

- **单一时钟**：Android 的 Camera2 `SENSOR_TIMESTAMP` 与 `SensorEvent.timestamp`
  同属 elapsedRealtime 纳秒域（开机单调钟）。录制开始时取一个 origin，
  所有流统一换算 `t_s = (ts_ns − origin_ns) / 1e9`。frames.csv 的时刻来自
  SENSOR_TIMESTAMP，**不是**帧到达时刻。写盘层再保证每条流时刻严格递增
  （撞钟则 +1µs 顶开，与 PC 录像端同一纪律）。
- **轴系**：格式的设备系（x 右、y 上、z 朝向操作者）与 Android 设备系定义相同，
  故加速度/陀螺逐分量直通；旋转矢量是 (x,y,z,w)，格式要标量在前的 (w,x,y,z) —
  纯分量重排，无符号翻转。两个系的世界系都是 ENU（东-北-天），姿态即
  「世界←设备」旋转，与格式语义一致；残余的世界系朝向差由导入器的
  Procrustes 配准吸收（与模拟/系留两端同一机制，已在 payoff 中验证）。
- **相机外参** `cam_from_device`：由相机 SENSOR_ORIENTATION θ 构造 —
  行向量 x_cam=(cosθ, sinθ, 0)，y_cam=(sinθ, −cosθ, 0)，z_cam=(0,0,−1)；
  θ=0 时恰好退化为格式参考安装 diag(1,−1,−1)（单元测试锁定）。
- **深度下采样**：DEPTH16 的低 13 位是毫米深度（高 3 位是置信度，掩掉）。
  针孔相机下等像素块 ≈ 等 tan 空间格，与格式的 8×8 tan 空间分区一致；
  每区取有效像素中位数，无有效像素记 −1。

## 构建

- 正常机器：`gradle :app:assembleDebug`（项目自带 Gradle 配置，AGP 8.5.2 /
  Kotlin 2.0.20 / compileSdk 34 / minSdk 26）。
- 本 VM：Gradle 守护进程的本地套接字握手在此沙箱里损坏（客户端读到乱码
  type tag，与配置无关），故用 `build_manual.sh` 以原始工具链
  （kotlinc + aapt2 + d8 + apksigner）产出同一 APK，并顺带跑完单元测试
  与 harness 证明。工具链在 `~/workspace/android-tools/`（JDK 17、
  SDK platform 34 / build-tools 34、Kotlin 2.0.20、ARCore 1.44 AAR）。
- 产物：`EchoCollector.apk`（debug 签名；versionName 0.2）。

## 无手机证明（已在本机跑通）

`build_manual.sh` 第 2–4 步（也可单独看 `validate_capture.py`）：

- `shared/` 的 JVM 单元测试 17 个全过：时刻换算与严格递增、旋转矢量重排、
  `camFromDevice(0)` = 参考安装、8×8 下采样（置信位掩码、中位数、−1 哨兵）、
  CSV/JSON 写出后可被解析。
- 桌面 harness 把模拟采集（`experiments/sensor_sim/capture_room2_sim`）当作
  合成 Android 事件流（旋转矢量、SENSOR_TIMESTAMP、DEPTH16 图）回放，
  **走应用自己的转换与写盘代码**生成采集文件夹，再由真正的 Python 读取器验证：
  - 读取器接受（format_version 1；320 帧 / 15950 IMU / 798 ToF）；
  - 无损：逐帧时刻 0 偏差，IMU 四元数 ≤1e-6（浮点格式化量级），ToF 逐格相同；
  - payoff 复现：ToF 尺度 **0.2500 m/unit**（参照 0.25，+0.00%），
    IMU 水平 **0.13°**（门槛 1.0°）。

## v0.10 改动（2026-11-04，朝向在录制时定死）

- **朝向一等字段**：录制开始时按设备显示旋转＋相机传感器朝向算定
  `video_rotation_deg = (sensor − display) mod 360`（本应用只用后置相机），
  写进 meta.json 的 `orientation` 块（capture_orientation 横/竖、
  video/display/sensor 三个旋转角、编码尺寸），并作为 orientation hint
  交给 Camera2 的 MediaRecorder 与 ARCore 的 MediaMuxer——video.mp4 本体
  与记录互相印证，桌面端不再猜朝向（桌面 sensor_record.parse_orientation
  校验该块、自相矛盾即报错，import 时透传进 job 记录）。
- 版本：versionCode 10 / versionName 0.10（build_manual.sh 的打包戳同步）。
- 构建状态：本机手动构建全程通过（JVM 17 测试、桌面 harness 合同验证、
  APK 打包＋apksigner 验签）；仍 **UNTESTED ON HARDWARE**——朝向 hint 在
  真机上的实际写入效果（及 ffprobe 读数）待第一次真机采集核对。

## v0.2 完善清单（本轮改了什么）

- **位姿转正**：ARCore 可用即走位姿后端（见上）；位姿逐行带跟踪状态；
  ARCore 后端只给真正进入编码器的帧打戳，停止时返回实际编码帧数供对账裁剪
  （v0.1 曾在编码输入队列满时凭空记帧）。桌面读取器解析 poses/frame_meta
  为正式可选流并做时刻单调校验；`Capture.trajectory_length_m()`（可只计
  TRACKING 段）与 `Capture.exposure_report()` 是两个现成的读数。
- **落盘抗杀**：写盘层改为流式（行到达即写、周期 flush）；meta.json 在
  录制开始就落盘（`finalized: false`），停止时改写为 `true` 并补齐统计。
  App 被杀/手机断电不再整段丢失，留下的文件夹如实标记未封口，桌面照常可读；
  App 再打开会提示发现未封口采集。
- **前台服务防杀**：录制由相机型前台服务 + 部分唤醒锁托管，通知栏可停；
  屏幕可以熄灭，手机可以揣兜里走。界面只是它的视图。
- **存储硬闸**：剩余 < 1.5 GB 拒绝开始；录制中跌破 300 MB 自动干净停止。
- **一键回传**：停止后（或从列表选任何一次采集）「打包发送」→ 整文件夹
  zip → 系统分享面板，自选通道发回电脑。自带一个只读 ContentProvider
  供分享授权（无 androidx 依赖）。
- **顺手修的真 bug**：mag.csv 的表头与 Python 读取器契约不一致
  （`mx_ut` vs `mx`）——带磁力计的真机采集会被读取器拒收，现已对齐。

## 第一次真机运行清单

1. 把 `EchoCollector.apk` 拷到手机，安装（需允许「安装未知应用」；debug 签名）。
2. 打开 App：看顶部能力行（机型、摄像头 FoV、深度后端是哪一种），
   并确认 `capabilities.json` 已生成（App 文件目录 `captures/` 的上一层）。
3. 点大按钮录 30 秒：慢走、环视房间一圈（最好走回起点闭环），再点一次停止。
   录制中状态行应出现「位姿 N」在涨——那是这次完善的主角。
4. 点「打包发送采集…」选刚录的这次，用顺手的通道发回电脑（或 USB 拷回
   `capture_*` 文件夹整个）。
5. 在电脑上跑：`.venv/bin/python android_collector/validate_capture.py <文件夹>`
   —— 输出 VALID 即格式过关；把结果发回来，据此决定下一步（深度后端是否命中、
   帧数对账是否吻合）。

## 无法在此验证的边界（诚实清单）

以下只有真机能回答，本机一律未验：

- 这台一加（PJA110 / Android 15）的 Camera2 时间戳源是否为 REALTIME、
  SENSOR_TIMESTAMP 与 SensorEvent 是否严格同域（绝大多数设备如此，但未验）；
- 是否存在 DEPTH16 深度相机、其分辨率/帧率；ARCore 是否已安装、是否支持 Depth API；
- MediaRecorder 实际编码帧数与 frames.csv 的对账在本机的帧丢弃行为
  （App 里已写了对账与裁剪逻辑，未在真机跑过）；
- 旋转矢量在 OxygenOS 上的实际采样率与长时间漂移；
- ARCore 后端的 MediaCodec 编码路径（代码完整，未运行过）；物理镜头定向
  （OutputConfiguration.setPhysicalCameraId）在这台一加上的接受度、被拒时
  的退回路径；前台服务在 OxygenOS 省电策略下的存活时长；
- 权限弹窗、发热、长时间录制的稳定性。

文件夹布局：`shared/`（纯 JVM：格式/转换/写盘，App 与 harness 共用）、
`app/`（Android 界面与采集）、`harness/`（桌面证明生成器）、
`validate_capture.py`（PC 端验证脚本）。
