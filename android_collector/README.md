# ECHO Collector — Android 采集端

**状态：UNTESTED ON HARDWARE（从未在真机上运行过）。**
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
| 位姿（额外流） | ARCore 6DoF | `poses.csv`（旁路文件，导入器忽略未知文件） |
| 附加 | 磁力计 / 气压计 / GPS(1Hz)，有则录 | `mag.csv` / `baro.csv` / `gps.csv` |

后端优先级：Camera2+DEPTH16 → ARCore（视频+位姿+深度）→ 纯 Camera2（无深度，
在 meta.json 里如实记录 `tof: none`）。**核心功能不依赖 root**；root 不被使用
（所有核心流都是公开 API）。

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
- 产物：`EchoCollector.apk`（debug 签名；versionName 0.1）。

## 无手机证明（已在本机跑通）

`build_manual.sh` 第 2–4 步（也可单独看 `validate_capture.py`）：

- `shared/` 的 JVM 单元测试 14 个全过：时刻换算与严格递增、旋转矢量重排、
  `camFromDevice(0)` = 参考安装、8×8 下采样（置信位掩码、中位数、−1 哨兵）、
  CSV/JSON 写出后可被解析。
- 桌面 harness 把模拟采集（`experiments/sensor_sim/capture_room2_sim`）当作
  合成 Android 事件流（旋转矢量、SENSOR_TIMESTAMP、DEPTH16 图）回放，
  **走应用自己的转换与写盘代码**生成采集文件夹，再由真正的 Python 读取器验证：
  - 读取器接受（format_version 1；320 帧 / 15950 IMU / 798 ToF）；
  - 无损：逐帧时刻 0 偏差，IMU 四元数 ≤1e-6（浮点格式化量级），ToF 逐格相同；
  - payoff 复现：ToF 尺度 **0.2500 m/unit**（参照 0.25，+0.00%），
    IMU 水平 **0.13°**（门槛 1.0°）。

## 第一次真机运行清单

1. 把 `EchoCollector.apk` 拷到手机，安装（需允许「安装未知应用」；debug 签名）。
2. 打开 App：看顶部能力行（机型、摄像头 FoV、深度后端是哪一种），
   并确认 `capabilities.json` 已生成（App 文件目录 `captures/` 的上一层）。
3. 点大按钮录 30 秒：慢走、环视房间一圈（最好走回起点闭环），再点一次停止。
4. 把生成的 `capture_*` 文件夹整个拷回电脑。
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
- ARCore 后端的 MediaCodec 编码路径（代码完整，未运行过）；
- 权限弹窗、发热、长时间录制的稳定性。

文件夹布局：`shared/`（纯 JVM：格式/转换/写盘，App 与 harness 共用）、
`app/`（Android 界面与采集）、`harness/`（桌面证明生成器）、
`validate_capture.py`（PC 端验证脚本）。
