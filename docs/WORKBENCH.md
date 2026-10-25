# Echo4D · Issues × PR 长期研究工作台

**更新：2026-10-25。** 这是一份跨会话工作台索引，不是经常自动刷新的仪表盘。开始新一轮时读 GitHub 实际最新状态；旧表项可能失效。

## 如何协同

```text
目标与边界 [#1]
   └── 研究总门禁 [#15]  ───────────────→ 独立精度验证 [#3]
         │                                 ↑
         ├── 输入与 RGB 位姿 [#2, #11]      │
         │     ├── #6 适配器               │
         │     └── #12 / #13 二选一审查 ────┤
         ├── 米制比例 [#14，已预注册三锚点] ┤
         └── 深度 B/C + Mesh [#15] ─────────┘

现有证据：[#7 公开 TUM 往返] → [#8 留出深度] → [#9 / #10 长尾误差]
设计文档：[PR #16]；协作协议：当前 PR。
```

### 记录职责

| 位置 | 写什么 | 不写什么 |
|---|---|---|
| Issue #1 | 长期愿景、不可冒充的边界、跨仓库关系 | 每次实验的完整日志 |
| Issue #15 | A/B/C 实验总门禁、优先级、可验证阶段进度、阻塞 | 与 #14 重复实现三锚点 |
| Issue #14 | 已锁定的三个 TUM 深度锚点、真正求尺度、7 个非训练视角 | 读取 GT Sim(3) scale 当求解答案 |
| Issue #11 | RGB-only 定位成果、第二真实序列的稳健性 | 宣称已完成密集米制 Mesh |
| Issue #3 | 几何独立验收 | 把同一设备 holdout 伪称独立激光真值 |
| PR | 可审查代码、契约、测试、机器原始证据、未解决点 | 用「已经支持」替代真实验证 |

## 每个研究循环的节奏

1. **选题（Issue）**：既有指标冻结的子任务优先，避免重复，标记 blocker 和明确的定义完成（DoD）。
2. **方案（Issue → Draft PR）**：记录输入集合和许可、时间戳、尺度来源、训练/验证隔离、资源上限、预计失败条件。
3. **实现（PR）**：小而可逆的改动，与已合并主干/依赖分支区分；测试要包含应当失败的案例。
4. **审查（CI + 真实实验）**：保留每个失败，分 E0–E4 说明证据强度，不能只汇总 pass。
5. **决定（Issue）**：可继续/需要修正/停止；选择基线 PR 时写出两边证据，不以“代码更多”选优。
6. **交接**：下一步只保留一个最高收益行动及其阻塞；不依赖旧聊天记录。

### 第一批优先级（快照，不等于已经执行）

- **P0**：对 [PR #12](https://github.com/REDrighthand05/echo4d/pull/12) 与 [PR #13](https://github.com/REDrighthand05/echo4d/pull/13) 做同数据合同与运行日志的技术比较；记录取舍，避免三个同名文件在两个分支互相覆盖。两者均依赖 #6。
- **P1**：在 [Issue #14](https://github.com/REDrighthand05/echo4d/issues/14) 交付真实三锚点米制尺度校准及未用深度/GT 的后验验证，严格保证 GT 不给求解器。
- **P2**：在 [Issue #15](https://github.com/REDrighthand05/echo4d/issues/15) 把估计位姿 + 真实深度完成 B 路 Mesh 闭环，之后再评测预测深度 C 路；对照同协议 A 组。
- **P3**：推进 [Issue #3](https://github.com/REDrighthand05/echo4d/issues/3) 的外部几何真值和多数据集验收，然后讨论动态 4D 或实时化。

本工作台不创建单独自动推送程序、不访问敏感数据，也不授权无人监督合并。定期检查由外部明确安排的任务负责；若未配置，需人工或下一次会话主动触发。

## 决策、复现与阻塞

任何重要技术结论至少写明：

```text
问题 / 容忍的失败：
候选 A / B，基准提交 SHA：
冻结输入与 sha256、许可、训练/验证划分：
运行环境 / 真实 CI URL / 命令：
比较指标（成功率 + 错误/长尾 + 速度 + 内存）：
首次失败与反例：
为何做此决定、未知条件：
后续回退/停止条件：
```

运行失败也算研究结果，不能静默删去。若是纯文档/静态检查，结论标 E1；不得写 E3/E4。

## 本次核对参考

- [项目档案 #1](https://github.com/REDrighthand05/echo4d/issues/1)
- [下一阶段实证 #15](https://github.com/REDrighthand05/echo4d/issues/15)
- [三锚点尺度 #14](https://github.com/REDrighthand05/echo4d/issues/14)
- [真实视频 SfM #11](https://github.com/REDrighthand05/echo4d/issues/11)
- [Draft PR #6](https://github.com/REDrighthand05/echo4d/pull/6)、[#8](https://github.com/REDrighthand05/echo4d/pull/8)、[#10](https://github.com/REDrighthand05/echo4d/pull/10)、[#12](https://github.com/REDrighthand05/echo4d/pull/12)、[#13](https://github.com/REDrighthand05/echo4d/pull/13)、[#16](https://github.com/REDrighthand05/echo4d/pull/16)
