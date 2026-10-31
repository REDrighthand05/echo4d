# echo_super — 超级集成层（永久工事）

建于 2026-10-30 旁聊。独立新目录，不改主线 `echo_scene/` 任何文件；
主线稳定后，把 `stage_*` 按下面的声明逐个注册进来即可合流。

## 结构

| 层 | 文件 | 作用 |
|---|---|---|
| 统一对象（PyObject层） | `core.py` | 万物一层皮：`kind + data + t + frame + source + meta`。任何来源的数据先进这层，就能被任意函数取用、组合、再喂回来。 |
| 内核注册表（Make/黑板层） | `registry.py` | 小函数只声明 `consumes → produces`。路由器按类型集合 BFS 找路：步数最少、同长取代价低的。同一产物有多种造法（如普通位姿 vs IMU 稳定）时自动选更优的那条到达方式。 |
| 方案即数据（Lisp层） | `plan.py` | 每次集成的方案是一份嵌套列表（S表达式），可存档、回放、改写、由程序生成。结果对象 `meta["sexp"]` 里带着它。 |
| 超级函数 | `super_fn.py` | `super_call(*任意个输入, target="scene")`：包装 → 找路 → 执行。已满足的步跳过；某内核失败时排除它重新找路（restart），有备用路线就绕过去。 |

## 用法

```python
from echo_super import make_super
from echo_super.demo import build_demo_registry   # 演示内核，真内核同签名

go = make_super(build_demo_registry())
go("room.mp4")                      # 视频 → 全链 → scene
go("room.mp4", "imu_log.csv")       # 多给一个 IMU → 自动换稳定轨迹路线
go(["a.jpg", "b.jpg"])              # 照片集 → 另一条入口
go(("points", pts))                 # 已有点云 → 只剩最后一步
go("room.mp4", execute=False)       # 只出方案，不执行
```

## 接真内核

现有主线阶段与声明的对应（合流时照此注册，一行一个）：

- `stage_extract` ≈ `video → frames`
- `stage_select` ≈ `frames → frames`（注意：同种进出在当前路由器里会被跳过，需先给产物换名，如 `frames → frames_selected`）
- `stage_poses` ≈ `frames_selected → poses`
- `stage_depth` ≈ `frames_selected → depth`
- `stage_fuse` ≈ `frames_selected + poses + depth → points`
- `stage_present` ≈ `points → scene`

## 已验证

`tests/test_echo_super.py` 7 项全过：全链路由、不定输入选更优路、
跳步、照片入口、方案 S表达式/JSON 往返、失败绕行。
