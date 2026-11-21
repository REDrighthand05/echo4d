# ECHO 场景重构与视效客户端 - 热更新源配置与交接运维指南

本指南面向接手 ECHO 客户端维护与发布的组织/开发者，详细说明客户端热更新源的寻址机制、覆盖配置方式、中继代理链条、以及如何构建和发布热更新包。

---

## 1. 架构与热更新源解耦机制

客户端热更新逻辑位于 `echo_app/update_source.py` 与 `echo_app/updater.py`，采用**三级配置覆盖机制**，彻底与单一仓库硬编码解耦：

```
优先级 1 (最高): 环境变量 (ECHO_UPDATE_REPO, ECHO_UPDATE_RELAY_BASE 等)
        ↓
优先级 2: update_source.json 配置文件 (支持本地安装根目录、工作区、APPDATA 用户目录)
        ↓
优先级 3 (默认): 默认官方源 (REDrighthand05/echo4d 及 REDrighthand05/echo4d-update)
```

### 1.1 核心寻址路径与三级故障转移

客户端检查更新时，按以下优先级依次尝试：

1. **jsDelivr CDN 中继节点**：
   - 地址模板：`https://cdn.jsdelivr.net/gh/{owner}/{repo}-update@main/`
   - 特点：国内免翻墙高速直连，延迟极低。
2. **GitHub Pages 镜像节点**：
   - 地址模板：`https://{owner}.github.io/{repo}-update/`
   - 特点：静态资源中继，适合静态更新包直链分发。
3. **GitHub Releases API 节点**：
   - 地址模板：`https://api.github.com/repos/{owner}/{repo}/releases`
   - 特点：官方动态 API，客户端自适应本地代理（探测 7897, 7890, 10809, 10808 等代理端口并支持用户自定义代理）。

---

## 2. 新组织接手维护：两步完成更新源接管

新组织接管 ECHO 软件分发与更新时，**无需重新编译客户端可执行文件**，任选以下一种方式即可接管：

### 方式 A：修改 `update_source.json`

在客户端安装根目录（与 `EchoScene.exe` 同级）、`app/` 目录或 `%APPDATA%\ECHO\` 下放置 `update_source.json`：

```json
{
  "$comment": "ECHO 热更新源配置文件。新组织接手维护时修改此文件即可完成重定向。",
  "repo": "YourOrganization/echo-scene",
  "relay_repo": "YourOrganization/echo-scene-update",
  "relay_bases": [
    "https://cdn.jsdelivr.net/gh/YourOrganization/echo-scene-update@main",
    "https://yourorganization.github.io/echo-scene-update"
  ],
  "latest_url": null,
  "releases_url": null
}
```

- `repo`: GitHub 主仓库名 (`Owner/Repo`)，用于发布 Release 和获取更新元数据。
- `relay_repo`: 中继仓库名，存放静态更新文件（可选）。
- `relay_bases`: 静态资源中继基准 URL 列表。若部署私有 CDN 或内网更新服务器，直接在此填入私有 HTTP/HTTPS 基础 URL。
- `latest_url` / `releases_url`: 自定义 API 端点（支持私有自建服务协议兼容端点）。

### 方式 B：通过环境变量注入（容器/自动化部署）

在运行环境或启动脚本中注入：
```powershell
$env:ECHO_UPDATE_REPO = "YourOrganization/echo-scene"
$env:ECHO_UPDATE_RELAY_BASE = "https://update.yourdomain.com/echo"
```

---

## 3. 热更新包格式与构建流程

ECHO 热更新遵循轻量级“原子热替换”原则：基础安装包（~600MB，包含 Python 解释器、PyQt/WebEngine 运行时、CUDA/Torch 动态库）作为固化宿主壳；所有业务逻辑、算法管线与前端界面位于 `app/` 目录下，更新包通常仅为数兆字节。

### 3.1 资产规范

每次 Release 发行版需挂载两个关键文件：
1. **`echoscene-manifest.json`**：目标版本的完整文件哈希清单（SHA256、相对路径、文件大小）。
2. **`EchoScene-Update.zip`**：增量差分包，包含：
   - `plan.json`：从起始版本升级到目标版本的迁移规划（变更文件、删除文件、文件大小）。
   - `target-manifest.json`：目标清单副本。
   - 需要增量替换的实际文件内容（保留目录结构）。

### 3.2 自动化构建步骤

维护组织可通过内置更新包生成器构建差分包：

```python
import json
from pathlib import Path
from echo_super.ds_batch5 import plan_update, order_apply_steps

# 1. 扫描生成当前版本的 manifest (app/ 目录下所有文件)
# 2. 对比上个版本的 manifest
# 3. 输出 plan.json 并打包变动文件至 EchoScene-Update.zip
```

构建后上传到 GitHub Release 标签（例如 `v0.7.0`）即可。

---

## 4. 客户端应用更新流程与安全防护机制

1. **检查与防竞态**：
   - 客户端在任务运行期间（如 3D 重建中）自动拦截并拒绝热更新，防止内存与磁盘文件不一致。
2. **下载与三重完整性校验**：
   - 客户端将 `EchoScene-Update.zip` 下载至 `.echo/staging/` 临时暂存区。
   - 严格校验每一个文件的 SHA256 与 `target-manifest.json` 记录，任何哈希不匹配立即清空 staging，绝不产生部分覆盖。
3. **原子应用与重启**：
   - 校验通过后写入 `.echo/pending.json`。
   - 客户端主进程退出，由外壳引导器（`--echo-apply`）执行原子文件替换，更新完成后自动重启客户端。
