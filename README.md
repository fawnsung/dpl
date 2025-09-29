# DPL - Data Parallel LLM 推理与资源监控平台

一个面向多节点的本地/私有化 LLM 推理编排与监控平台：
- 前端 WebUI 展示集群资源（CPU/GPU/内存/网络/磁盘）并提供聊天与批处理数据集的交互界面
- 后端 Gateway 统一代理对上游 LLM（Ollama/OpenAI 兼容）的请求，负载感知并行调度，支持节点锁/解锁
- 节点 Monitor Agent 采集单机硬件指标并提供锁管理 API，供 Gateway 选路与抢占
- 附带健康检查脚本便于对 Tailscale 网络与各节点可用性巡检


## 功能速览

- 集群资源总览与单机详情：CPU 利用率、内存、网卡吞吐、磁盘吞吐、GPU 利用率/显存/温度/功耗（NVIDIA），支持 Apple Silicon 基础探测
- 多源 LLM 代理：优先识别 Ollama `/api/chat`，必要时自动回退 `/api/generate`；兼容 OpenAI 风格 `/v1/chat/completions`
- 负载感知调度：在线、未锁且模型匹配优先，按 GPU→CPU 低负载优先分配；节点锁在会话/任务结束时自动释放，并含兜底解锁
- 模型管理：列举/显示运行中模型，按需触发预热启动（对 Ollama 走 `/api/generate` 预热）
- 数据集批处理：上传 JSON 数组，后台并行切分到多节点处理，实时进度与统计，支持优雅停止与结果下载
- 系统告警：基于 CPU/内存/GPU 指标的阈值规则生成活动告警
- 健康检查脚本：一键巡检 Tailscale、节点监控 API 与 LLM API 可用性


## 目录结构

- `frontend/`：静态 Web 前端（`index.html`、`script.js`），通过 `gateway` 提供静态托管
- `gateway/`：核心 FastAPI 网关与负载均衡、SSE 转发、数据集任务编排、告警逻辑（`gateway.py`）
- `monitor_agent/`：单节点 FastAPI 监控代理（`agent.py`），采集硬件指标并提供锁/解锁 API
- `scripts/`：运维脚本（如 `health_check.py`）
- `alpaca_data.json`：样例/测试数据（体积较大）
- `requirements.txt`：Python 依赖
- `README.md`：本说明文档


## 环境要求

- Python 3.10+
- 建议：NVIDIA 驱动与 CUDA（如需采集 GPU 详细指标），或 Apple Silicon（基础探测）
- 建议：Tailscale（或其它内网穿透/专线）用于多节点互通
- 节点本地 LLM 服务：
  - 优先 Ollama（默认 `http://127.0.0.1:11434`），或其它 OpenAI 兼容端点


## 安装依赖

```bash
pip install -r requirements.txt
```

（Windows PowerShell 建议使用虚拟环境）


## 配置说明

所有网关侧节点拓扑在 `gateway/gateway.py` 的 `Settings.NODES` 中定义：

```python
class Settings:
    GATEWAY_PORT = 8000
    NODES = [
        {
            "id": 1, "name": "Node 1 (Host 1)",
            "monitor_base_url": "http://127.0.0.1:8001",
            "llm_url": "http://127.0.0.1:11434/api/chat",
        },
        # ... 更多节点 ...
    ]
```

- 将每个节点的监控代理地址（`monitor_base_url`）与 LLM 接口地址（`llm_url`）改为实际可达的地址（可通过 Tailscale/SSH Tunnel 映射）。
- 如使用 Ollama，`llm_url` 建议填 `/api/chat`；网关在 404/405/501 时将自动回退到 `/api/generate`。
- 网关监听端口默认为 `8000`，可改 `GATEWAY_PORT`。

节点侧（`monitor_agent/agent.py`）支持通过环境变量/`.env` 指定默认模型与 Ollama 主机：
- `OLLAMA_MODEL`：优先使用的模型名（否则会通过 `/api/tags` 自动选择）
-	`OLLAMA_HOST`：默认 `http://127.0.0.1:11434`
- `PORT`：监控代理监听端口（默认 `8001`）


## 启动与访问

1) 启动各节点监控代理（在每台节点机器）：
```bash
cd monitor_agent
python agent.py
```

2) 启动网关（在项目根目录）：
```bash
python -m uvicorn gateway.gateway:app --reload --host 0.0.0.0 --port 8000
```

3) 打开前端：
- 浏览器访问 `http://localhost:8000/`
- 默认托管 `frontend/` 静态页面


## 前端使用指南

- Hosts Overview：显示各节点在线/离线、CPU/内存/GPU 关键指标与运行模型
- Host Details：单节点多图表实时曲线（CPU、网卡 Mbps、磁盘 MB/s、GPU 温度/功耗/利用率）
- Chat：选择模型（或自动），与 LLM 进行流式对话；页面会先显示分配到的节点，再逐步输出内容
- Dataset Processing：
  - 上传 JSON 文件（数组，每条包含 `instruction` 与可选 `input` 字段）
  - 可下拉选择处理数据条数；创建任务后实时显示总体进度、节点处理统计、错误明细
  - 任务结束可下载带 `model_output` 的合并结果
- Alerts：展示由网关定期检测生成的活动告警


## 主要 API（Gateway）

- 节点状态与告警
  - `GET /api/status/all`：获取节点状态缓存（在线/锁定/指标）
  - `GET /api/alerts`：获取当前活动告警
  - `GET /api/cluster/overview`：集群级聚合指标（用于前端多主机折线图）

- 模型管理与会话
  - `GET /api/models`：列举可用/运行中模型（合并节点上报与 Ollama `/api/tags` 兜底）
  - `POST /api/models/start/{model}`：在可用节点预热启动指定模型
  - `POST /api/chat/completions`：统一聊天代理（自动选择最佳节点并转发，SSE 流式返回）
  - `POST /api/unlock/all`：紧急解锁所有节点（当异常锁死时）

- 数据集批处理（异步）
  - `POST /api/dataset/upload`：上传 JSON 文件并创建任务，立即返回 `job_id`
  - `GET /api/dataset/status/{job_id}`：任务简要进度
  - `GET /api/dataset/detailed-status/{job_id}`：详细统计（成功/失败/未处理、节点统计、时长等）
  - `POST /api/dataset/stop/{job_id}`：优雅停止任务（等待各节点处理当前条目后退出）
  - `GET /api/dataset/result/{job_id}`：下载已完成或已停止的任务结果
  - `POST /api/dataset/cleanup/all`：强制清理卡住任务并尝试解锁所有节点

- 诊断与 Prometheus 代理
  - `GET /api/debug/info`、`GET /api/debug/diagnosis`：运行时诊断信息
  - `GET /api/prometheus/query_range`：Prometheus 查询代理（`PROMETHEUS_URL` 可在 `gateway.py` 中调整）


## 节点监控代理（Monitor Agent）

- `GET /status`：返回本机锁状态、CPU/内存/GPU 指标、当前模型、网络与磁盘吞吐（基于采样快照计算）
- `POST /lock`、`POST /unlock`：节点锁管理（网关在任务/会话生命周期内使用）
- GPU 采集：
  - NVIDIA：通过 `pynvml` 采集利用率/显存/温度/功耗
  - macOS：尝试 `powermetrics` 与 PyTorch MPS 可用性做轻量检测


## 健康检查脚本

`scripts/health_check.py` 支持巡检：
- Tailscale 运行状态
- 各节点监控 API 与 LLM API 可用性

示例命令：
```bash
python scripts/health_check.py           # 单次巡检
python scripts/health_check.py -c -i 60  # 持续巡检，每 60 秒一次
```

注意：脚本中 `NODES` 的 IP/端口需按实际环境调整。


## 数据集格式示例

上传的 JSON 文件需为数组，每个元素示例：

```json
{
  "instruction": "请为下面标题写一段简介",
  "input": "数据并行推理平台设计"
}
```

结果下载将为与原始条目一一对应的数组，并在每条增加 `model_output` 字段（成功时为上游返回对象；失败时为 `{ "error": "..." }`）。


## 常见问题（FAQ）

- 前端显示所有主机离线？
  - 检查网关是否启动并监听 `8000`；确认 `Settings.NODES` 的 `monitor_base_url` 可达；确认各节点监控代理已启动。

- 指定模型却提示未找到？
  - 网关会尝试在可用节点通过 `/api/generate` 预热启动该模型；确保节点的 Ollama 已拉取对应模型镜像或有网络能力自动拉取。

- 节点锁住不释放？
  - 网关在流式结束或任务收尾会解锁，若 [DONE] 丢失会在 `finally` 兜底；仍异常时调用 `POST /api/unlock/all`。

- GPU 指标为空？
  - NVIDIA：确认驱动安装并可被 `pynvml` 访问；macOS 上仅提供有限探测。


## 安全与部署建议

- 生产环境务必收敛 CORS 白名单与前端托管域名
- 网关与监控代理建议放置在受控内网，通过 Tailscale/VPN 访问
- 对外暴露时考虑在网关前加反向代理与鉴权（如 OAuth/Token）


## Roadmap（可选）

- 更完善的调度策略（结合历史时延、并发度、自适应权重）
- 模型多副本并行与一致性归并
- 更丰富的告警规则与通知通道（Webhook/Email）
- 前端可视化与多集群视图增强


## License

本项目依赖的第三方组件按其各自许可证分发。项目自身许可证请根据需求补充。


# dpl
