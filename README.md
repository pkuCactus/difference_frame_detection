# Difference Frame Detection

基于差异帧检测的智能视频分析管道，通过帧间相似度判断减少不必要的 AI 推理计算，支持 RKNN NPU 加速与 RTSP 实时流处理。

## 功能特性

- **差异帧检测**：通过 SSIM 等相似度算法比较当前帧与参考帧，仅在检测到目标且画面变化显著时生成事件
- **RKNN 目标检测**：支持瑞芯微 RK3566 等平台的 RKNN 模型推理，支持 YOLOv8 等模型
- **RTSP 实时流处理**：支持 RTSP 视频流接入、重连、解码与可视化验证
- **BYTETrack 目标跟踪**：多目标跟踪，支持确认帧数、最大丢失帧数等参数配置
- **事件分析**：支持图像/视频两种事件模式，可配置视频事件时长
- **状态机管理**：核心管道采用状态机驱动（Idle → Running → Paused → Stopped）
- **ONNX 转 RKNN Web 工具**：提供基于 Web 的 ONNX 模型到 RKNN 模型转换服务，支持多平台芯片
- **Webhook 事件推送**：支持将事件图像通过 HTTP POST 推送到指定服务端
- **本地事件保存控制**：可配置是否保存事件图像到本地，以及是否叠加检测框
- **性能统计**：实时输出检测耗时、帧差耗时、处理帧数、事件数等统计信息
- **模拟流服务器**：内置 Python RTSP 模拟流服务器，便于本地测试
- **Vision API 服务端**：内置 Flask 服务端用于接收和存储 webhook 推送的图像

## 处理流程

1. **RTSP 收流解码** — 从网络摄像机获取实时视频流并解码
2. **目标检测** — 使用 RKNN/YOLO 模型检测画面中的目标（如行人）
3. **目标跟踪** — BYTETrack 对检测到的目标进行持续跟踪
4. **差异检测** — 对比当前帧与参考帧的相似度，判断画面是否发生显著变化
5. **事件分析** — **仅当检测到目标且存在画面差异时**，才生成事件（图像或视频片段）

## 系统架构

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  RTSP Client │────▶│ RKNN/YOLO   │────▶│ BYTETrack   │────▶│ Frame Diff  │
│  (收流解码)   │     │ (目标检测)   │     │ (目标跟踪)   │     │ (差异检测)   │
└─────────────┘     └──────┬──────┘     └─────────────┘     └──────┬──────┘
                           │                                         │
                           └─────────────────────────────────────────┘
                                               │
                                               ▼
                                      ┌─────────────────┐
                                      │  Event Analyzer │
                                      │ (有目标且差异)   │
                                      └─────────────────┘
```

## 目录结构

```
.
├── src/                    # C++ 源码
│   ├── common/             # 通用模块（配置、日志、性能统计、类型定义）
│   ├── core/               # 核心管道与状态机
│   ├── rtsp/               # RTSP 客户端与验证器
│   ├── decoder/            # 帧解码器
│   ├── camera/             # 相机能力检查与外部检测读取
│   ├── detection/          # RKNN 检测器、适配器与 YOLO 后处理
│   ├── tracking/           # BYTETrack 多目标跟踪
│   ├── analysis/           # 帧差异、相似度计算、事件分析
│   └── utils/              # 工具类（帧队列等）
├── tests/                  # 单元测试（GoogleTest）
├── config/                 # 配置文件示例
├── scripts/                # 辅助脚本
│   ├── rtsp_stream_server.py              # RTSP 模拟流服务器
│   ├── vision_api_server.py               # Vision API 服务端（接收 webhook）
│   ├── rk3566_run.sh                      # RK3566 设备运行脚本
│   └── onnx_to_rknn/                      # ONNX 转 RKNN Web 工具
│       ├── app.py                         # Flask Web 服务
│       ├── convert_worker.py              # 转换工作进程
│       ├── docker-compose.yml             # Docker 编排
│       ├── Dockerfile.toolkit             # rknn-toolkit 镜像
│       ├── Dockerfile.toolkit2            # rknn-toolkit2 镜像
│       └── templates/                     # Web 页面模板
├── docs/                   # 文档目录
├── CMakeLists.txt          # CMake 构建配置
└── README.md               # 本文件
```

## 依赖环境

- **CMake** >= 3.14
- **C++17** 兼容编译器
- **OpenCV**（视频解码与图像处理）
- **libcurl**（HTTP 请求）
- **yaml-cpp**（YAML 配置解析）
- **GoogleTest**（单元测试，可选）
- **RKNN SDK**（RK3566 平台部署时需要）

## 编译构建

### 标准构建（Stub 模式，无 RKNN 硬件）

适用于 x86 开发机，检测模块使用 stub 实现：

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### RK3566 平台构建（真实 RKNN）

在瑞芯微 RK3566 设备上开启 RKNN 支持（**本地编译，非交叉编译**）：

```bash
mkdir build && cd build
cmake -DRK3566_PLATFORM=ON ..
make -j$(nproc)
```

### Android 平台构建（交叉编译）

使用 Android NDK 交叉编译：

```bash
mkdir build && cd build
cmake -DANDROID_BUILD=ON ..
make -j$(nproc)
```

### 构建选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `BUILD_TESTS` | `ON` | 是否构建单元测试（交叉编译时自动禁用） |
| `RK3566_PLATFORM` | `OFF` | RK3566 设备本地编译（启用真实 RKNN） |
| `ANDROID_BUILD` | `OFF` | Android 交叉编译（使用 NDK） |

### 运行测试

```bash
cd build
make difference_detection_tests
./tests/difference_detection_tests
```

## 运行方式

### RK3566 设备部署指南

完整端到端部署流程：

#### 步骤 1：修改配置文件

编辑 `config/config.yaml`，根据实际环境调整：

```yaml
rtsp:
  url: "rtsp://YOUR_CAMERA_IP:554/stream1"  # 替换为实际 RTSP 地址

local_detection:
  model_path: "models/yolov5s_rk3566.rknn"   # RKNN 模型路径（相对于 deploy 目录）
  model_type: "yolov5"
  conf_threshold: 0.5
  detect_interval: 3

event_analysis:
  mode: "image"
  webhook_url: "http://YOUR_SERVER_IP:8080/api/vision"  # webhook 接收服务器
  webhook_enabled: true      # 是否启用 webhook 推送
  save_img: true             # 是否本地保存事件图像
  with_box: true             # 图像是否叠加检测框

logging:
  level: "INFO"
  file_path: "outputs/pipeline.log"
```

#### 步骤 2：编译

```bash
bash build.sh
```

编译产物位于 `build/deploy/` 目录，包含：
- `difference_detection` - 主程序
- `config/config.yaml` - 配置文件
- `models/*.rknn` - RKNN 模型
- `rk3566_run.sh` - 运行脚本
- `*.so` - 依赖库（RKNN、FFmpeg、OpenCV、c++_shared）

#### 步骤 3：拷贝到 RK3566 设备

将 `build/deploy/` 目录完整拷贝到 RK3566 设备：

```bash
# 方式一：scp 拷贝
adb push build/deploy /data/local/tmp/frame_difference

# 方式二：通过 USB/网络共享拷贝
```

#### 步骤 4：在 RK3566 上运行

```bash
cd /home/user/difference_detection
sh rk3566_run.sh config/config.yaml
```

或直接运行：

```bash
export LD_LIBRARY_PATH=.:$LD_LIBRARY_PATH
./difference_detection config/config.yaml
```

#### 步骤 5：验证功能

**验证 RTSP 拉流：**
- 查看日志输出 `pipeline.log`，确认连接成功
- 日志中应显示帧率、分辨率等信息

**验证目标检测：**
- 日志中应显示检测耗时和检测到的目标数量
- 如配置 `save_img: true`，事件图像保存到 `outputs/` 目录

**验证 Webhook 推送：**
- 启动 Vision API 服务端：
  ```bash
  python scripts/vision_api_server.py --port 8080 --output-dir received_events
  ```
- 检查服务端 `received_events/` 目录是否收到图像

#### 步骤 6：分析保存的图像

事件图像根据配置保存：
- **本地保存**：`outputs/event_*.jpg`（`save_img: true`）
- **Webhook 接收**：服务端 `received_events/` 目录

### 主程序

```bash
./difference_detection [选项] [配置文件]
```

### 命令行选项

| 选项 | 说明 |
|------|------|
| `--test-rtsp` | 单独验证 RTSP 收流 |
| `--visualize-rtsp` | 验证 RTSP 收流并可视化显示 |
| `--rtsp-url <url>` | 指定 RTSP 流地址（覆盖配置文件） |
| `--test-duration <sec>` | 测试持续时间，默认 10 秒，0 表示只收一帧 |
| `detect <image> [config]` | 对单张图片进行目标检测并输出结果 |
| `-h, --help` | 显示帮助信息 |

### 示例

```bash
# 使用默认配置文件运行管道
./difference_detection config/config.yaml

# 单独测试 RTSP 收流
./difference_detection --test-rtsp --rtsp-url rtsp://192.168.1.100:554/stream

# RTSP 收流可视化(3566上不可用)
./difference_detection --visualize-rtsp --rtsp-url rtsp://localhost:8554/live

# 对单张图片进行检测调试（输出带框结果到 outputs/ 目录）
./difference_detection detect bus.jpg config/config.yaml
```

## 配置文件

配置文件采用 YAML 格式，示例见 [config/config.yaml](config/config.yaml)：

```yaml
rtsp:
  url: "rtsp://192.168.1.100:554/stream"
  reconnect_interval_ms: 3000

camera_detection:
  enabled: false
  protocol: "REST"
  endpoint: "http://192.168.1.100/api/detection"
  poll_interval_ms: 100
  camera_id: "camera_001"
  timeout_ms: 3000

local_detection:
  model_path: "models/yolov5s_rk3566.rknn"
  model_type: "yolov5"
  conf_threshold: 0.5
  nms_threshold: 0.3
  detect_interval: 3
  timeout_ms: 500

tracker:
  enabled: true
  confirm_frames: 3
  max_lost_frames: 30
  high_threshold: 0.5
  low_threshold: 0.1
  match_threshold: 0.5

ref_frame:
  similarity_threshold: 0.85
  compare_method: "ssim"
  update_strategy: "newest"
  compare_roi_only: true

event_analysis:
  mode: "image"
  video_duration_sec: 5
  webhook_url: "http://localhost:8080/api/vision"
  webhook_enabled: true
  save_img: true
  with_box: true

logging:
  level: "INFO"
  file_path: "outputs/pipeline.log"
```

### 配置项说明

| 分组 | 关键项 | 说明 |
|------|--------|------|
| `rtsp` | `url` | RTSP 流地址 |
| `local_detection` | `model_path` | RKNN 模型文件路径 |
| `local_detection` | `detect_interval` | 每隔 N 帧进行一次检测 |
| `tracker` | `confirm_frames` | 目标连续出现 N 帧后确认跟踪 |
| `tracker` | `max_lost_frames` | 目标丢失 N 帧后移除跟踪 |
| `ref_frame` | `similarity_threshold` | 相似度阈值，低于此值视为画面变化 |
| `ref_frame` | `update_strategy` | 参考帧更新策略：`newest` 或 `adaptive` |
| `event_analysis` | `mode` | 事件模式：`image` 或 `video` |
| `event_analysis` | `webhook_url` | Webhook 推送地址 |
| `event_analysis` | `webhook_enabled` | 是否启用 webhook 推送 |
| `event_analysis` | `save_img` | 是否保存事件图像到本地 `outputs/` 目录 |
| `event_analysis` | `with_box` | 保存/推送的图像是否叠加检测框 |

## Vision API 服务端

位于 `scripts/vision_api_server.py`，用于接收来自 `difference_detection` 的 webhook 事件推送。

### 快速开始

```bash
pip install flask
python scripts/vision_api_server.py --port 8080 --output-dir received_events
```

### 接口说明

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/vision` | POST | 接收事件图像（JSON 格式：`{image_base64, timestamp}`） |
| `/api/health` | GET | 健康检查 |

### 运行方式

```bash
# 默认参数启动
python scripts/vision_api_server.py

# 指定监听地址和输出目录
python scripts/vision_api_server.py --host 0.0.0.0 --port 8080 --output-dir ./events
```

## ONNX 转 RKNN Web 工具

位于 `scripts/onnx_to_rknn/`，提供基于浏览器的 ONNX 模型到 RKNN 模型转换服务。

### 快速开始

```bash
cd scripts/onnx_to_rknn
pip install -r requirements.txt
python app.py
```

访问 http://localhost:5000 上传 ONNX 模型并选择目标芯片平台进行转换。

### 支持平台

- **rknn-toolkit**：RK1808、RV1109、RV1126、RK3399Pro
- **rknn-toolkit2**：RK3562、RK3566、RK3568、RK3576、RK3588

### Docker 方式

```bash
cd scripts/onnx_to_rknn
docker-compose up --build
```

## RTSP 模拟流服务器

用于本地测试 RTSP 收流功能：

```bash
python scripts/rtsp_stream_server.py
```

默认启动 RTSP 服务于 `rtsp://localhost:8554/live`。

## 编码风格

- C++ 函数、类、结构体采用**大驼峰**命名（`PascalCase`）
- C++ 变量采用**小驼峰**命名（`camelCase`）
- Python 函数和变量采用**蛇形**命名（`snake_case`）
- Python 类采用**大驼峰**命名

## 许可证

[MIT](LICENSE)
