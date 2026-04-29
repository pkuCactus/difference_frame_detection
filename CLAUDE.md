# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目约定

- 使用 C++17
- 先写测试，再写实现
- 输出中文说明（注释、日志、用户提示均使用中文）
- 优先最小改动，小步提交，避免大规模重构
- 编码风格：C++ 函数、类、结构体采用大驼峰（PascalCase），变量名采用小驼峰（camelCase）；Python 函数和变量采用 snake_case，类采用大驼峰
- 编译后的二进制可拷贝到 `/tmp` 下执行，需要确保每一步执行没有问题

## 构建与测试

### 本地构建（x86，Stub 模式，无 RKNN 硬件）

```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### RK3566 交叉编译（Android NDK）

```bash
mkdir build && cd build
cmake -DRK3566_PLATFORM=ON ..
make -j$(nproc)
```

RK3566_PLATFORM 默认为 ON。CMake 会自动下载 NDK、构建 OpenCV/FFmpeg/CURL 等依赖。子模块（opencv、yaml-cpp、googletest、rknn-sdk）会在首次构建时自动初始化。

### 运行测试

```bash
cd build
make difference_detection_tests
./tests/difference_detection_tests
```

运行单个测试用例：

```bash
./tests/difference_detection_tests --gtest_filter=TestSuiteName.TestName
```

交叉编译时 `BUILD_TESTS` 自动关闭。x86 本地编译时，测试链接 `tests/rknn_api_stub.cpp` 中的 RKNN API stub，以在无硬件环境下编译通过。

### 部署产物

构建完成后，二进制、依赖库、配置文件、模型、运行脚本会自动复制到 `build/deploy/` 目录。在 RK3566 设备上可直接运行 `deploy/rk3566_run.sh`。

## 高层架构

### 管道与状态机

核心入口为 `main.cpp`，有三种运行模式：
1. **默认管道模式**：`./difference_detection config/config.yaml`
2. **RTSP 测试模式**：`--test-rtsp` / `--visualize-rtsp`
3. **单图检测调试**：`detect <image> [config]`，结果保存到 `outputs/`

`Pipeline` 是外层包装，实际工作线程由 `StateMachine` 驱动。状态流转如下：

```
INIT → CONNECTING → CHECK_CAPABILITY
  │
  ├─ 相机支持外部检测 ──→ CAMERA_DETECTION_MODE ──┐
  └─ 不支持或禁用 ─────→ LOCAL_DETECTION_MODE ────┤
                                                  ▼
                                    DIFFERENCE_ANALYSIS
                                         │
                    相似 ──→ 跳过事件，回到检测模式
                    不相似 ──→ EVENT_ANALYSIS → UPDATE_REF → 回到检测模式
```

状态机每帧在一个大循环中执行一次状态处理。`FrameDiffAnalyzer` 对比当前帧与参考帧的相似度，**仅当检测到目标（有框）且画面差异时才触发事件**。

### 检测架构

本地 RKNN 检测由 `Detector`（`detection/detector.h`）统一实现，直接调用 `rknn_api`，包含完整的预处理、后处理和 NMS：
- **预处理**：支持 letterbox 和直接 resize 两种模式，自动处理 BGR→RGB 转换
- **推理**：通过 `rknn_init` / `rknn_run` / `rknn_outputs_get` 执行，自动读取模型输入维度、类型和格式（NCHW/NHWC）
- **后处理**：内置 YOLOv3/YOLOv5/YOLOv8 解码逻辑，支持量化模型（INT8）的反量化，通过 ANCHORS 和网格解码检测框
- **NMS**：基于置信度和 IoU 的非极大值抑制，当前只检测 person（class 0）

`IDetector` 为抽象接口，`Detector` 是唯一实现。x86 本地测试通过 `tests/rknn_api_stub.cpp` 提供 RKNN API 的 stub 实现。

### 事件分析

`EventAnalyzer` 支持三种输出控制（通过配置文件 `event_analysis` 段）：
- `save_img`：是否保存事件图像到本地 `outputs/` 目录
- `with_box`：保存/推送的图像是否叠加检测框
- `webhook_enabled` + `webhook_url`：是否通过 HTTP POST 推送 base64 编码图像到 Vision API 服务端

`EventAnalyzer::AnalyzeImage` 和 `AnalyzeVideo` 分别处理图像模式和视频模式事件。视频模式会回放 `VideoFrameBuffer` 中缓存的历史帧。

### 参考帧差异检测

`FrameDiffAnalyzer` 支持三种相似度算法：`ssim`（默认）、`pixel_diff`、`phash`。配置项 `compare_roi_only` 为 `true` 时，仅对比检测框所在 ROI 区域，忽略背景变化。

参考帧更新策略：
- `newest`：每帧检测到目标时即更新参考帧
- `default`（即 `adaptive`）：仅在检测到差异、完成事件分析后才更新参考帧

### 目标跟踪

`ByteTracker` 实现 BYTETrack 多目标跟踪。 tracker 将检测框转换为 `Track` 对象，状态包括 `Tentative`（待确认）、`Tracked`（已确认）、`Lost`（丢失）、`Removed`（已移除）。只有 `Tracked` 状态的跟踪结果会传递给下游（差异检测和事件分析）。

## 开发与调试要点

- **本地 x86 开发**：无需 RKNN 硬件，stub 会返回 -1，检测器测试需要 mock 或 stub 数据
- **模型路径**：默认配置中 `model_path` 为相对路径 `models/yolov5s_rk3566.rknn`，运行时需确保工作目录下有该文件
- **Vision API 服务端**：`scripts/vision_api_server.py` 用于本地接收 webhook，启动命令 `python scripts/vision_api_server.py --port 8080 --output-dir received_events`
- **RTSP 模拟流**：`python scripts/rtsp_stream_server.py`，默认地址 `rtsp://localhost:8554/live`
- **日志**：`Logger` 为单例，宏 `LOG_INFO` / `LOG_DEBUG` / `LOG_WARN` / `LOG_ERROR` 自动记录文件和行号
- **性能统计**：`PerformanceStats` + `ScopedTimer` 用于统计各环节耗时，`PipelineStats` 汇总后每 10 秒输出一次
