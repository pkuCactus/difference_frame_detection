# AGENTS.md

## 开发约定（来自 CLAUDE.md）

- 使用 **C++17**
- **先写测试，再写实现**
- 输出**中文说明**，编码要有适当的注释
- 优先最小改动，小步提交，避免大规模重构
- 编码风格：函数、类、结构体采用**大驼峰**，变量采用**小驼峰**，不要用**snake**形式命名
- 编译后的可执行文件需拷贝到 `/tmp` 下执行，确保每一步没问题

## 构建

```bash
cd build
cmake ..                    # 默认：Stub模式（非RK3566平台）
cmake .. -DRK3566_PLATFORM=ON   # RK3566平台：启用真实RKNN SDK
make -j4
```

**依赖项：**
- OpenCV、CURL：系统包
- yaml-cpp：从 `/home/jianzhong/miniconda3/lib/cmake/yaml-cpp` 加载（CMake中硬编码路径）
- GTest：仅测试需要

## 运行

```bash
# 编译产物在 build/ 下，但权限问题导致无法直接执行
# 必须复制到 /tmp 下运行
cp build/difference_detection /tmp/
chmod +x /tmp/difference_detection
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/jianzhong/miniconda3/lib /tmp/difference_detection config/config.yaml
```

## 测试

```bash
# 编译测试（BUILD_TESTS 默认为 ON）
make -j4

# 复制到 /tmp 执行（同上，build/ 下无执行权限）
cp build/tests/difference_detection_tests /tmp/
chmod +x /tmp/difference_detection_tests
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/home/jianzhong/miniconda3/lib /tmp/difference_detection_tests
```

**现有测试：** 66个测试，18个测试套件，全部通过。

## 项目架构

**入口：** `src/main.cpp` → `Pipeline` → `StateMachine`（10状态循环）

**核心模块：**
- `src/core/` — 状态机、Pipeline主控
- `src/rtsp/` — RTSP拉流客户端（含断线重连）
- `src/camera/` — 相机能力探测 + 检测结果读取（REST/ONVIF stub）
- `src/detection/` — RKNN检测器（Stub/真实模式）、YOLO后处理
- `src/tracking/` — ByteTrack跟踪器（可选）
- `src/analysis/` — 帧差异分析（SSIM/pixel_diff/phash）、事件分析
- `src/utils/` — 帧队列、视频缓冲

**配置：** `config/config.yaml` — YAML格式，支持验证和序列化

## RKNN 平台切换

- **非RK3566平台：** 默认编译，RKNN API为空实现（Stub模式），检测器返回空结果
- **RK3566平台：** `cmake .. -DRK3566_PLATFORM=ON`，链接 `librknn_api.so`
- 模型输入：832×448，NCHW格式，像素值 `/255.0` 归一化，letterbox灰色填充(114)

## 日志系统

- 仅输出到文件（不输出到控制台）
- 格式：`[时间] [级别] [文件名:行号] 消息`
- 单文件存储，不做轮转
- 宏：`LOG_DEBUG/LOG_INFO/LOG_WARN/LOG_ERROR`

## 注意事项

- 所有整数类型使用 `<cstdint>`：`int32_t` 等
- 头文件位于 `src/` 各模块目录下（无独立 `include/` 目录）
- 相机检测结果匹配失败直接跳过，**不重试**
- Ref帧更新策略：`newest`（检测到有目标就更新，默认）或 `default`（仅进入事件分析后更新）
