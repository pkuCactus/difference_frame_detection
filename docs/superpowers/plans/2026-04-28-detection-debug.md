# 检测模型单图调试功能实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 main.cpp 中增加 `detect` 子命令，支持从配置文件读取参数、对单张图像进行 RKNN 检测推理、终端打印结果并将绘制了检测框的图像保存到 `outputs/` 目录。

**Architecture:** 将 `EventAnalyzer::drawBoxes` 提取为 `src/common/visualization` 公共工具，避免代码重复；在 `main.cpp` 中新增 `runDetectDebug` 函数处理 detect 子命令的完整流程；通过命令行解析识别 `detect <image> [config]` 格式。

**Tech Stack:** C++17, OpenCV, yaml-cpp, GoogleTest, CMake

---

## 文件结构

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/common/visualization.h` | 新建 | 声明 `DrawBoundingBoxes` |
| `src/common/visualization.cpp` | 新建 | 实现绘制检测框与置信度标签 |
| `tests/test_visualization.cpp` | 新建 | 可视化工具单元测试 |
| `src/analysis/event_analyzer.cpp` | 修改 | 删除 `drawBoxes`，改为调用 `DrawBoundingBoxes` |
| `src/analysis/event_analyzer.h` | 修改 | 删除 `drawBoxes` 私有方法声明 |
| `src/main.cpp` | 修改 | 扩展命令行解析，新增 `runDetectDebug` |
| `tests/test_detector.cpp` | 修改 | 补充单图推理+可视化集成测试 |
| `CMakeLists.txt` | 修改 | `SOURCES` 增加 `visualization.cpp` |
| `tests/CMakeLists.txt` | 修改 | 测试源文件增加 `visualization.cpp` |

---

### Task 1: 提取公共可视化工具 `DrawBoundingBoxes`

**Files:**
- Create: `src/common/visualization.h`
- Create: `src/common/visualization.cpp`
- Create: `tests/test_visualization.cpp`
- Modify: `src/analysis/event_analyzer.cpp` (删除 `drawBoxes` 方法体)
- Modify: `src/analysis/event_analyzer.h` (删除 `drawBoxes` 声明)
- Modify: `CMakeLists.txt`
- Modify: `tests/CMakeLists.txt`

- [ ] **Step 1: Write the failing test**

创建 `tests/test_visualization.cpp`：

```cpp
#include <gtest/gtest.h>
#include "common/visualization.h"
#include "common/types.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(VisualizationTest, DrawBoundingBoxes) {
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    std::vector<BoundingBox> boxes = {
        BoundingBox(10, 10, 20, 20, 0.95f, 0)
    };

    DrawBoundingBoxes(frame, boxes);

    // 验证矩形框上边缘像素为绿色 (BGR: 0, 255, 0)
    cv::Vec3b pixel = frame.at<cv::Vec3b>(10, 15);
    EXPECT_EQ(pixel[0], 0);
    EXPECT_EQ(pixel[1], 255);
    EXPECT_EQ(pixel[2], 0);
}

TEST(VisualizationTest, EmptyBoxes) {
    cv::Mat frame(100, 100, CV_8UC3, cv::Scalar(0, 0, 0));
    DrawBoundingBoxes(frame, {});

    // 验证图像仍为全黑
    cv::Mat gray;
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
    EXPECT_EQ(cv::countNonZero(gray), 0);
}
```

- [ ] **Step 2: Run test to verify it fails**

Run:
```bash
cd /data4/hjz/difference_detection/build && cmake --build . --target difference_detection_tests 2>&1 | tail -20
```

Expected: 编译失败，提示 `DrawBoundingBoxes` 未定义 或 `common/visualization.h` 不存在。

- [ ] **Step 3: Write minimal implementation**

创建 `src/common/visualization.h`：

```cpp
#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace diff_det {

void DrawBoundingBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes);

} // namespace diff_det
```

创建 `src/common/visualization.cpp`：

```cpp
#include "common/visualization.h"
#include <sstream>
#include <iomanip>

namespace diff_det {

void DrawBoundingBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes) {
    for (const auto& box : boxes) {
        cv::Rect rect(static_cast<int>(box.x1), static_cast<int>(box.y1),
                      static_cast<int>(box.x2 - box.x1), static_cast<int>(box.y2 - box.y1));

        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << box.conf;
        std::string text = "person: " + oss.str();

        cv::Point textPos(static_cast<int>(box.x1), static_cast<int>(box.y1) - 5);
        cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }
}

} // namespace diff_det
```

修改 `CMakeLists.txt`，在 `SOURCES` 列表中增加：
```
src/common/visualization.cpp
```

修改 `tests/CMakeLists.txt`，在 `TEST_SOURCES` 列表中增加：
```
${CMAKE_SOURCE_DIR}/src/common/visualization.cpp
test_visualization.cpp
```

- [ ] **Step 4: Run test to verify it passes**

Run:
```bash
cd /data4/hjz/difference_detection/build && cmake .. && cmake --build . --target difference_detection_tests -j$(nproc) && ./tests/difference_detection_tests --gtest_filter="VisualizationTest.*"
```

Expected: 两个测试都 PASS。

- [ ] **Step 5: Refactor EventAnalyzer to use new utility**

修改 `src/analysis/event_analyzer.h`：
- 删除私有方法 `void drawBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes);` 的声明

修改 `src/analysis/event_analyzer.cpp`：
- 在文件顶部增加 `#include "common/visualization.h"`
- 删除 `drawBoxes` 方法实现
- `AnalyzeImage` 中 `drawBoxes(annotatedFrame, boxes);` 改为 `DrawBoundingBoxes(annotatedFrame, boxes);`
- `AnalyzeVideo` 中两处 `drawBoxes` 调用均改为 `DrawBoundingBoxes`：
  - `drawBoxes(annotatedFrame, boxes);` → `DrawBoundingBoxes(annotatedFrame, boxes);`
  - `drawBoxes(annotated, boxes);` → `DrawBoundingBoxes(annotated, boxes);`

- [ ] **Step 6: Run existing tests to verify no regression**

Run:
```bash
cd /data4/hjz/difference_detection/build && cmake --build . --target difference_detection_tests -j$(nproc) && ./tests/difference_detection_tests --gtest_filter="EventAnalyzer*"
```

Expected: 所有 EventAnalyzer 相关测试 PASS。

- [ ] **Step 7: Commit**

```bash
git add src/common/visualization.h src/common/visualization.cpp tests/test_visualization.cpp src/analysis/event_analyzer.h src/analysis/event_analyzer.cpp CMakeLists.txt tests/CMakeLists.txt
git commit -m "$(cat <<'EOF'
feat: 提取公共可视化工具 DrawBoundingBoxes

- 将 EventAnalyzer::drawBoxes 提取为 src/common/visualization
- 新增 test_visualization.cpp 验证绘制逻辑
- EventAnalyzer 复用新的公共工具

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: 添加 detect 子命令到 main.cpp

**Files:**
- Modify: `src/main.cpp`

- [ ] **Step 1: Write the failing test (detector + visualization integration)**

修改 `tests/test_detector.cpp`，在末尾添加：

```cpp
TEST(RknnDetectorTest, DetectSingleImage) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    EXPECT_TRUE(detector.Init());

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes = detector.Detect(frame);

    // stub 模式下返回空框，验证接口可正常调用
    EXPECT_TRUE(boxes.empty());
}

TEST(RknnDetectorTest, DetectAndDrawBoxes) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    EXPECT_TRUE(detector.Init());

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes = detector.Detect(frame);

    // stub 模式下返回空框，验证绘制不崩溃
    cv::Mat result = frame.clone();
    DrawBoundingBoxes(result, boxes);
    EXPECT_EQ(result.size(), frame.size());
    EXPECT_EQ(result.type(), frame.type());
}
```

在 `tests/test_detector.cpp` 顶部添加：
```cpp
#include "common/visualization.h"
```

- [ ] **Step 2: Run integration test to verify it compiles and passes**

Run:
```bash
cd /data4/hjz/difference_detection/build && cmake --build . --target difference_detection_tests -j$(nproc) && ./tests/difference_detection_tests --gtest_filter="RknnDetectorTest.DetectSingleImage:RknnDetectorTest.DetectAndDrawBoxes"
```

Expected: 两个测试都 PASS。`DetectSingleImage` 验证 Detect() 接口可正常调用；`DetectAndDrawBoxes` 验证 Detect + 可视化集成不崩溃。

- [ ] **Step 3: Implement command-line parsing and runDetectDebug**

修改 `src/main.cpp`：

1. `CmdLineArgs` 结构体新增字段：
```cpp
struct CmdLineArgs {
    std::string configPath = "config/config.yaml";
    std::string rtspUrl;
    bool testRtsp = false;
    bool visualizeRtsp = false;
    int testDuration = 10;
    bool showHelp = false;
    std::string subCommand;      // 新增
    std::string imagePath;       // 新增
};
```

2. `parseArgs` 函数修改（保持现有小驼峰命名 `parseArgs`）：
```cpp
CmdLineArgs parseArgs(int argc, char* argv[]) {
    CmdLineArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test-rtsp") {
            args.testRtsp = true;
        } else if (arg == "--visualize-rtsp") {
            args.visualizeRtsp = true;
        } else if (arg == "--rtsp-url" && i + 1 < argc) {
            args.rtspUrl = argv[++i];
        } else if (arg == "--test-duration" && i + 1 < argc) {
            args.testDuration = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            args.showHelp = true;
        } else if (arg[0] != '-') {
            if (args.subCommand.empty() && arg == "detect") {
                args.subCommand = arg;
            } else if (args.subCommand == "detect" && args.imagePath.empty()) {
                args.imagePath = arg;
            } else if (args.subCommand == "detect" && !args.imagePath.empty()) {
                args.configPath = arg;
            } else {
                args.configPath = arg;
            }
        }
    }
    return args;
}
```

**注意：** 非选项参数若不是 `"detect"`，仍按原有行为被当作配置文件路径（`configPath`）。此行为与现有接口兼容。

3. `printUsage` 函数添加 detect 用法：
```cpp
void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options] [config.yaml]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --test-rtsp           单独验证RTSP收流" << std::endl;
    std::cout << "  --visualize-rtsp      验证RTSP收流并可视化显示" << std::endl;
    std::cout << "  --rtsp-url <url>      指定RTSP流地址(覆盖配置文件)" << std::endl;
    std::cout << "  --test-duration <sec> 测试持续时间,默认10秒,0表示只收一帧" << std::endl;
    std::cout << "  -h, --help            显示帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "Commands:" << std::endl;
    std::cout << "  detect <image> [config]  单图检测调试,结果保存到outputs/" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " config/config.yaml" << std::endl;
    std::cout << "  " << programName << " --test-rtsp --rtsp-url rtsp://localhost:8554/stream1" << std::endl;
    std::cout << "  " << programName << " --visualize-rtsp --rtsp-url rtsp://localhost:8554/stream1 --test-duration 30" << std::endl;
    std::cout << "  " << programName << " detect /tmp/test.jpg" << std::endl;
    std::cout << "  " << programName << " detect /tmp/test.jpg config/config.yaml" << std::endl;
}
```

4. 在 `src/main.cpp` 中添加必要的头文件：
```cpp
#include "detection/rknn_detector.h"
#include "common/visualization.h"
#include <filesystem>
#include <chrono>
#include <sstream>
```

5. 新增 `runDetectDebug` 函数（在 `runRtspTest` 之后、`main` 之前）：
```cpp
int runDetectDebug(const CmdLineArgs& args) {
    try {
        if (args.imagePath.empty()) {
            std::cerr << "错误: detect 子命令需要提供图像路径" << std::endl;
            std::cerr << "用法: detect <image_path> [config_path]" << std::endl;
            return 1;
        }

        Config config = Config::FromFile(args.configPath);

        std::cout << "========================================" << std::endl;
        std::cout << "Detection Model Debug" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Config file: " << args.configPath << std::endl;
        std::cout << "Image: " << args.imagePath << std::endl;
        std::cout << "Model: " << config.localDetection.modelPath << std::endl;
        std::cout << "Model type: " << config.localDetection.modelType << std::endl;
        std::cout << "Conf threshold: " << config.localDetection.confThreshold << std::endl;
        std::cout << "========================================" << std::endl;

        RknnDetector detector(config.localDetection);
        if (!detector.Init()) {
            std::cerr << "错误: 检测器初始化失败" << std::endl;
            return 1;
        }

        cv::Mat frame = cv::imread(args.imagePath);
        if (frame.empty()) {
            std::cerr << "错误: 无法加载图像: " << args.imagePath << std::endl;
            return 1;
        }

        std::cout << "Image loaded: " << frame.cols << "x" << frame.rows << std::endl;

        auto boxes = detector.Detect(frame);

        std::cout << "----------------------------------------" << std::endl;
        if (boxes.empty()) {
            std::cout << "检测结果: 未检测到目标" << std::endl;
        } else {
            std::cout << "检测结果: 检测到 " << boxes.size() << " 个目标" << std::endl;
            for (size_t i = 0; i < boxes.size(); ++i) {
                std::cout << "  [" << i + 1 << "] "
                          << "x1=" << static_cast<int>(boxes[i].x1) << ", "
                          << "y1=" << static_cast<int>(boxes[i].y1) << ", "
                          << "x2=" << static_cast<int>(boxes[i].x2) << ", "
                          << "y2=" << static_cast<int>(boxes[i].y2) << ", "
                          << "conf=" << std::fixed << std::setprecision(2) << boxes[i].conf
                          << std::endl;
            }
        }
        std::cout << "----------------------------------------" << std::endl;

        cv::Mat result = frame.clone();
        DrawBoundingBoxes(result, boxes);

        std::filesystem::path imagePath(args.imagePath);
        std::string baseName = imagePath.stem().string();

        auto now = std::chrono::system_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();

        std::ostringstream oss;
        oss << "outputs/detect_" << ms << "_" << baseName << ".jpg";
        std::string outputPath = oss.str();

        std::filesystem::create_directories("outputs");

        if (cv::imwrite(outputPath, result)) {
            std::cout << "结果已保存: " << outputPath << std::endl;
        } else {
            std::cerr << "错误: 保存结果图像失败: " << outputPath << std::endl;
            return 1;
        }

        std::cout << "========================================" << std::endl;

    } catch (const YAML::Exception& e) {
        std::cerr << "YAML config error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
```

6. 修改 `main` 函数，增加 `detect` 子命令路由：
```cpp
int main(int argc, char* argv[]) {
    CmdLineArgs args = parseArgs(argc, argv);

    if (args.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    if (args.subCommand == "detect") {
        return runDetectDebug(args);
    }

    if (args.testRtsp || args.visualizeRtsp) {
        return runRtspTest(args);
    }

    return runPipeline(args);
}
```

- [ ] **Step 4: Build the project**

Run:
```bash
cd /data4/hjz/difference_detection/build && cmake .. && cmake --build . --target difference_detection -j$(nproc)
```

Expected: 编译成功，无错误无警告。

- [ ] **Step 5: Run unit tests to verify no regression**

Run:
```bash
cd /data4/hjz/difference_detection/build && cmake --build . --target difference_detection_tests -j$(nproc) && ./tests/difference_detection_tests
```

Expected: 所有测试 PASS。

- [ ] **Step 6: Manual integration test**

创建测试图像：
```bash
mkdir -p /tmp/det_test && python3 -c "import numpy as np; import cv2; img = np.zeros((480, 640, 3), dtype=np.uint8) + 128; cv2.imwrite('/tmp/det_test/test.jpg', img)"
```

运行调试命令：
```bash
cd /data4/hjz/difference_detection/build && ./difference_detection detect /tmp/det_test/test.jpg
```

Expected:
- 终端输出 `"Detection Model Debug"` 信息头
- 输出图像尺寸 `"Image loaded: 640x480"`
- 输出检测结果（stub 模式下为空）
- 输出 `"结果已保存: outputs/detect_..._test.jpg"`
- 验证文件存在：`ls outputs/detect_*.jpg`

- [ ] **Step 7: Commit**

```bash
git add src/main.cpp tests/test_detector.cpp
git commit -m "$(cat <<'EOF'
feat: 增加 detect 子命令进行单图检测调试

- 支持 ./difference_detection detect <image> [config]
- 从配置文件读取 local_detection 参数初始化 RKNN 检测器
- 终端打印检测框坐标和置信度
- 将绘制检测框的结果保存到 outputs/ 目录

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## 验证清单

- [ ] `DrawBoundingBoxes` 单元测试通过（绘制像素验证 + 空框不崩溃）
- [ ] `EventAnalyzer` 现有测试无回归
- [ ] `RknnDetector` 集成测试通过（Detect + DrawBoundingBoxes 不崩溃）
- [ ] 项目整体编译无错误无警告
- [ ] `./difference_detection detect /tmp/test.jpg` 正确输出结果并保存图像
- [ ] `./difference_detection --help` 显示 detect 用法
- [ ] `./difference_detection`（无参数）仍能进入原有 pipeline 模式
- [ ] `./difference_detection detect /tmp/test.jpg config/config.yaml` 正确读取自定义配置并执行
- [ ] 未指定子命令时原有 pipeline 模式行为不变
