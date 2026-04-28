# 检测模型单图调试功能设计文档

## 背景

在 RK3566 目标设备上调试本地 RKNN 检测模型时，需要一个快速验证手段：加载单张图像，运行检测推理，在终端查看检测框坐标与置信度，并将带框的结果图保存到 `outputs/` 目录。

## 目标

- `main.cpp` 支持 `detect` 子命令，接受图像路径
- 从配置文件读取 `local_detection` 参数初始化模型
- 对单张图像执行推理，终端打印结果
- 将绘制了检测框的结果图保存到 `outputs/` 目录

## 命令行接口

```bash
./difference_detection detect <image_path> [config_path]
```

- `detect`：子命令，触发单图检测调试模式
- `<image_path>`：待检测图像路径（必填）
- `[config_path]`：配置文件路径（可选，默认 `config/config.yaml`）

示例：
```bash
./difference_detection detect /tmp/test.jpg
./difference_detection detect /tmp/test.jpg config/config.yaml
```

## 执行流程

`runDetectDebug(const CmdLineArgs& args)` 函数执行以下步骤：

1. 读取配置 `Config::FromFile(args.configPath)`
2. 初始化检测器：`RknnDetector detector(config.localDetection)`，调用 `detector.Init()`
3. 加载图像：`cv::Mat frame = cv::imread(args.imagePath)`，失败则报错退出
4. 执行推理：`std::vector<BoundingBox> boxes = detector.Detect(frame)`
5. 终端打印检测结果：
   - 检测到目标时：输出 `检测到 N 个目标`，随后逐行打印每个框的坐标和置信度
   - 未检测到目标时：`未检测到目标`
6. 复制原图，调用 `DrawBoundingBoxes(result, boxes)` 绘制检测框与置信度标签
7. 生成输出文件名：`outputs/detect_<timestamp>_<basename>.jpg`
8. 保存结果图：`cv::imwrite(outputPath, result)`
9. 终端打印保存路径：`Result saved to: outputs/...`

## 代码结构调整

### 提取公共绘制工具

将 `EventAnalyzer::drawBoxes` 提取为公共函数，避免与新调试功能重复：

- **新建** `src/common/visualization.h`：声明 `void DrawBoundingBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes);`
- **新建** `src/common/visualization.cpp`：实现从 `EventAnalyzer::drawBoxes` 原样迁移
- **修改** `src/analysis/event_analyzer.cpp`：删除私有 `drawBoxes` 方法，改为调用 `DrawBoundingBoxes`
- **修改** `CMakeLists.txt` 与 `tests/CMakeLists.txt`：增加 `src/common/visualization.cpp`

## 测试策略

### test_visualization.cpp

- `TEST(VisualizationTest, DrawBoundingBoxes)`：创建空白图像，绘制一个已知坐标的框，验证框边缘像素为绿色（`cv::Scalar(0, 255, 0)`）
- `TEST(VisualizationTest, EmptyBoxes)`：传入空 boxes，验证图像像素未被修改

### test_detector.cpp（补充）

- `TEST(RknnDetectorTest, DetectSingleImage)`：创建测试图像，验证 `Detect()` 接口能正常调用（stub 模式下返回空框不报错）

## 错误处理

| 场景 | 行为 |
|------|------|
| 图像文件不存在或 `cv::imread` 失败 | 打印错误信息，返回非零退出码 |
| 模型初始化失败（`Init()` 返回 false） | 打印错误信息，返回非零退出码 |
| 检测结果为空 | 正常保存原图（不绘制框），终端打印提示 |
| `outputs` 目录不存在 | `std::filesystem::create_directories` 自动创建 |
| 保存结果图失败 | 捕获异常，打印错误信息 |

## 兼容性

- 保持现有所有命令行选项和用法不变
- `detect` 子命令与其他选项（如 `--help`）互不影响
- 未指定子命令时，原有 pipeline 模式行为不变
