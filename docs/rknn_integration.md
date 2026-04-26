# RKNN模型加载推理实现说明

## 概述

本项目的RKNN检测器支持两种模式：
1. **Stub模式**：在非RK3566平台上使用，用于开发和测试
2. **RKNN模式**：在RK3566平台上使用真实RKNN Runtime SDK

## 文件结构

```
src/detection/
├── rknn_detector.h/cpp    # RKNN检测器主实现
├── rknn_adapter.h/cpp     # RKNN平台适配器
├── yolo_postprocess.h/cpp # YOLO后处理
└── detector.h             # 检测器接口
```

## RKNN API集成

### rknn_adapter.h/cpp

`RknnAdapter`类封装了RKNN Runtime SDK的主要API：

```cpp
class RknnAdapter {
public:
    bool init(const std::string& modelPath);          // rknn_init
    bool queryInputOutputInfo();                      // rknn_query
    bool setInputBuffer(const uint8_t* data, int size); // rknn_inputs_set
    bool run();                                       // rknn_run
    bool getOutputBuffer(float* data, int size);      // rknn_outputs_get
    void release();                                   // rknn_destroy
};
```

### 条件编译

通过`RK3566_PLATFORM`宏控制编译模式：

```cmake
# 在RK3566平台编译时启用
cmake .. -DRK3566_PLATFORM=ON
```

## 模型输入规格

| 参数 | 值 |
|------|-----|
| 输入尺寸 | 832 × 448 |
| 输入通道 | 3 (RGB) |
| 输入格式 | NHWC |
| 输入类型 | UINT8 |
| 归一化 | 像素值 / 255.0 |
| Letterbox | 灰色填充(114, 114, 114) |

## YOLO后处理支持

| 模型版本 | 输出形状 | 说明 |
|---------|---------|------|
| YOLOv3/v5 | [1, 25200, 85] | 包含80类+obj_conf |
| YOLOv8 | [1, 8400, 84] | 直接输出类别置信度 |

## RK3566平台集成步骤

### 1. 安装RKNN SDK

```bash
# 从瑞芯微官方获取RKNN Runtime SDK
# 安装到系统路径 /usr/lib 和 /usr/include
```

### 2. 编译

```bash
cd build
cmake .. -DRK3566_PLATFORM=ON
make -j4
```

### 3. 模型转换

使用RKNN-Toolkit将YOLO模型转换为.rknn格式：

```python
from rknn.api import RKNN

rknn = RKNN()
rknn.config(target_platform='rk3566')
rknn.load_yolo_model(model='yolov8.onnx')
rknn.build(do_quantization=True, dataset='dataset.txt')
rknn.export_rknn('yolov8.rknn')
```

### 4. 运行

```bash
./difference_detection config/config.yaml
```

## Stub模式说明

在非RK3566平台（如开发PC）上：

- 模型加载返回空输出
- 所有RKNN API调用为空实现
- 可进行其他模块的功能测试
- 编译时自动检测平台并启用stub模式

## 性能优化建议

1. **NPU核心绑定**：绑定推理线程到NPU专用核心
2. **零拷贝输入**：直接使用NPU内存映射
3. **批处理推理**：多帧合并处理
4. **异步推理**：推理和后处理并行

## 日志示例

```
[INFO] Initializing RKNN detector: model=/data/models/yolov8.rknn
[INFO] Model file found: /data/models/yolov8.rknn, size=12345678 bytes
[INFO] Model input: 832x448x3
[INFO] Model output: n_elems=8400
[INFO] RKNN inference completed successfully
[INFO] Detected 3 persons in 25ms
```

## 测试覆盖

| 测试内容 | 测试用例 |
|---------|---------|
| 平台检测 | PlatformCheck |
| 初始化 | InitStubMode |
| 输入尺寸 | InputSize |
| 输入缓冲 | SetInputBuffer |
| 推理执行 | RunInference |
| 输出获取 | GetOutput |