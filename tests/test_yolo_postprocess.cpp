#include <gtest/gtest.h>
#include "detection/yolo_postprocess.h"

using namespace diff_det;

TEST(YoloPostprocessTest, ProcessRknnYolov5NoSigmoidDoubleApplication) {
    YoloPostprocess postprocess("yolov5", 0.25f, 0.45f);

    // 模拟一个 640x640 模型的三层输出，值已经是 sigmoid 后的结果
    // stride 8: 80x80 grid, 255 channels
    std::vector<float> layer0(80 * 80 * 255, 0.0f);
    // stride 16: 40x40 grid
    std::vector<float> layer1(40 * 40 * 255, 0.0f);
    // stride 32: 20x20 grid
    std::vector<float> layer2(20 * 20 * 255, 0.0f);

    // 在 layer0 (stride=8) 的 (10, 10) 位置，anchor 0 上放置一个人
    int gridW = 80, gridH = 80;
    int gx = 10, gy = 10, a = 0;
    int pixelIdx = gy * gridW + gx;
    int gridPixels = gridW * gridH;
    int baseC = a * 85;

    // x=0.6, y=0.6 -> cx=(0.6*2-0.5+10)*8=84.8, cy=84.8
    // w=0.5, h=0.5 -> bw=(0.5*2)^2*10=10, bh=10
    // objConf=0.9, class 0 (person) conf=0.85
    layer0[(baseC + 0) * gridPixels + pixelIdx] = 0.6f;  // x (already sigmoided)
    layer0[(baseC + 1) * gridPixels + pixelIdx] = 0.6f;  // y
    layer0[(baseC + 2) * gridPixels + pixelIdx] = 0.5f;  // w
    layer0[(baseC + 3) * gridPixels + pixelIdx] = 0.5f;  // h
    layer0[(baseC + 4) * gridPixels + pixelIdx] = 0.9f;  // obj_conf
    layer0[(baseC + 5) * gridPixels + pixelIdx] = 0.85f; // class 0 (person)

    std::vector<std::vector<float>> outputs = {layer0, layer1, layer2};

    auto boxes = postprocess.ProcessRknnYolov5(
        outputs, 640, 640, 1.0f, 1.0f, 0, 0, 640, 640);

    // 应该检测到至少一个人
    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
    EXPECT_GT(boxes[0].conf, 0.25f);

    // 验证 box 坐标大致正确（中心在 84.8, 84.8，宽高 10x10）
    EXPECT_NEAR(boxes[0].x1, 84.8f - 5.0f, 1.0f);
    EXPECT_NEAR(boxes[0].y1, 84.8f - 5.0f, 1.0f);
    EXPECT_NEAR(boxes[0].x2, 84.8f + 5.0f, 1.0f);
    EXPECT_NEAR(boxes[0].y2, 84.8f + 5.0f, 1.0f);
}

TEST(YoloPostprocessTest, ProcessRknnYolov5NonSquareInput) {
    YoloPostprocess postprocess("yolov5", 0.25f, 0.45f);

    // 模拟 832x448 模型的三层输出（非正方形）
    // stride 8: 104x56
    std::vector<float> layer0(104 * 56 * 255, 0.0f);
    // stride 16: 52x28
    std::vector<float> layer1(52 * 28 * 255, 0.0f);
    // stride 32: 26x14
    std::vector<float> layer2(26 * 14 * 255, 0.0f);

    int gridW = 104, gridH = 56;
    int gx = 20, gy = 10, a = 0;
    int pixelIdx = gy * gridW + gx;
    int gridPixels = gridW * gridH;
    int baseC = a * 85;

    layer0[(baseC + 0) * gridPixels + pixelIdx] = 0.6f;
    layer0[(baseC + 1) * gridPixels + pixelIdx] = 0.6f;
    layer0[(baseC + 2) * gridPixels + pixelIdx] = 0.5f;
    layer0[(baseC + 3) * gridPixels + pixelIdx] = 0.5f;
    layer0[(baseC + 4) * gridPixels + pixelIdx] = 0.9f;
    layer0[(baseC + 5) * gridPixels + pixelIdx] = 0.85f;

    std::vector<std::vector<float>> outputs = {layer0, layer1, layer2};

    auto boxes = postprocess.ProcessRknnYolov5(
        outputs, 832, 448, 1.0f, 1.0f, 0, 0, 832, 448);

    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
}

TEST(YoloPostprocessTest, ProcessRknnYolov5EmptyOutputs) {
    YoloPostprocess postprocess("yolov5", 0.25f, 0.45f);
    std::vector<std::vector<float>> outputs;

    auto boxes = postprocess.ProcessRknnYolov5(
        outputs, 640, 640, 1.0f, 1.0f, 0, 0, 640, 640);

    EXPECT_EQ(boxes.size(), 0u);
}
