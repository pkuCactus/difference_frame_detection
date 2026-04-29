#include <gtest/gtest.h>
#include "detection/rknn_detector.h"
#include "common/config.h"
#include "common/visualization.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(RknnDetectorTest, Init) {
    LocalDetectionConfig config;
    config.modelPath = "/tmp/test.rknn";
    
    RknnDetector detector(config);
    EXPECT_TRUE(detector.Init());
}

TEST(RknnDetectorTest, SetConfThreshold) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    
    detector.SetConfThreshold(0.7f);
}

TEST(RknnDetectorTest, Preprocess) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    detector.Init();
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes = detector.Detect(frame);
    
    EXPECT_TRUE(boxes.empty());
}

TEST(RknnDetectorTest, EmptyFrame) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    detector.Init();

    cv::Mat emptyFrame;
    std::vector<BoundingBox> boxes = detector.Detect(emptyFrame);

    EXPECT_TRUE(boxes.empty());
}

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

TEST(RknnDetectorTest, DecodeYolov5RknnSynthetic) {
    // stride 8: 80x80 grid, 255 channels
    std::vector<float> layer0(80 * 80 * 255, 0.0f);
    // stride 16: 40x40 grid
    std::vector<float> layer1(40 * 40 * 255, 0.0f);
    // stride 32: 20x20 grid
    std::vector<float> layer2(20 * 20 * 255, 0.0f);

    int gridW = 80, gridH = 80;
    int gx = 10, gy = 10, a = 0;
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
    std::vector<std::vector<int32_t>> dims = {
        {1, 255, 80, 80},
        {1, 255, 40, 40},
        {1, 255, 20, 20}
    };

    auto boxes = RknnDetector::DecodeYolov5Rknn(
        outputs, dims, 640, 640, 0.25f, 0.45f, 640, 640, 1.0f, 1.0f, 0, 0);

    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
    EXPECT_GT(boxes[0].conf, 0.25f);
    EXPECT_NEAR(boxes[0].x1, 84.8f - 5.0f, 1.0f);
    EXPECT_NEAR(boxes[0].y1, 84.8f - 5.0f, 1.0f);
    EXPECT_NEAR(boxes[0].x2, 84.8f + 5.0f, 1.0f);
    EXPECT_NEAR(boxes[0].y2, 84.8f + 5.0f, 1.0f);
}

TEST(RknnDetectorTest, DecodeYolov5RknnNonSquare) {
    std::vector<float> layer0(104 * 56 * 255, 0.0f);
    std::vector<float> layer1(52 * 28 * 255, 0.0f);
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
    std::vector<std::vector<int32_t>> dims = {
        {1, 255, 56, 104},
        {1, 255, 28, 52},
        {1, 255, 14, 26}
    };

    auto boxes = RknnDetector::DecodeYolov5Rknn(
        outputs, dims, 448, 832, 0.25f, 0.45f, 832, 448, 1.0f, 1.0f, 0, 0);

    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
}

TEST(RknnDetectorTest, DecodeYolov5RknnEmpty) {
    std::vector<std::vector<float>> outputs;
    std::vector<std::vector<int32_t>> dims;

    auto boxes = RknnDetector::DecodeYolov5Rknn(
        outputs, dims, 640, 640, 0.25f, 0.45f, 640, 640, 1.0f, 1.0f, 0, 0);

    EXPECT_EQ(boxes.size(), 0u);
}

TEST(RknnDetectorTest, DecodeYolov8Synthetic) {
    // 模拟 10 个 anchor，84 个值/anchor（4 bbox + 80 classes）
    std::vector<float> output(10 * 84, 0.0f);

    // 第 5 个 anchor 设置为 person（class 0），置信度 0.9
    int anchorIdx = 5;
    int base = anchorIdx * 84;
    output[base + 0] = 50.0f;  // cx
    output[base + 1] = 50.0f;  // cy
    output[base + 2] = 20.0f;  // w
    output[base + 3] = 30.0f;  // h
    output[base + 4] = 0.9f;   // class 0 conf

    std::vector<int32_t> dims = {1, 84, 10};

    auto boxes = RknnDetector::DecodeYolov8(
        output, dims, 0.25f, 0.45f, 640, 640, 1.0f, 1.0f, 0, 0);

    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
    EXPECT_GT(boxes[0].conf, 0.25f);
    EXPECT_NEAR(boxes[0].x1, 50.0f - 10.0f, 0.1f);
    EXPECT_NEAR(boxes[0].y1, 50.0f - 15.0f, 0.1f);
    EXPECT_NEAR(boxes[0].x2, 50.0f + 10.0f, 0.1f);
    EXPECT_NEAR(boxes[0].y2, 50.0f + 15.0f, 0.1f);
}

TEST(RknnDetectorTest, DecodeYolov8InferNumClasses) {
    // 构造 6 个值/anchor（4 bbox + 2 classes）
    std::vector<float> output(5 * 6, 0.0f);

    int base = 2 * 6;
    output[base + 0] = 10.0f;
    output[base + 1] = 10.0f;
    output[base + 2] = 4.0f;
    output[base + 3] = 4.0f;
    output[base + 4] = 0.8f;  // class 0
    output[base + 5] = 0.1f;  // class 1

    std::vector<int32_t> dims = {1, 6, 5};

    auto boxes = RknnDetector::DecodeYolov8(
        output, dims, 0.25f, 0.45f, 100, 100, 1.0f, 1.0f, 0, 0);

    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
}

TEST(RknnDetectorTest, DecodeYolov5SingleSynthetic) {
    // 3 个 anchor，85 个值/anchor（5 + 80 classes）
    std::vector<float> output(3 * 85, 0.0f);

    int base = 1 * 85;
    output[base + 0] = 30.0f;  // cx
    output[base + 1] = 40.0f;  // cy
    output[base + 2] = 10.0f;  // w
    output[base + 3] = 20.0f;  // h
    output[base + 4] = 0.95f;  // obj_conf
    output[base + 5] = 0.88f;  // class 0

    std::vector<int32_t> dims = {1, 85, 3};

    auto boxes = RknnDetector::DecodeYolov5Single(
        output, dims, 0.25f, 0.45f, 100, 100, 1.0f, 1.0f, 0, 0);

    EXPECT_GT(boxes.size(), 0u);
    EXPECT_EQ(boxes[0].label, 0);
    EXPECT_NEAR(boxes[0].x1, 30.0f - 5.0f, 0.1f);
}