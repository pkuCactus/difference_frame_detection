#include <gtest/gtest.h>
#include "detection/detector.h"
#include "common/config.h"
#include "common/visualization.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(DetectorTest, Init) {
    LocalDetectionConfig config;
    config.modelPath = "/tmp/test.rknn";

    Detector detector(config);
    EXPECT_TRUE(detector.Init());
}

TEST(DetectorTest, SetConfThreshold) {
    LocalDetectionConfig config;
    Detector detector(config);

    detector.SetConfThreshold(0.7f);
}

TEST(DetectorTest, SetNmsThreshold) {
    LocalDetectionConfig config;
    Detector detector(config);

    detector.SetNmsThreshold(0.5f);
}

TEST(DetectorTest, EmptyFrame) {
    LocalDetectionConfig config;
    Detector detector(config);
    detector.Init();

    cv::Mat emptyFrame;
    std::vector<BoundingBox> boxes = detector.Detect(emptyFrame);

    EXPECT_TRUE(boxes.empty());
}

TEST(DetectorTest, DetectSingleImage) {
    LocalDetectionConfig config;
    Detector detector(config);
    EXPECT_TRUE(detector.Init());

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes = detector.Detect(frame);

    // stub 模式下返回空框，验证接口可正常调用
    EXPECT_TRUE(boxes.empty());
}

TEST(DetectorTest, DetectAndDrawBoxes) {
    LocalDetectionConfig config;
    Detector detector(config);
    EXPECT_TRUE(detector.Init());

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes = detector.Detect(frame);

    // stub 模式下返回空框，验证绘制不崩溃
    cv::Mat result = frame.clone();
    DrawBoundingBoxes(result, boxes);
    EXPECT_EQ(result.size(), frame.size());
    EXPECT_EQ(result.type(), frame.type());
}
