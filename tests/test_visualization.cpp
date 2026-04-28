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
