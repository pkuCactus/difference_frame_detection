#include <gtest/gtest.h>
#include "detection/rknn_detector.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(RknnDetectorTest, Init) {
    LocalDetectionConfig config;
    config.modelPath = "/tmp/test.rknn";
    
    RknnDetector detector(config);
    EXPECT_TRUE(detector.init());
}

TEST(RknnDetectorTest, SetConfThreshold) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    
    detector.setConfThreshold(0.7f);
}

TEST(RknnDetectorTest, Preprocess) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    detector.init();
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes = detector.detect(frame);
    
    EXPECT_TRUE(boxes.empty());
}

TEST(RknnDetectorTest, EmptyFrame) {
    LocalDetectionConfig config;
    RknnDetector detector(config);
    detector.init();
    
    cv::Mat emptyFrame;
    std::vector<BoundingBox> boxes = detector.detect(emptyFrame);
    
    EXPECT_TRUE(boxes.empty());
}