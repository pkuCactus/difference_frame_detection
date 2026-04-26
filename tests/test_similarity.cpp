#include <gtest/gtest.h>
#include "analysis/similarity.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(SsimCalculatorTest, IdenticalFrames) {
    SsimCalculator calculator;
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    float similarity = calculator.Calculate(frame, frame);
    EXPECT_NEAR(similarity, 1.0f, 0.01f);
}

TEST(SsimCalculatorTest, DifferentFrames) {
    SsimCalculator calculator;
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    
    float similarity = calculator.Calculate(frame1, frame2);
    EXPECT_LT(similarity, 0.9f);
}

TEST(SsimCalculatorTest, EmptyFrames) {
    SsimCalculator calculator;
    
    cv::Mat emptyFrame;
    cv::Mat frame(480, 640, CV_8UC3);
    
    float similarity = calculator.Calculate(emptyFrame, frame);
    EXPECT_FLOAT_EQ(similarity, 0.0f);
}

TEST(SsimCalculatorTest, DifferentSizes) {
    SsimCalculator calculator;
    
    cv::Mat frame1(480, 640, CV_8UC3);
    cv::Mat frame2(320, 240, CV_8UC3);
    
    float similarity = calculator.Calculate(frame1, frame2);
    EXPECT_FLOAT_EQ(similarity, 0.0f);
}

TEST(PixelDiffCalculatorTest, IdenticalFrames) {
    PixelDiffCalculator calculator;
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    float similarity = calculator.Calculate(frame, frame);
    EXPECT_NEAR(similarity, 1.0f, 0.01f);
}

TEST(PixelDiffCalculatorTest, DifferentFrames) {
    PixelDiffCalculator calculator;
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(64, 64, 64));
    
    float similarity = calculator.Calculate(frame1, frame2);
    EXPECT_LT(similarity, 1.0f);
}

TEST(PixelDiffCalculatorTest, EmptyFrames) {
    PixelDiffCalculator calculator;
    
    cv::Mat emptyFrame;
    cv::Mat frame(480, 640, CV_8UC3);
    
    float similarity = calculator.Calculate(emptyFrame, frame);
    EXPECT_FLOAT_EQ(similarity, 0.0f);
}

TEST(PixelDiffCalculatorTest, Name) {
    SsimCalculator ssim;
    PixelDiffCalculator pixelDiff;
    
    EXPECT_EQ(ssim.Name(), "ssim");
    EXPECT_EQ(pixelDiff.Name(), "pixel_diff");
}