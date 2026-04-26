#include <gtest/gtest.h>
#include "analysis/frame_diff.h"
#include "analysis/similarity.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(FrameDiffAnalyzerTest, Init) {
    RefFrameConfig config;
    FrameDiffAnalyzer analyzer(config);
    
    EXPECT_FALSE(analyzer.hasRef());
}

TEST(FrameDiffAnalyzerTest, UpdateRefWithNewestStrategy) {
    RefFrameConfig config;
    config.updateStrategy = "newest";
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    analyzer.setBoxesForRoi(boxes);
    analyzer.updateRef(frame);
    
    EXPECT_TRUE(analyzer.hasRef());
}

TEST(FrameDiffAnalyzerTest, UpdateRefWithDefaultStrategy) {
    RefFrameConfig config;
    config.updateStrategy = "default";
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    analyzer.updateRef(frame);
    
    EXPECT_TRUE(analyzer.hasRef());
}

TEST(FrameDiffAnalyzerTest, EmptyRef) {
    RefFrameConfig config;
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    EXPECT_FALSE(analyzer.isSimilar(frame, cv::Mat()));
}

TEST(FrameDiffAnalyzerTest, SimilarFrames) {
    RefFrameConfig config;
    config.similarityThreshold = 0.85f;
    config.compareMethod = "ssim";
    config.updateStrategy = "default";
    
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    analyzer.updateRef(frame1);
    
    EXPECT_TRUE(analyzer.isSimilar(frame2, analyzer.getRef()));
}

TEST(FrameDiffAnalyzerTest, DifferentFrames) {
    RefFrameConfig config;
    config.similarityThreshold = 0.85f;
    config.compareMethod = "ssim";
    config.updateStrategy = "default";
    
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    
    analyzer.updateRef(frame1);
    
    EXPECT_FALSE(analyzer.isSimilar(frame2, analyzer.getRef()));
}

TEST(FrameDiffAnalyzerTest, Reset) {
    RefFrameConfig config;
    config.updateStrategy = "default";
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    analyzer.updateRef(frame);
    
    analyzer.reset();
    
    EXPECT_FALSE(analyzer.hasRef());
}