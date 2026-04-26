#include <gtest/gtest.h>
#include "analysis/frame_diff.h"
#include "analysis/similarity.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(FrameDiffAnalyzerTest, Init) {
    RefFrameConfig config;
    FrameDiffAnalyzer analyzer(config);
    
    EXPECT_FALSE(analyzer.HasRef());
}

TEST(FrameDiffAnalyzerTest, UpdateRefWithNewestStrategy) {
    RefFrameConfig config;
    config.updateStrategy = "newest";
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    analyzer.SetBoxesForRoi(boxes);
    analyzer.UpdateRef(frame);
    
    EXPECT_TRUE(analyzer.HasRef());
}

TEST(FrameDiffAnalyzerTest, UpdateRefWithDefaultStrategy) {
    RefFrameConfig config;
    config.updateStrategy = "default";
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    analyzer.UpdateRef(frame);
    
    EXPECT_TRUE(analyzer.HasRef());
}

TEST(FrameDiffAnalyzerTest, EmptyRef) {
    RefFrameConfig config;
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    EXPECT_FALSE(analyzer.IsSimilar(frame, cv::Mat()));
}

TEST(FrameDiffAnalyzerTest, SimilarFrames) {
    RefFrameConfig config;
    config.similarityThreshold = 0.85f;
    config.compareMethod = "ssim";
    config.updateStrategy = "default";
    
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    analyzer.UpdateRef(frame1);
    
    EXPECT_TRUE(analyzer.IsSimilar(frame2, analyzer.GetRef()));
}

TEST(FrameDiffAnalyzerTest, DifferentFrames) {
    RefFrameConfig config;
    config.similarityThreshold = 0.85f;
    config.compareMethod = "ssim";
    config.updateStrategy = "default";
    
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
    
    analyzer.UpdateRef(frame1);
    
    EXPECT_FALSE(analyzer.IsSimilar(frame2, analyzer.GetRef()));
}

TEST(FrameDiffAnalyzerTest, Reset) {
    RefFrameConfig config;
    config.updateStrategy = "default";
    FrameDiffAnalyzer analyzer(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    analyzer.UpdateRef(frame);
    
    analyzer.reset();
    
    EXPECT_FALSE(analyzer.HasRef());
}