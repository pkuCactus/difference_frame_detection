#include <gtest/gtest.h>
#include "analysis/event_analyzer.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(EventAnalyzerTest, Constructor) {
    EventAnalysisConfig config;
    config.mode = "image";
    config.videoDurationSec = 5;

    EventAnalyzer analyzer(config);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeImageEmptyFrame) {
    EventAnalysisConfig config;
    EventAnalyzer analyzer(config);

    bool callbackCalled = false;
    analyzer.setEventCallback([&callbackCalled](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
        callbackCalled = true;
    });

    cv::Mat emptyFrame;
    std::vector<BoundingBox> boxes;
    boxes.emplace_back(100.0f, 100.0f, 200.0f, 200.0f, 0.9f, 0);

    analyzer.AnalyzeImage(emptyFrame, boxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeImageEmptyBoxes) {
    EventAnalysisConfig config;
    EventAnalyzer analyzer(config);

    bool callbackCalled = false;
    analyzer.setEventCallback([&callbackCalled](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
        callbackCalled = true;
    });

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> emptyBoxes;

    analyzer.AnalyzeImage(frame, emptyBoxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeImageValidInput) {
    EventAnalysisConfig config;
    EventAnalyzer analyzer(config);

    bool callbackCalled = false;
    int receivedFrameId = 0;
    analyzer.setEventCallback([&callbackCalled, &receivedFrameId](const cv::Mat&, const std::vector<BoundingBox>&, int frameId, int64_t) {
        callbackCalled = true;
        receivedFrameId = frameId;
    });

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    std::vector<BoundingBox> boxes;
    boxes.emplace_back(100.0f, 100.0f, 200.0f, 200.0f, 0.9f, 0);

    analyzer.AnalyzeImage(frame, boxes);

    EXPECT_TRUE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 1);
    EXPECT_EQ(receivedFrameId, 1);
}

TEST(EventAnalyzerTest, AnalyzeVideoEmptyFrames) {
    EventAnalysisConfig config;
    config.mode = "video";
    EventAnalyzer analyzer(config);

    bool callbackCalled = false;
    analyzer.setEventCallback([&callbackCalled](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
        callbackCalled = true;
    });

    std::vector<cv::Mat> emptyFrames;
    std::vector<BoundingBox> boxes;
    boxes.emplace_back(100.0f, 100.0f, 200.0f, 200.0f, 0.9f, 0);

    analyzer.AnalyzeVideo(emptyFrames, boxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeVideoEmptyBoxes) {
    EventAnalysisConfig config;
    config.mode = "video";
    EventAnalyzer analyzer(config);

    bool callbackCalled = false;
    analyzer.setEventCallback([&callbackCalled](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
        callbackCalled = true;
    });

    std::vector<cv::Mat> frames;
    frames.push_back(cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128)));
    std::vector<BoundingBox> emptyBoxes;

    analyzer.AnalyzeVideo(frames, emptyBoxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeVideoValidInput) {
    EventAnalysisConfig config;
    config.mode = "video";
    EventAnalyzer analyzer(config);

    int callbackCount = 0;
    analyzer.setEventCallback([&callbackCount](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
        callbackCount++;
    });

    std::vector<cv::Mat> frames;
    frames.push_back(cv::Mat(480, 640, CV_8UC3, cv::Scalar(128, 128, 128)));
    frames.push_back(cv::Mat(480, 640, CV_8UC3, cv::Scalar(64, 64, 64)));

    std::vector<BoundingBox> boxes;
    boxes.emplace_back(100.0f, 100.0f, 200.0f, 200.0f, 0.9f, 0);

    analyzer.AnalyzeVideo(frames, boxes);

    EXPECT_EQ(callbackCount, 2);
    EXPECT_EQ(analyzer.GetEventCount(), 1);
}

TEST(VideoBufferTest, Constructor) {
    VideoBuffer buffer(100);
    EXPECT_EQ(buffer.Size(), 0);
}

TEST(VideoBufferTest, AddFrame) {
    VideoBuffer buffer(10);

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    buffer.addFrame(frame, 1, 1000);

    EXPECT_EQ(buffer.Size(), 1);
}

TEST(VideoBufferTest, AddEmptyFrame) {
    VideoBuffer buffer(10);

    cv::Mat emptyFrame;
    buffer.addFrame(emptyFrame, 1, 1000);

    EXPECT_EQ(buffer.Size(), 0);
}

TEST(VideoBufferTest, MaxSizeLimit) {
    VideoBuffer buffer(3);

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    buffer.addFrame(frame, 1, 1000);
    buffer.addFrame(frame, 2, 2000);
    buffer.addFrame(frame, 3, 3000);
    buffer.addFrame(frame, 4, 4000);

    EXPECT_EQ(buffer.Size(), 3);
}

TEST(VideoBufferTest, GetFrames) {
    VideoBuffer buffer(10);

    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(64, 64, 64));
    cv::Mat frame3(480, 640, CV_8UC3, cv::Scalar(32, 32, 32));

    buffer.addFrame(frame1, 1, 1000);
    buffer.addFrame(frame2, 2, 2000);
    buffer.addFrame(frame3, 3, 3000);

    auto frames = buffer.getFrames(2);
    EXPECT_EQ(frames.size(), 2);
}

TEST(VideoBufferTest, GetFramesMoreThanAvailable) {
    VideoBuffer buffer(10);

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    buffer.addFrame(frame, 1, 1000);

    auto frames = buffer.getFrames(5);
    EXPECT_EQ(frames.size(), 1);
}

TEST(VideoBufferTest, GetFramesByDuration) {
    VideoBuffer buffer(30);

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    for (int i = 0; i < 10; ++i) {
        buffer.addFrame(frame, i, i * 1000);
    }

    auto frames = buffer.getFramesByDuration(2, 5.0);
    EXPECT_EQ(frames.size(), 10);
}

TEST(VideoBufferTest, Clear) {
    VideoBuffer buffer(10);

    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    buffer.addFrame(frame, 1, 1000);
    buffer.addFrame(frame, 2, 2000);

    EXPECT_EQ(buffer.Size(), 2);

    buffer.Clear();
    EXPECT_EQ(buffer.Size(), 0);
}
