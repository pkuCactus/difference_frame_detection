#include <gtest/gtest.h>
#include "analysis/event_analyzer.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

namespace {

// Test helpers to eliminate copy-paste in tests
class EventAnalyzerTestHelper {
public:
    static EventAnalysisConfig MakeImageConfig() {
        EventAnalysisConfig config;
        config.mode = "image";
        config.videoDurationSec = 5;
        return config;
    }

    static EventAnalysisConfig MakeVideoConfig() {
        EventAnalysisConfig config;
        config.mode = "video";
        config.videoDurationSec = 5;
        return config;
    }

    static cv::Mat MakeFrame(const cv::Scalar& color = cv::Scalar(128, 128, 128)) {
        return cv::Mat(480, 640, CV_8UC3, color);
    }

    static std::vector<BoundingBox> MakeBoxes(size_t count = 1) {
        std::vector<BoundingBox> boxes;
        for (size_t i = 0; i < count; ++i) {
            boxes.emplace_back(100.0f, 100.0f, 200.0f, 200.0f, 0.9f, 0);
        }
        return boxes;
    }

    static EventCallback MakeFlagCallback(bool& flag) {
        return [&flag](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
            flag = true;
        };
    }

    static EventCallback MakeCountCallback(int& count) {
        return [&count](const cv::Mat&, const std::vector<BoundingBox>&, int, int64_t) {
            ++count;
        };
    }
};

} // namespace

TEST(EventAnalyzerTest, Constructor) {
    auto config = EventAnalyzerTestHelper::MakeImageConfig();
    EventAnalyzer analyzer(config);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeImageEmptyFrame) {
    EventAnalyzer analyzer(EventAnalyzerTestHelper::MakeImageConfig());

    bool callbackCalled = false;
    analyzer.setEventCallback(EventAnalyzerTestHelper::MakeFlagCallback(callbackCalled));

    cv::Mat emptyFrame;
    auto boxes = EventAnalyzerTestHelper::MakeBoxes();

    analyzer.AnalyzeImage(emptyFrame, boxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeImageEmptyBoxes) {
    EventAnalyzer analyzer(EventAnalyzerTestHelper::MakeImageConfig());

    bool callbackCalled = false;
    analyzer.setEventCallback(EventAnalyzerTestHelper::MakeFlagCallback(callbackCalled));

    auto frame = EventAnalyzerTestHelper::MakeFrame();
    std::vector<BoundingBox> emptyBoxes;

    analyzer.AnalyzeImage(frame, emptyBoxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeImageValidInput) {
    EventAnalyzer analyzer(EventAnalyzerTestHelper::MakeImageConfig());

    bool callbackCalled = false;
    int receivedFrameId = 0;
    analyzer.setEventCallback([&callbackCalled, &receivedFrameId](const cv::Mat&, const std::vector<BoundingBox>&, int frameId, int64_t) {
        callbackCalled = true;
        receivedFrameId = frameId;
    });

    auto frame = EventAnalyzerTestHelper::MakeFrame();
    auto boxes = EventAnalyzerTestHelper::MakeBoxes();

    analyzer.AnalyzeImage(frame, boxes);

    EXPECT_TRUE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 1);
    EXPECT_EQ(receivedFrameId, 1);
}

TEST(EventAnalyzerTest, AnalyzeVideoEmptyFrames) {
    EventAnalyzer analyzer(EventAnalyzerTestHelper::MakeVideoConfig());

    bool callbackCalled = false;
    analyzer.setEventCallback(EventAnalyzerTestHelper::MakeFlagCallback(callbackCalled));

    std::vector<cv::Mat> emptyFrames;
    auto boxes = EventAnalyzerTestHelper::MakeBoxes();

    analyzer.AnalyzeVideo(emptyFrames, boxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeVideoEmptyBoxes) {
    EventAnalyzer analyzer(EventAnalyzerTestHelper::MakeVideoConfig());

    bool callbackCalled = false;
    analyzer.setEventCallback(EventAnalyzerTestHelper::MakeFlagCallback(callbackCalled));

    std::vector<cv::Mat> frames;
    frames.push_back(EventAnalyzerTestHelper::MakeFrame());
    std::vector<BoundingBox> emptyBoxes;

    analyzer.AnalyzeVideo(frames, emptyBoxes);

    EXPECT_FALSE(callbackCalled);
    EXPECT_EQ(analyzer.GetEventCount(), 0);
}

TEST(EventAnalyzerTest, AnalyzeVideoValidInput) {
    EventAnalyzer analyzer(EventAnalyzerTestHelper::MakeVideoConfig());

    int callbackCount = 0;
    analyzer.setEventCallback(EventAnalyzerTestHelper::MakeCountCallback(callbackCount));

    std::vector<cv::Mat> frames;
    frames.push_back(EventAnalyzerTestHelper::MakeFrame(cv::Scalar(128, 128, 128)));
    frames.push_back(EventAnalyzerTestHelper::MakeFrame(cv::Scalar(64, 64, 64)));

    auto boxes = EventAnalyzerTestHelper::MakeBoxes();

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

    buffer.addFrame(EventAnalyzerTestHelper::MakeFrame(), 1, 1000);

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

    auto frame = EventAnalyzerTestHelper::MakeFrame();
    buffer.addFrame(frame, 1, 1000);
    buffer.addFrame(frame, 2, 2000);
    buffer.addFrame(frame, 3, 3000);
    buffer.addFrame(frame, 4, 4000);

    EXPECT_EQ(buffer.Size(), 3);
}

TEST(VideoBufferTest, GetFrames) {
    VideoBuffer buffer(10);

    buffer.addFrame(EventAnalyzerTestHelper::MakeFrame(cv::Scalar(128, 128, 128)), 1, 1000);
    buffer.addFrame(EventAnalyzerTestHelper::MakeFrame(cv::Scalar(64, 64, 64)), 2, 2000);
    buffer.addFrame(EventAnalyzerTestHelper::MakeFrame(cv::Scalar(32, 32, 32)), 3, 3000);

    auto frames = buffer.getFrames(2);
    EXPECT_EQ(frames.size(), 2);
}

TEST(VideoBufferTest, GetFramesMoreThanAvailable) {
    VideoBuffer buffer(10);

    buffer.addFrame(EventAnalyzerTestHelper::MakeFrame(), 1, 1000);

    auto frames = buffer.getFrames(5);
    EXPECT_EQ(frames.size(), 1);
}

TEST(VideoBufferTest, GetFramesByDuration) {
    VideoBuffer buffer(30);

    auto frame = EventAnalyzerTestHelper::MakeFrame();
    for (int i = 0; i < 10; ++i) {
        buffer.addFrame(frame, i, i * 1000);
    }

    auto frames = buffer.getFramesByDuration(2, 5.0);
    EXPECT_EQ(frames.size(), 10);
}

TEST(VideoBufferTest, Clear) {
    VideoBuffer buffer(10);

    auto frame = EventAnalyzerTestHelper::MakeFrame();
    buffer.addFrame(frame, 1, 1000);
    buffer.addFrame(frame, 2, 2000);

    EXPECT_EQ(buffer.Size(), 2);

    buffer.Clear();
    EXPECT_EQ(buffer.Size(), 0);
}
