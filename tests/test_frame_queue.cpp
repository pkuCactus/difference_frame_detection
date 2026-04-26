#include <gtest/gtest.h>
#include "utils/frame_queue.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(FrameQueueTest, Init) {
    FrameQueue queue(10);
    
    EXPECT_TRUE(queue.Empty());
    EXPECT_EQ(queue.Size(), 0);
    EXPECT_EQ(queue.GetMaxSize(), 10);
}

TEST(FrameQueueTest, PushPop) {
    FrameQueue queue(5);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    queue.Push(frame, 1, 1000);
    EXPECT_EQ(queue.Size(), 1);
    
    cv::Mat poppedFrame;
    int frameId;
    int64_t timestamp;
    
    EXPECT_TRUE(queue.Pop(poppedFrame, frameId, timestamp));
    EXPECT_EQ(frameId, 1);
    EXPECT_EQ(timestamp, 1000);
    EXPECT_FALSE(poppedFrame.empty());
    EXPECT_TRUE(queue.Empty());
}

TEST(FrameQueueTest, Overflow) {
    FrameQueue queue(3);
    
    cv::Mat frame(480, 640, CV_8UC3);
    
    for (int i = 0; i < 5; ++i) {
        queue.Push(frame, i, i * 1000);
    }
    
    EXPECT_EQ(queue.Size(), 3);
    
    cv::Mat poppedFrame;
    int frameId;
    int64_t timestamp;
    
    queue.Pop(poppedFrame, frameId, timestamp);
    EXPECT_EQ(frameId, 2);
}

TEST(FrameQueueTest, Clear) {
    FrameQueue queue(10);
    
    cv::Mat frame(480, 640, CV_8UC3);
    
    queue.Push(frame, 1, 1000);
    queue.Push(frame, 2, 2000);
    
    EXPECT_EQ(queue.Size(), 2);
    
    queue.Clear();
    
    EXPECT_TRUE(queue.Empty());
}

TEST(VideoFrameBufferTest, Init) {
    VideoFrameBuffer buffer(100);
    
    EXPECT_TRUE(buffer.Empty());
    EXPECT_EQ(buffer.Size(), 0);
}

TEST(VideoFrameBufferTest, AddFrame) {
    VideoFrameBuffer buffer(10);
    
    cv::Mat frame(480, 640, CV_8UC3, cv::Scalar(128, 128, 128));
    
    buffer.AddFrame(frame, 1, 1000);
    EXPECT_EQ(buffer.Size(), 1);
    
    EXPECT_EQ(buffer.GetOldestFrameId(), 1);
    EXPECT_EQ(buffer.GetNewestFrameId(), 1);
}

TEST(VideoFrameBufferTest, GetFrames) {
    VideoFrameBuffer buffer(10);
    
    cv::Mat frame1(480, 640, CV_8UC3, cv::Scalar(100, 100, 100));
    cv::Mat frame2(480, 640, CV_8UC3, cv::Scalar(200, 200, 200));
    cv::Mat frame3(480, 640, CV_8UC3, cv::Scalar(300, 300, 300));
    
    buffer.AddFrame(frame1, 1, 1000);
    buffer.AddFrame(frame2, 2, 2000);
    buffer.AddFrame(frame3, 3, 3000);
    
    std::vector<cv::Mat> frames = buffer.GetFrames(2);
    EXPECT_EQ(frames.size(), 2);
    
    std::vector<FrameWithMeta> framesMeta = buffer.GetFramesWithMeta(2);
    EXPECT_EQ(framesMeta.size(), 2);
    EXPECT_EQ(framesMeta[0].frameId, 2);
    EXPECT_EQ(framesMeta[1].frameId, 3);
}

TEST(VideoFrameBufferTest, GetFramesByDuration) {
    VideoFrameBuffer buffer(100);
    
    cv::Mat frame(480, 640, CV_8UC3);
    
    for (int i = 0; i < 90; ++i) {
        buffer.AddFrame(frame, i, i * 33);
    }
    
    std::vector<cv::Mat> frames = buffer.GetFramesByDuration(3, 30.0);
    EXPECT_EQ(frames.size(), 90);
}