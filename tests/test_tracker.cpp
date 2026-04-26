#include <gtest/gtest.h>
#include "tracking/byte_tracker.h"
#include "common/config.h"
#include "common/types.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

TEST(ByteTrackerTest, Init) {
    TrackerConfig config;
    ByteTracker tracker(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    std::vector<BoundingBox> boxes;
    
    std::vector<Track> tracks = tracker.update(frame, boxes);
    EXPECT_TRUE(tracks.empty());
}

TEST(ByteTrackerTest, UpdateWithBoxes) {
    TrackerConfig config;
    config.confirmFrames = 1;
    ByteTracker tracker(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    std::vector<Track> tracks = tracker.update(frame, boxes);
    EXPECT_GE(tracks.size(), 1);
    EXPECT_GE(tracks[0].trackId, 0);
}

TEST(ByteTrackerTest, MultiFrameTracking) {
    TrackerConfig config;
    config.confirmFrames = 1;
    config.maxLostFrames = 30;
    ByteTracker tracker(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    std::vector<Track> tracks = tracker.update(frame, boxes);
    EXPECT_GE(tracks.size(), 1);
    
    int firstTrackId = tracks[0].trackId;
    
    for (int i = 0; i < 3; ++i) {
        BoundingBox similarBox(105, 105, 205, 205, 0.9f, 0);
        std::vector<BoundingBox> frameBoxes;
        frameBoxes.push_back(similarBox);
        
        tracks = tracker.update(frame, frameBoxes);
    }
    
    EXPECT_GE(tracks.size(), 1);
    EXPECT_EQ(tracks[0].trackId, firstTrackId);
    EXPECT_EQ(tracks[0].state, TrackState::Tracked);
}

TEST(ByteTrackerTest, Predict) {
    TrackerConfig config;
    config.confirmFrames = 1;
    ByteTracker tracker(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    tracker.update(frame, boxes);
    std::vector<Track> predictions = tracker.predict();
    
    EXPECT_FALSE(predictions.empty());
}

TEST(ByteTrackerTest, Reset) {
    TrackerConfig config;
    config.confirmFrames = 1;
    ByteTracker tracker(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    tracker.update(frame, boxes);
    tracker.reset();
    
    std::vector<Track> tracks = tracker.update(frame, {});
    EXPECT_TRUE(tracks.empty());
}