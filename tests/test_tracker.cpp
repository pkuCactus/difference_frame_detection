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
    
    std::vector<Track> tracks = tracker.Update(frame, boxes);
    EXPECT_TRUE(tracks.empty());
}

TEST(ByteTrackerTest, UpdateWithBoxes) {
    TrackerConfig config;
    config.confirmFrames = 1;
    ByteTracker tracker(config);
    
    cv::Mat frame(480, 640, CV_8UC3);
    
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));
    
    std::vector<Track> tracks = tracker.Update(frame, boxes);
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
    
    std::vector<Track> tracks = tracker.Update(frame, boxes);
    EXPECT_GE(tracks.size(), 1);
    
    int firstTrackId = tracks[0].trackId;
    
    for (int i = 0; i < 3; ++i) {
        BoundingBox similarBox(105, 105, 205, 205, 0.9f, 0);
        std::vector<BoundingBox> frameBoxes;
        frameBoxes.push_back(similarBox);
        
        tracks = tracker.Update(frame, frameBoxes);
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
    
    tracker.Update(frame, boxes);
    std::vector<Track> predictions = tracker.Predict();
    
    EXPECT_FALSE(predictions.empty());
}

TEST(ByteTrackerTest, Reset) {
    TrackerConfig config;
    config.confirmFrames = 1;
    ByteTracker tracker(config);

    cv::Mat frame(480, 640, CV_8UC3);
    std::vector<BoundingBox> boxes;
    boxes.push_back(BoundingBox(100, 100, 200, 200, 0.9f, 0));

    tracker.Update(frame, boxes);
    tracker.reset();

    std::vector<Track> tracks = tracker.Update(frame, {});
    EXPECT_TRUE(tracks.empty());
}

TEST(ByteTrackerTest, TentativeTrackGetsConfirmed) {
    TrackerConfig config;
    config.confirmFrames = 3;
    config.maxLostFrames = 30;
    config.highThreshold = 0.5f;
    ByteTracker tracker(config);

    cv::Mat frame(480, 640, CV_8UC3);

    for (int i = 0; i < 5; ++i) {
        std::vector<BoundingBox> boxes;
        boxes.push_back(BoundingBox(100 + i * 2, 100 + i * 2, 200 + i * 2, 200 + i * 2, 0.9f, 0));
        std::vector<Track> tracks = tracker.Update(frame, boxes);

        if (i < 2) {
            EXPECT_TRUE(tracks.empty()) << "Frame " << i << " should have no confirmed tracks";
        } else {
            EXPECT_EQ(tracks.size(), 1) << "Frame " << i << " should have 1 confirmed track";
            if (!tracks.empty()) {
                EXPECT_EQ(tracks[0].state, TrackState::Tracked);
            }
        }
    }
}