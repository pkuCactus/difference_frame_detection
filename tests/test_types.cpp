#include <gtest/gtest.h>
#include "common/types.h"

using namespace diff_det;

TEST(BoundingBoxTest, DefaultValues) {
    BoundingBox box;
    EXPECT_FLOAT_EQ(box.x1, 0);
    EXPECT_FLOAT_EQ(box.y1, 0);
    EXPECT_FLOAT_EQ(box.x2, 0);
    EXPECT_FLOAT_EQ(box.y2, 0);
    EXPECT_FLOAT_EQ(box.conf, 0);
    EXPECT_EQ(box.label, 0);
}

TEST(BoundingBoxTest, Constructor) {
    BoundingBox box(10, 20, 100, 200, 0.85f, 1);
    
    EXPECT_FLOAT_EQ(box.x1, 10);
    EXPECT_FLOAT_EQ(box.y1, 20);
    EXPECT_FLOAT_EQ(box.x2, 100);
    EXPECT_FLOAT_EQ(box.y2, 200);
    EXPECT_FLOAT_EQ(box.conf, 0.85f);
    EXPECT_EQ(box.label, 1);
}

TEST(BoundingBoxTest, WidthHeightArea) {
    BoundingBox box(0, 0, 100, 50, 0.9f, 0);
    
    EXPECT_FLOAT_EQ(box.width(), 100);
    EXPECT_FLOAT_EQ(box.height(), 50);
    EXPECT_FLOAT_EQ(box.area(), 5000);
}

TEST(TrackTest, DefaultValues) {
    Track track;
    EXPECT_EQ(track.trackId, -1);
    EXPECT_FLOAT_EQ(track.x, 0);
    EXPECT_FLOAT_EQ(track.y, 0);
    EXPECT_FLOAT_EQ(track.w, 0);
    EXPECT_FLOAT_EQ(track.h, 0);
    EXPECT_FLOAT_EQ(track.score, 0);
    EXPECT_EQ(track.state, TrackState::Tentative);
}

TEST(TrackTest, ToBoundingBox) {
    Track track;
    track.trackId = 1;
    track.x = 10;
    track.y = 20;
    track.w = 100;
    track.h = 50;
    track.score = 0.9f;
    
    BoundingBox box = track.toBoundingBox();
    
    EXPECT_FLOAT_EQ(box.x1, 10);
    EXPECT_FLOAT_EQ(box.y1, 20);
    EXPECT_FLOAT_EQ(box.x2, 110);
    EXPECT_FLOAT_EQ(box.y2, 70);
    EXPECT_FLOAT_EQ(box.conf, 0.9f);
}

TEST(CameraDetectionResultTest, DefaultValues) {
    CameraDetectionResult result;
    EXPECT_EQ(result.objNum, 0);
    EXPECT_EQ(result.timeStamp, 0);
    EXPECT_EQ(result.frameId, -1);
    EXPECT_TRUE(result.objs.empty());
}

TEST(FrameInfoTest, DefaultValues) {
    FrameInfo info;
    EXPECT_EQ(info.frameId, -1);
    EXPECT_EQ(info.timeStamp, 0);
    EXPECT_FALSE(info.hasPerson);
    EXPECT_TRUE(info.boxes.empty());
    EXPECT_TRUE(info.tracks.empty());
}