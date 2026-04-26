#include <gtest/gtest.h>
#include "core/state.h"

using namespace diff_det;

TEST(StateTest, StateToString) {
    EXPECT_STREQ(StateToString(State::INIT), "INIT");
    EXPECT_STREQ(StateToString(State::CONNECTING), "CONNECTING");
    EXPECT_STREQ(StateToString(State::CHECK_CAPABILITY), "CHECK_CAPABILITY");
    EXPECT_STREQ(StateToString(State::CAMERA_DETECTION_MODE), "CAMERA_DETECTION_MODE");
    EXPECT_STREQ(StateToString(State::LOCAL_DETECTION_MODE), "LOCAL_DETECTION_MODE");
    EXPECT_STREQ(StateToString(State::DIFFERENCE_ANALYSIS), "DIFFERENCE_ANALYSIS");
    EXPECT_STREQ(StateToString(State::EVENT_ANALYSIS), "EVENT_ANALYSIS");
    EXPECT_STREQ(StateToString(State::UPDATE_REF), "UPDATE_REF");
    EXPECT_STREQ(StateToString(State::RECONNECTING), "RECONNECTING");
    EXPECT_STREQ(StateToString(State::ERROR), "ERROR");
}