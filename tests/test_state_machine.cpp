#include <gtest/gtest.h>
#include "core/state.h"

using namespace diff_det;

TEST(StateTest, StateToString) {
    EXPECT_STREQ(stateToString(State::INIT), "INIT");
    EXPECT_STREQ(stateToString(State::CONNECTING), "CONNECTING");
    EXPECT_STREQ(stateToString(State::CHECK_CAPABILITY), "CHECK_CAPABILITY");
    EXPECT_STREQ(stateToString(State::CAMERA_DETECTION_MODE), "CAMERA_DETECTION_MODE");
    EXPECT_STREQ(stateToString(State::LOCAL_DETECTION_MODE), "LOCAL_DETECTION_MODE");
    EXPECT_STREQ(stateToString(State::DIFFERENCE_ANALYSIS), "DIFFERENCE_ANALYSIS");
    EXPECT_STREQ(stateToString(State::EVENT_ANALYSIS), "EVENT_ANALYSIS");
    EXPECT_STREQ(stateToString(State::UPDATE_REF), "UPDATE_REF");
    EXPECT_STREQ(stateToString(State::RECONNECTING), "RECONNECTING");
    EXPECT_STREQ(stateToString(State::ERROR), "ERROR");
}