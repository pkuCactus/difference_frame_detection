#pragma once

namespace diff_det {

enum class State {
    INIT,
    CONNECTING,
    CHECK_CAPABILITY,
    CAMERA_DETECTION_MODE,
    LOCAL_DETECTION_MODE,
    DIFFERENCE_ANALYSIS,
    EVENT_ANALYSIS,
    UPDATE_REF,
    RECONNECTING,
    ERROR
};

inline const char* StateToString(State state) {
    switch (state) {
        case State::INIT: return "INIT";
        case State::CONNECTING: return "CONNECTING";
        case State::CHECK_CAPABILITY: return "CHECK_CAPABILITY";
        case State::CAMERA_DETECTION_MODE: return "CAMERA_DETECTION_MODE";
        case State::LOCAL_DETECTION_MODE: return "LOCAL_DETECTION_MODE";
        case State::DIFFERENCE_ANALYSIS: return "DIFFERENCE_ANALYSIS";
        case State::EVENT_ANALYSIS: return "EVENT_ANALYSIS";
        case State::UPDATE_REF: return "UPDATE_REF";
        case State::RECONNECTING: return "RECONNECTING";
        case State::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

}