#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace diff_det {

struct BoundingBox {
    float x1;
    float y1;
    float x2;
    float y2;
    float conf;
    int32_t label;
    
    BoundingBox() : x1(0), y1(0), x2(0), y2(0), conf(0), label(0) {}
    
    BoundingBox(float x1Val, float y1Val, float x2Val, float y2Val, float confVal, int32_t labelVal)
        : x1(x1Val), y1(y1Val), x2(x2Val), y2(y2Val), conf(confVal), label(labelVal) {}
    
    float Width() const { return x2 - x1; }
    float Height() const { return y2 - y1; }
    float Area() const { return Width() * Height(); }
};

enum class TrackState {
    Tentative,
    Tracked,
    Lost,
    Removed
};

struct Track {
    int32_t trackId;
    float x;
    float y;
    float w;
    float h;
    float score;
    TrackState state;
    
    Track() : trackId(-1), x(0), y(0), w(0), h(0), score(0), state(TrackState::Tentative) {}
    
    BoundingBox ToBoundingBox() const {
        return BoundingBox(x, y, x + w, y + h, score, 0);
    }
};

struct CameraDetectionResult {
    int32_t objNum;
    std::vector<BoundingBox> objs;
    int64_t timeStamp;
    int32_t frameId;
    
    CameraDetectionResult() : objNum(0), timeStamp(0), frameId(-1) {}
};

struct FrameInfo {
    int32_t frameId;
    int64_t timeStamp;
    bool hasPerson;
    std::vector<BoundingBox> boxes;
    std::vector<Track> tracks;
    
    FrameInfo() : frameId(-1), timeStamp(0), hasPerson(false) {}
};

}