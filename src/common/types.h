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
    int label;
    
    BoundingBox() : x1(0), y1(0), x2(0), y2(0), conf(0), label(0) {}
    
    BoundingBox(float _x1, float _y1, float _x2, float _y2, float _conf, int _label)
        : x1(_x1), y1(_y1), x2(_x2), y2(_y2), conf(_conf), label(_label) {}
    
    float width() const { return x2 - x1; }
    float height() const { return y2 - y1; }
    float area() const { return width() * height(); }
};

enum class TrackState {
    Tentative,
    Tracked,
    Lost,
    Removed
};

struct Track {
    int trackId;
    float x;
    float y;
    float w;
    float h;
    float score;
    TrackState state;
    
    Track() : trackId(-1), x(0), y(0), w(0), h(0), score(0), state(TrackState::Tentative) {}
    
    BoundingBox toBoundingBox() const {
        return BoundingBox(x, y, x + w, y + h, score, 0);
    }
};

struct CameraDetectionResult {
    int objNum;
    std::vector<BoundingBox> objs;
    int64_t timeStamp;
    int frameId;
    
    CameraDetectionResult() : objNum(0), timeStamp(0), frameId(-1) {}
};

struct FrameInfo {
    int frameId;
    int64_t timeStamp;
    bool hasPerson;
    std::vector<BoundingBox> boxes;
    std::vector<Track> tracks;
    
    FrameInfo() : frameId(-1), timeStamp(0), hasPerson(false) {}
};

}