#pragma once

#include "common/types.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>
#include <chrono>
#include <deque>

namespace diff_det {

class ICameraDetectionReader {
public:
    virtual ~ICameraDetectionReader() = default;
    
    virtual bool init(const CameraDetectionConfig& config) = 0;
    virtual CameraDetectionResult getDetectionResult() = 0;
    virtual bool matchFrame(int frameId, int64_t timestamp, 
                            const CameraDetectionResult& result) = 0;
};

class CameraDetectionReader : public ICameraDetectionReader {
public:
    CameraDetectionReader();
    ~CameraDetectionReader();
    
    bool init(const CameraDetectionConfig& config) override;
    CameraDetectionResult getDetectionResult() override;
    bool matchFrame(int frameId, int64_t timestamp, 
                    const CameraDetectionResult& result) override;
    
private:
    CameraDetectionResult fetchByRest();
    CameraDetectionResult fetchByOnvif();
    
    CameraDetectionConfig config_;
    std::deque<CameraDetectionResult> resultQueue_;
    int64_t lastFetchTime_;
    int64_t lastMatchedFrameId_;
    int64_t lastMatchedTimestamp_;
};

}