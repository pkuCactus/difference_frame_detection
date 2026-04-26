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
    
    virtual bool Init(const CameraDetectionConfig& config) = 0;
    virtual CameraDetectionResult GetDetectionResult() = 0;
    virtual bool MatchFrame(int frameId, int64_t timestamp, 
                            const CameraDetectionResult& result) = 0;
};

class CameraDetectionReader : public ICameraDetectionReader {
public:
    CameraDetectionReader();
    ~CameraDetectionReader();
    
    bool Init(const CameraDetectionConfig& config) override;
    CameraDetectionResult GetDetectionResult() override;
    bool MatchFrame(int frameId, int64_t timestamp, 
                    const CameraDetectionResult& result) override;
    
private:
    CameraDetectionResult FetchByRest();
    CameraDetectionResult FetchByOnvif();
    
    CameraDetectionConfig config_;
    std::deque<CameraDetectionResult> resultQueue_;
    int64_t lastFetchTime_;
    int64_t lastMatchedFrameId_;
    int64_t lastMatchedTimestamp_;
};

}