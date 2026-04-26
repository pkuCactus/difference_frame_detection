#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace diff_det {

class ITracker {
public:
    virtual ~ITracker() = default;
    
    virtual std::vector<Track> Update(const cv::Mat& frame, 
                                       const std::vector<BoundingBox>& boxes) = 0;
    virtual std::vector<Track> Predict() = 0;
    virtual void reset() = 0;
};

}