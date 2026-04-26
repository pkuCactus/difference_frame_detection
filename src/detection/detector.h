#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace diff_det {

class IDetector {
public:
    virtual ~IDetector() = default;
    
    virtual std::vector<BoundingBox> detect(const cv::Mat& frame) = 0;
    virtual bool init() = 0;
    virtual void setConfThreshold(float threshold) = 0;
};

}