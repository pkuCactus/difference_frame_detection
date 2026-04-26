#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace diff_det {

class IDetector {
public:
    virtual ~IDetector() = default;
    
    virtual std::vector<BoundingBox> Detect(const cv::Mat& frame) = 0;
    virtual bool Init() = 0;
    virtual void SetConfThreshold(float threshold) = 0;
};

}