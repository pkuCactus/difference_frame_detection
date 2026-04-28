#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>

namespace diff_det {

void DrawBoundingBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes);

} // namespace diff_det
