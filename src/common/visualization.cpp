#include "common/visualization.h"
#include <sstream>
#include <iomanip>

namespace diff_det {

void DrawBoundingBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes) {
    for (const auto& box : boxes) {
        cv::Rect rect(static_cast<int>(box.x1), static_cast<int>(box.y1),
                      static_cast<int>(box.x2 - box.x1), static_cast<int>(box.y2 - box.y1));

        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);

        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << box.conf;
        std::string text = "person: " + oss.str();

        cv::Point textPos(static_cast<int>(box.x1), static_cast<int>(box.y1) - 5);
        cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 1);
    }
}

} // namespace diff_det
