#include "analysis/frame_diff.h"
#include "common/logger.h"
#include <algorithm>

namespace diff_det {

FrameDiffAnalyzer::FrameDiffAnalyzer(const RefFrameConfig& config)
    : threshold_(config.similarityThreshold)
    , compareMethod_(config.compareMethod)
    , updateStrategy_(config.updateStrategy)
    , compareRoiOnly_(config.compareRoiOnly)
    , refUpdateCount_(0) {
    
    if (compareMethod_ == "ssim") {
        calculator_ = std::make_unique<SsimCalculator>();
    } else if (compareMethod_ == "pixel_diff") {
        calculator_ = std::make_unique<PixelDiffCalculator>();
    } else if (compareMethod_ == "phash") {
        calculator_ = std::make_unique<HashCalculator>();
    } else {
        LOG_WARN("Unknown compare method: " + compareMethod_ + ", using ssim");
        compareMethod_ = "ssim";
        calculator_ = std::make_unique<SsimCalculator>();
    }
    
    LOG_INFO("FrameDiffAnalyzer initialized: method=" + compareMethod_ +
             ", threshold=" + std::to_string(threshold_) +
             ", updateStrategy=" + updateStrategy_ +
             ", compareRoiOnly=" + std::string(compareRoiOnly_ ? "true" : "false"));
}

bool FrameDiffAnalyzer::IsSimilar(const cv::Mat& current, const cv::Mat& ref) {
    if (current.empty()) {
        LOG_WARN("Current frame is Empty");
        return false;
    }
    
    if (ref.empty()) {
        LOG_INFO("Ref frame is Empty, must enter event analysis");
        return false;
    }
    
    cv::Mat compareCurrent = current;
    cv::Mat compareRef = ref;
    
    if (compareRoiOnly_ && !currentBoxes_.empty()) {
        compareCurrent = ExtractRoi(current, currentBoxes_);
        
        std::vector<BoundingBox> refBoxes = currentBoxes_;
        compareRef = ExtractRoi(ref, refBoxes);
        
        if (compareCurrent.empty() || compareRef.empty()) {
            LOG_WARN("ROI extraction failed, using full frame");
            compareCurrent = current;
            compareRef = ref;
        }
    }
    
    if (compareCurrent.size() != compareRef.size()) {
        cv::resize(compareCurrent, compareCurrent, compareRef.size());
    }
    
    float similarity = calculator_->Calculate(compareCurrent, compareRef);
    
    bool isSimilar = similarity >= threshold_;
    
    LOG_INFO("Frame comparison: similarity=" + std::to_string(similarity) +
             ", threshold=" + std::to_string(threshold_) +
             ", result=" + std::string(isSimilar ? "similar" : "different"));

    return isSimilar;
}

void FrameDiffAnalyzer::UpdateRef(const cv::Mat& frame) {
    if (frame.empty()) {
        LOG_WARN("Cannot update ref with empty frame");
        return;
    }
    
    if (updateStrategy_ == "newest") {
        if (!currentBoxes_.empty()) {
            refFrame_ = frame.clone();
            refUpdateCount_++;
            LOG_INFO("Ref frame updated (newest strategy), count=" + std::to_string(refUpdateCount_));
        }
    } else if (updateStrategy_ == "default") {
        refFrame_ = frame.clone();
        refUpdateCount_++;
        LOG_INFO("Ref frame updated (default strategy), count=" + std::to_string(refUpdateCount_));
    } else {
        LOG_WARN("Unknown update strategy: " + updateStrategy_);
        refFrame_ = frame.clone();
        refUpdateCount_++;
    }
}

bool FrameDiffAnalyzer::HasRef() {
    return !refFrame_.empty();
}

cv::Mat FrameDiffAnalyzer::GetRef() {
    return refFrame_;
}

void FrameDiffAnalyzer::Reset() {
    refFrame_.release();
    currentBoxes_.clear();
    refUpdateCount_ = 0;
    LOG_INFO("FrameDiffAnalyzer reset");
}

void FrameDiffAnalyzer::SetBoxesForRoi(const std::vector<BoundingBox>& boxes) {
    currentBoxes_ = boxes;
}

void FrameDiffAnalyzer::SetThreshold(float threshold) {
    threshold_ = threshold;
    LOG_INFO("Threshold updated to: " + std::to_string(threshold_));
}

cv::Mat FrameDiffAnalyzer::ExtractRoi(const cv::Mat& frame, 
                                        const std::vector<BoundingBox>& boxes) {
    if (boxes.empty() || frame.empty()) {
        return cv::Mat();
    }
    
    float minX = frame.cols;
    float minY = frame.rows;
    float maxX = 0;
    float maxY = 0;
    
    for (const auto& box : boxes) {
        minX = std::min(minX, box.x1);
        minY = std::min(minY, box.y1);
        maxX = std::max(maxX, box.x2);
        maxY = std::max(maxY, box.y2);
    }
    
    minX = std::max(0.0f, minX);
    minY = std::max(0.0f, minY);
    maxX = std::min((float)frame.cols, maxX);
    maxY = std::min((float)frame.rows, maxY);
    
    if (maxX <= minX || maxY <= minY) {
        LOG_WARN("Invalid ROI bounds");
        return cv::Mat();
    }
    
    cv::Rect roi(static_cast<int>(minX), static_cast<int>(minY),
                 static_cast<int>(maxX - minX), static_cast<int>(maxY - minY));
    
    return frame(roi).clone();
}

}