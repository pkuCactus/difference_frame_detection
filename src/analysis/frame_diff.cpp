#include "analysis/frame_diff.h"
#include "common/logger.h"
#include <algorithm>
#include <functional>
#include <unordered_map>

namespace diff_det {

namespace {

RefUpdateStrategy ParseUpdateStrategy(const std::string& strategy) {
    static const std::unordered_map<std::string, RefUpdateStrategy> kStrategyMap = {
        {"newest", RefUpdateStrategy::kNewest},
        {"default", RefUpdateStrategy::kDefault},
    };
    auto it = kStrategyMap.find(strategy);
    if (it != kStrategyMap.end()) {
        return it->second;
    }
    LOG_WARN("Unknown update strategy: " + strategy + ", using default");
    return RefUpdateStrategy::kDefault;
}

std::unique_ptr<ISimilarityCalculator> CreateCalculator(const std::string& method,
                                                          std::string& outMethod) {
    static const std::unordered_map<std::string, std::function<std::unique_ptr<ISimilarityCalculator>()>> kCalculatorFactory = {
        {"ssim", []() { return std::make_unique<SsimCalculator>(); }},
        {"pixel_diff", []() { return std::make_unique<PixelDiffCalculator>(); }},
        {"phash", []() { return std::make_unique<HashCalculator>(); }},
    };
    auto it = kCalculatorFactory.find(method);
    if (it != kCalculatorFactory.end()) {
        outMethod = method;
        return it->second();
    }
    LOG_WARN("Unknown compare method: " + method + ", using ssim");
    outMethod = "ssim";
    return std::make_unique<SsimCalculator>();
}

const char* StrategyToString(RefUpdateStrategy strategy) {
    switch (strategy) {
        case RefUpdateStrategy::kNewest: return "newest";
        case RefUpdateStrategy::kDefault: return "default";
    }
    return "default";
}

} // namespace

FrameDiffAnalyzer::FrameDiffAnalyzer(const RefFrameConfig& config)
    : threshold_(config.similarityThreshold)
    , compareRoiOnly_(config.compareRoiOnly)
    , refUpdateCount_(0) {

    updateStrategy_ = ParseUpdateStrategy(config.updateStrategy);
    calculator_ = CreateCalculator(config.compareMethod, compareMethod_);

    LOG_INFO("FrameDiffAnalyzer initialized: method=" + compareMethod_ +
             ", threshold=" + std::to_string(threshold_) +
             ", updateStrategy=" + StrategyToString(updateStrategy_) +
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
    
    if (updateStrategy_ == RefUpdateStrategy::kNewest && currentBoxes_.empty()) {
        return;
    }

    refFrame_ = frame.clone();
    refUpdateCount_++;
    LOG_INFO("Ref frame updated (" + std::string(StrategyToString(updateStrategy_)) + " strategy), count=" + std::to_string(refUpdateCount_));
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