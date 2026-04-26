#pragma once

#include "common/config.h"
#include "common/types.h"
#include "analysis/similarity.h"
#include <opencv2/opencv.hpp>
#include <memory>

namespace diff_det {

enum class RefUpdateStrategy {
    kNewest,
    kDefault
};

class IFrameDiffAnalyzer {
public:
    virtual ~IFrameDiffAnalyzer() = default;
    
    virtual bool IsSimilar(const cv::Mat& current, const cv::Mat& ref) = 0;
    virtual void UpdateRef(const cv::Mat& frame) = 0;
    virtual bool HasRef() = 0;
    virtual cv::Mat GetRef() = 0;
    virtual void Reset() = 0;
};

class FrameDiffAnalyzer : public IFrameDiffAnalyzer {
public:
    FrameDiffAnalyzer(const RefFrameConfig& config);
    
    bool IsSimilar(const cv::Mat& current, const cv::Mat& ref) override;
    void UpdateRef(const cv::Mat& frame) override;
    bool HasRef() override;
    cv::Mat GetRef() override;
    void Reset() override;
    
    void SetBoxesForRoi(const std::vector<BoundingBox>& boxes);
    void SetThreshold(float threshold);
    
    int GetRefUpdateCount() const { return refUpdateCount_; }
    
private:
    cv::Mat ExtractRoi(const cv::Mat& frame, const std::vector<BoundingBox>& boxes);
    
    float threshold_;
    std::string compareMethod_;
    RefUpdateStrategy updateStrategy_;
    bool compareRoiOnly_;
    cv::Mat refFrame_;
    std::vector<BoundingBox> currentBoxes_;
    std::unique_ptr<ISimilarityCalculator> calculator_;
    
    int refUpdateCount_;
};

}