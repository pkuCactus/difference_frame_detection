#pragma once

#include "common/config.h"
#include "common/types.h"
#include "analysis/similarity.h"
#include <opencv2/opencv.hpp>
#include <memory>

namespace diff_det {

class IFrameDiffAnalyzer {
public:
    virtual ~IFrameDiffAnalyzer() = default;
    
    virtual bool isSimilar(const cv::Mat& current, const cv::Mat& ref) = 0;
    virtual void updateRef(const cv::Mat& frame) = 0;
    virtual bool hasRef() = 0;
    virtual cv::Mat getRef() = 0;
    virtual void reset() = 0;
};

class FrameDiffAnalyzer : public IFrameDiffAnalyzer {
public:
    FrameDiffAnalyzer(const RefFrameConfig& config);
    
    bool isSimilar(const cv::Mat& current, const cv::Mat& ref) override;
    void updateRef(const cv::Mat& frame) override;
    bool hasRef() override;
    cv::Mat getRef() override;
    void reset() override;
    
    void setBoxesForRoi(const std::vector<BoundingBox>& boxes);
    void setThreshold(float threshold);
    
    int getRefUpdateCount() const { return refUpdateCount_; }
    
private:
    cv::Mat extractRoi(const cv::Mat& frame, const std::vector<BoundingBox>& boxes);
    
    float threshold_;
    std::string compareMethod_;
    std::string updateStrategy_;
    bool compareRoiOnly_;
    cv::Mat refFrame_;
    std::vector<BoundingBox> currentBoxes_;
    std::unique_ptr<ISimilarityCalculator> calculator_;
    
    int refUpdateCount_;
};

}