#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include "rknn_api.h"
#include "common/types.h"
#include "common/config.h"
#include "common/logger.h"
#include "common/performance_stats.h"

namespace diff_det {

class IDetector {
public:
    virtual ~IDetector() = default;

    virtual std::vector<BoundingBox> Detect(const cv::Mat& frame) = 0;
    virtual bool Init() = 0;
    virtual void SetConfThreshold(float threshold) = 0;
    virtual void SetNmsThreshold(float threshold) = 0;
};

class Detector : public IDetector {
public:
    Detector(const LocalDetectionConfig& config);
    ~Detector() override = default;
    std::vector<BoundingBox> Detect(const cv::Mat& frame) override;
    std::vector<BoundingBox> Detect(const std::string& imagePath);
    bool Init() override;
    void DeInit();
    void SetConfThreshold(float threshold) override;
    void SetNmsThreshold(float threshold) override;
    void SetPerformanceStats(PerformanceStats* performanceStats);
    int32_t GetInputWidth() {return modelInputWidth_;}
    int32_t GetInputHeight() {return modelInputHeight_;}
    double GetLastDetectTime() {return 0.;};

private:
    bool LoadModel();
    bool LoadLabels();
    void SetupInputs(const cv::Mat& frame);
    cv::Mat PreProcess(const cv::Mat& frame, float& scaleX, float& scaleY, int32_t& offsetX, int32_t& offsetY);
    void PostProcess(const cv::Mat& frame, const std::vector<rknn_output> outputs, float scaleW, float scaleH, int32_t offsetW,
        int32_t offsetH, std::vector<BoundingBox>& boxes);
    std::vector<BoundingBox> Nms(std::vector<BoundingBox> &boxes);

private:
    rknn_context rknnCtx_ {0};
    bool initialized_ {false};
    int32_t inputNum_ {0};
    int32_t outputNum_ {0};
    int32_t modelInputWidth_ {0};
    int32_t modelInputHeight_ {0};
    int32_t modelInputChannel_ {0};

    LocalDetectionConfig config_ {};
    std::vector<std::string> labelNames_ {};
    std::vector<rknn_tensor_attr> inputAttrs_ {};
    std::vector<rknn_tensor_attr> outputAttrs_ {};
};

}