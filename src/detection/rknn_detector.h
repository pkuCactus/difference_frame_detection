#pragma once

#include "detection/detector.h"
#include "detection/yolo_postprocess.h"
#include "detection/rknn_adapter.h"
#include "common/config.h"
#include "common/performance_stats.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>
#include <memory>
#include <cstdint>

namespace diff_det {

#define RKNN_MODEL_INPUT_WIDTH 832
#define RKNN_MODEL_INPUT_HEIGHT 448

struct RknnInputInfo {
    int32_t index;
    int32_t Width;
    int32_t Height;
    int32_t channel;
    int32_t format;
    int32_t type;
    int32_t Size;
};

struct RknnOutputInfo {
    int32_t index;
    int32_t Size;
    int32_t want_float;
    int32_t fmt;
    int32_t type;
    int32_t n_dims;
    int32_t dims[4];
};

struct RknnModelInfo {
    int32_t version;
    int32_t input_num;
    int32_t output_num;
    std::vector<RknnInputInfo> inputs;
    std::vector<RknnOutputInfo> outputs;
};

class RknnDetector : public IDetector {
public:
    RknnDetector(const LocalDetectionConfig& config);
    ~RknnDetector();
    
    std::vector<BoundingBox> Detect(const cv::Mat& frame) override;
    bool Init() override;
    void SetConfThreshold(float threshold) override;
    
    void setPerformanceStats(PerformanceStats* stats);
    
    double GetLastDetectTime();
    int32_t GetTotalDetections();
    int32_t GetInputWidth() const { return inputWidth_; }
    int32_t GetInputHeight() const { return inputHeight_; }
    bool IsModelLoaded() const { return modelLoaded_; }
    
private:
    bool LoadModel();
    bool QueryModelInfo();
    bool PrepareInputBuffers();
    bool PrepareOutputBuffers();
    void ReleaseBuffers();
    cv::Mat Preprocess(const cv::Mat& frame, float& scaleX, float& scaleY,
                       int32_t& offsetX, int32_t& offsetY);
    std::vector<float> RunInference(const cv::Mat& preprocessed);
    std::vector<float> ParseOutputs();
    
    void FillInputBuffer(const cv::Mat& preprocessed);
    
    LocalDetectionConfig config_;
    bool initialized_;
    bool modelLoaded_;
    std::unique_ptr<YoloPostprocess> postprocess_;
    
    PerformanceStats* perfStats_;
    
    double lastDetectTime_;
    int32_t totalDetections_;
    
    float scaleX_;
    float scaleY_;
    int32_t offsetX_;
    int32_t offsetY_;
    
    int32_t lastFrameWidth_;
    int32_t lastFrameHeight_;
    
    int32_t inputWidth_;
    int32_t inputHeight_;
    int32_t inputChannel_;
    
    RknnModelInfo modelInfo_;
    
    std::vector<float> inputBuffer_;
    std::vector<float> outputBuffer_;
    std::vector<std::vector<float>> multiOutputBuffers_;

    std::unique_ptr<RknnAdapter> rknnAdapter_;
};

}