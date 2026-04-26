#pragma once

#include "detection/detector.h"
#include "detection/yolo_postprocess.h"
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
    int32_t width;
    int32_t height;
    int32_t channel;
    int32_t format;
    int32_t type;
    int32_t size;
};

struct RknnOutputInfo {
    int32_t index;
    int32_t size;
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
    
    std::vector<BoundingBox> detect(const cv::Mat& frame) override;
    bool init() override;
    void setConfThreshold(float threshold) override;
    
    void setPerformanceStats(PerformanceStats* stats);
    
    double getLastDetectTime();
    int32_t getTotalDetections();
    int32_t getInputWidth() const { return inputWidth_; }
    int32_t getInputHeight() const { return inputHeight_; }
    bool isModelLoaded() const { return modelLoaded_; }
    
private:
    bool loadModel();
    bool queryModelInfo();
    bool prepareInputBuffers();
    bool prepareOutputBuffers();
    void releaseBuffers();
    
    cv::Mat preprocess(const cv::Mat& frame, float& scaleX, float& scaleY,
                       int32_t& offsetX, int32_t& offsetY);
    std::vector<float> runInference(const cv::Mat& preprocessed);
    std::vector<float> parseOutputs();
    
    void fillInputBuffer(const cv::Mat& preprocessed);
    
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
    
    void* rknnCtx_;
    bool useStubMode_;
};

}