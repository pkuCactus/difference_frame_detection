#include "detection/rknn_detector.h"
#include "common/logger.h"
#include <chrono>
#include <fstream>
#include <cstring>
#include <iomanip>

namespace diff_det {

RknnDetector::RknnDetector(const LocalDetectionConfig& config) 
    : config_(config)
    , initialized_(false)
    , modelLoaded_(false)
    , perfStats_(nullptr)
    , lastDetectTime_(0)
    , totalDetections_(0)
    , scaleX_(1.0f)
    , scaleY_(1.0f)
    , offsetX_(0)
    , offsetY_(0)
    , lastFrameWidth_(0)
    , lastFrameHeight_(0)
    , inputWidth_(RKNN_MODEL_INPUT_WIDTH)
    , inputHeight_(RKNN_MODEL_INPUT_HEIGHT)
    , inputChannel_(3)
    , rknnCtx_(nullptr)
    , useStubMode_(true) {
    
    postprocess_ = std::make_unique<YoloPostprocess>(
        config.modelType, config.confThreshold);
}

RknnDetector::~RknnDetector() {
    releaseBuffers();
    
    if (initialized_) {
        LOG_INFO("RknnDetector destroyed, total detections: " + std::to_string(totalDetections_));
    }
}

bool RknnDetector::init() {
    LOG_INFO("Initializing RKNN detector: model=" + config_.modelPath +
             ", model_type=" + config_.modelType +
             ", conf_threshold=" + std::to_string(config_.confThreshold) +
             ", detect_interval=" + std::to_string(config_.detectInterval));
    
    if (!loadModel()) {
        LOG_WARN("Failed to load RKNN model, using stub mode for testing");
        useStubMode_ = true;
        initialized_ = true;
        return true;
    }
    
    if (!queryModelInfo()) {
        LOG_ERROR("Failed to query model info");
        releaseBuffers();
        useStubMode_ = true;
        initialized_ = true;
        return true;
    }
    
    if (!prepareInputBuffers()) {
        LOG_ERROR("Failed to prepare input buffers");
        releaseBuffers();
        useStubMode_ = true;
        initialized_ = true;
        return true;
    }
    
    if (!prepareOutputBuffers()) {
        LOG_ERROR("Failed to prepare output buffers");
        releaseBuffers();
        useStubMode_ = true;
        initialized_ = true;
        return true;
    }
    
    modelLoaded_ = true;
    useStubMode_ = false;
    initialized_ = true;
    
    LOG_INFO("RKNN detector initialized successfully (NCHW input): " + 
             std::to_string(inputWidth_) + "x" + std::to_string(inputHeight_) +
             "x" + std::to_string(inputChannel_));
    
    return true;
}

bool RknnDetector::loadModel() {
    std::ifstream file(config_.modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_WARN("Model file not found: " + config_.modelPath + ", using stub mode");
        return false;
    }
    
    std::streamsize size = file.tellg();
    file.close();
    
    if (size == 0) {
        LOG_WARN("Model file is empty: " + config_.modelPath);
        return false;
    }
    
    LOG_INFO("Model file found: " + config_.modelPath + ", size=" + std::to_string(size) + " bytes");
    
    LOG_INFO("Note: RKNN model requires RKNN Runtime SDK on RK3566 platform");
    LOG_INFO("      Input format: NCHW (Channel x Height x Width)");
    LOG_INFO("      Stub mode will be used for testing on non-RK3566 platforms");
    
    return false;
}

bool RknnDetector::queryModelInfo() {
    modelInfo_.version = 1;
    modelInfo_.input_num = 1;
    modelInfo_.output_num = 1;
    
    modelInfo_.inputs.resize(1);
    modelInfo_.inputs[0].index = 0;
    modelInfo_.inputs[0].width = inputWidth_;
    modelInfo_.inputs[0].height = inputHeight_;
    modelInfo_.inputs[0].channel = inputChannel_;
    modelInfo_.inputs[0].format = 1;
    modelInfo_.inputs[0].type = 0;
    modelInfo_.inputs[0].size = inputWidth_ * inputHeight_ * inputChannel_;
    
    modelInfo_.outputs.resize(1);
    modelInfo_.outputs[0].index = 0;
    modelInfo_.outputs[0].n_dims = 3;
    
    if (config_.modelType == "yolov5" || config_.modelType == "yolov3") {
        modelInfo_.outputs[0].dims[0] = 1;
        modelInfo_.outputs[0].dims[1] = 25200;
        modelInfo_.outputs[0].dims[2] = 85;
        modelInfo_.outputs[0].size = 25200 * 85 * sizeof(float);
    } else if (config_.modelType == "yolov8") {
        modelInfo_.outputs[0].dims[0] = 1;
        modelInfo_.outputs[0].dims[1] = 8400;
        modelInfo_.outputs[0].dims[2] = 84;
        modelInfo_.outputs[0].size = 8400 * 84 * sizeof(float);
    } else {
        modelInfo_.outputs[0].dims[0] = 1;
        modelInfo_.outputs[0].dims[1] = 25200;
        modelInfo_.outputs[0].dims[2] = 85;
        modelInfo_.outputs[0].size = 25200 * 85 * sizeof(float);
    }
    
    modelInfo_.outputs[0].want_float = 1;
    
    LOG_INFO("Model info: input_num=" + std::to_string(modelInfo_.input_num) +
             ", output_num=" + std::to_string(modelInfo_.output_num) +
             ", input_format=NCHW" +
             ", output_size=" + std::to_string(modelInfo_.outputs[0].size));
    
    return true;
}

bool RknnDetector::prepareInputBuffers() {
    int32_t inputSize = inputWidth_ * inputHeight_ * inputChannel_;
    inputBuffer_.resize(inputSize);
    
    LOG_INFO("Input buffer prepared (NCHW float): size=" + std::to_string(inputSize) + 
             " floats = " + std::to_string(inputChannel_) + "x" +
             std::to_string(inputHeight_) + "x" + std::to_string(inputWidth_) +
             ", bytes=" + std::to_string(inputSize * sizeof(float)));
    
    return true;
}

bool RknnDetector::prepareOutputBuffers() {
    int32_t outputSize = modelInfo_.outputs[0].size / sizeof(float);
    outputBuffer_.resize(outputSize);
    
    LOG_INFO("Output buffer prepared: size=" + std::to_string(outputSize) + " floats");
    
    return true;
}

void RknnDetector::releaseBuffers() {
    inputBuffer_.clear();
    outputBuffer_.clear();
    modelLoaded_ = false;
    
    if (rknnCtx_) {
        LOG_INFO("RKNN context would be released on RK3566 platform");
        rknnCtx_ = nullptr;
    }
}

std::vector<BoundingBox> RknnDetector::detect(const cv::Mat& frame) {
    if (!initialized_) {
        LOG_ERROR("Detector not initialized");
        return {};
    }
    
    if (frame.empty()) {
        LOG_WARN("Empty frame received");
        return {};
    }
    
    auto startTime = std::chrono::high_resolution_clock::now();
    
    if (perfStats_) {
        perfStats_->startTimer("detector_preprocess");
    }
    
    float scaleX, scaleY;
    int32_t offsetX, offsetY;
    cv::Mat preprocessed = preprocess(frame, scaleX, scaleY, offsetX, offsetY);
    
    if (perfStats_) {
        perfStats_->endTimer("detector_preprocess");
        perfStats_->startTimer("detector_inference");
    }
    
    std::vector<float> outputs = runInference(preprocessed);
    
    if (perfStats_) {
        perfStats_->endTimer("detector_inference");
        perfStats_->startTimer("detector_postprocess");
    }
    
    std::vector<BoundingBox> boxes;
    
    if (!outputs.empty()) {
        boxes = postprocess_->process(
            outputs,
            static_cast<int32_t>(outputs.size()),
            inputWidth_,
            inputHeight_,
            lastFrameWidth_,
            lastFrameHeight_,
            scaleX,
            scaleY,
            offsetX,
            offsetY);
    }
    
    if (perfStats_) {
        perfStats_->endTimer("detector_postprocess");
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    lastDetectTime_ = static_cast<double>(duration.count());
    
    totalDetections_++;
    if (perfStats_) {
        perfStats_->incrementCounter("total_detections");
        perfStats_->incrementCounter("detection_time_ms", static_cast<int32_t>(lastDetectTime_));
        if (!boxes.empty()) {
            perfStats_->incrementCounter("detections_with_person", static_cast<int32_t>(boxes.size()));
        }
    }
    
    if (!boxes.empty()) {
        std::ostringstream oss;
        oss << "Detected " << boxes.size() << " persons in " << lastDetectTime_ << "ms: ";
        for (size_t i = 0; i < std::min(boxes.size(), static_cast<size_t>(3)); ++i) {
            oss << "[" << static_cast<int32_t>(boxes[i].x1) << "," << static_cast<int32_t>(boxes[i].y1) << ","
                << static_cast<int32_t>(boxes[i].x2) << "," << static_cast<int32_t>(boxes[i].y2) << " conf=" 
                << std::fixed << std::setprecision(2) << boxes[i].conf << "] ";
        }
        LOG_INFO(oss.str());
    } else {
        LOG_DEBUG("No persons detected in " + std::to_string(lastDetectTime_) + "ms");
    }
    
    return boxes;
}

void RknnDetector::setConfThreshold(float threshold) {
    config_.confThreshold = threshold;
    if (postprocess_) {
        postprocess_->setConfThreshold(threshold);
    }
    LOG_INFO("Confidence threshold updated to: " + std::to_string(threshold));
}

void RknnDetector::setPerformanceStats(PerformanceStats* stats) {
    perfStats_ = stats;
}

double RknnDetector::getLastDetectTime() {
    return lastDetectTime_;
}

int32_t RknnDetector::getTotalDetections() {
    return totalDetections_;
}

cv::Mat RknnDetector::preprocess(const cv::Mat& frame, 
                                  float& scaleX, float& scaleY,
                                  int32_t& offsetX, int32_t& offsetY) {
    lastFrameWidth_ = frame.cols;
    lastFrameHeight_ = frame.rows;
    
    scaleX = static_cast<float>(inputWidth_) / static_cast<float>(frame.cols);
    scaleY = static_cast<float>(inputHeight_) / static_cast<float>(frame.rows);
    float scale = std::min(scaleX, scaleY);
    
    scaleX = scale;
    scaleY = scale;
    
    int32_t newWidth = static_cast<int32_t>(frame.cols * scale);
    int32_t newHeight = static_cast<int32_t>(frame.rows * scale);
    
    offsetX = (inputWidth_ - newWidth) / 2;
    offsetY = (inputHeight_ - newHeight) / 2;
    
    scaleX_ = scaleX;
    scaleY_ = scaleY;
    offsetX_ = offsetX;
    offsetY_ = offsetY;
    
    cv::Mat resized;
    cv::resize(frame, resized, cv::Size(newWidth, newHeight), 0, 0, cv::INTER_LINEAR);
    
    cv::Mat letterbox(inputHeight_, inputWidth_, CV_8UC3, cv::Scalar(114, 114, 114));
    
    if (newWidth > 0 && newHeight > 0) {
        cv::Rect roi(offsetX, offsetY, newWidth, newHeight);
        resized.copyTo(letterbox(roi));
    }
    
    LOG_DEBUG("Preprocess: original=" + std::to_string(frame.cols) + "x" + std::to_string(frame.rows) +
              ", letterbox=" + std::to_string(inputWidth_) + "x" + std::to_string(inputHeight_) +
              ", scale=" + std::to_string(scale) +
              ", offset=(" + std::to_string(offsetX) + "," + std::to_string(offsetY) + ")");
    
    return letterbox;
}

void RknnDetector::fillInputBuffer(const cv::Mat& preprocessed) {
    int32_t expectedPixels = inputWidth_ * inputHeight_ * inputChannel_;
    int32_t actualPixels = static_cast<int32_t>(preprocessed.total() * preprocessed.channels());
    
    if (actualPixels != expectedPixels) {
        LOG_ERROR("Input size mismatch: expected " + std::to_string(expectedPixels) +
                  " pixels, got " + std::to_string(actualPixels));
        return;
    }
    
    inputBuffer_.resize(expectedPixels);
    
    cv::Mat channels[3];
    cv::split(preprocessed, channels);
    
    int32_t channelSize = inputWidth_ * inputHeight_;
    
    for (int32_t c = 0; c < inputChannel_; ++c) {
        int32_t channelOffset = c * channelSize;
        for (int32_t i = 0; i < channelSize; ++i) {
            inputBuffer_[channelOffset + i] = static_cast<float>(channels[c].data[i]) / 255.0f;
        }
    }
    
    LOG_DEBUG("Input buffer filled (NCHW normalized): C=" + std::to_string(inputChannel_) +
              ", H=" + std::to_string(inputHeight_) +
              ", W=" + std::to_string(inputWidth_) +
              ", normalized=/255.0, total=" + std::to_string(inputBuffer_.size()) + " floats");
}

std::vector<float> RknnDetector::runInference(const cv::Mat& preprocessed) {
    if (useStubMode_) {
        LOG_DEBUG("Running inference in stub mode (no actual RKNN inference)");
        return {};
    }
    
    fillInputBuffer(preprocessed);
    
    LOG_DEBUG("RKNN inference would run on RK3566 NPU here");
    LOG_DEBUG("Input (NCHW): " + std::to_string(inputChannel_) + "x" +
              std::to_string(inputHeight_) + "x" + std::to_string(inputWidth_));
    LOG_DEBUG("Output: " + std::to_string(modelInfo_.outputs[0].size) + " bytes");
    
    std::vector<float> outputs = parseOutputs();
    
    return outputs;
}

std::vector<float> RknnDetector::parseOutputs() {
    std::vector<float> outputs;
    
    if (outputBuffer_.empty()) {
        return outputs;
    }
    
    outputs = outputBuffer_;
    
    LOG_DEBUG("Parsed " + std::to_string(outputs.size()) + " output values");
    
    return outputs;
}

}