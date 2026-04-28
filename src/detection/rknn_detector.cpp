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
    , rknnAdapter_(nullptr) {
    
    postprocess_ = std::make_unique<YoloPostprocess>(
        config.modelType, config.confThreshold);
}

RknnDetector::~RknnDetector() {
    ReleaseBuffers();
    
    if (initialized_) {
        LOG_INFO("RknnDetector destroyed, total detections: " + std::to_string(totalDetections_));
    }
}

bool RknnDetector::Init() {
    LOG_INFO("Initializing RKNN detector: model=" + config_.modelPath +
             ", model_type=" + config_.modelType +
             ", conf_threshold=" + std::to_string(config_.confThreshold) +
             ", detect_interval=" + std::to_string(config_.detectInterval));

    rknnAdapter_ = std::make_unique<RknnAdapter>();
    
    if (!rknnAdapter_->Init(config_.modelPath)) {
        LOG_WARN("Failed to initialize RKNN adapter, detector will return empty results");
        initialized_ = true;
        return true;
    }
    
    inputWidth_ = rknnAdapter_->GetInputWidth();
    inputHeight_ = rknnAdapter_->GetInputHeight();
    inputChannel_ = rknnAdapter_->GetInputChannel();
    
    LOG_INFO("RKNN adapter initialized: input=" +
             std::to_string(inputChannel_) + "x" +
             std::to_string(inputHeight_) + "x" +
             std::to_string(inputWidth_) + " (NCHW)");
    
    if (!PrepareInputBuffers()) {
        LOG_ERROR("Failed to prepare input buffers");
        return true;
    }
    
    if (!PrepareOutputBuffers()) {
        LOG_ERROR("Failed to prepare output buffers");
        return true;
    }
    
    QueryModelInfo();
    
    modelLoaded_ = true;
    initialized_ = true;
    
    LOG_INFO("RKNN detector initialized successfully");
    
    return true;
}

bool RknnDetector::FallbackToStubMode(bool releaseBuffers) {
    if (releaseBuffers) {
        ReleaseBuffers();
    }
    initialized_ = true;
    return true;
}

bool RknnDetector::LoadModel() {
    if (!rknnAdapter_ || !rknnAdapter_->IsInitialized()) {
        return false;
    }
    return rknnAdapter_->IsInitialized();
}

bool RknnDetector::QueryModelInfo() {
    if (!rknnAdapter_ || !rknnAdapter_->IsInitialized()) {
        modelInfo_.version = 1;
        modelInfo_.input_num = 1;
        modelInfo_.output_num = 1;
        
        modelInfo_.inputs.resize(1);
        modelInfo_.inputs[0].index = 0;
        modelInfo_.inputs[0].Width = inputWidth_;
        modelInfo_.inputs[0].Height = inputHeight_;
        modelInfo_.inputs[0].channel = inputChannel_;
        modelInfo_.inputs[0].format = 1;
        modelInfo_.inputs[0].type = 0;
        modelInfo_.inputs[0].Size = inputWidth_ * inputHeight_ * inputChannel_;
        
        modelInfo_.outputs.resize(1);
        modelInfo_.outputs[0].index = 0;
        modelInfo_.outputs[0].n_dims = 3;
        
        if (config_.modelType == "yolov5" || config_.modelType == "yolov3") {
            modelInfo_.outputs[0].dims[0] = 1;
            modelInfo_.outputs[0].dims[1] = 25200;
            modelInfo_.outputs[0].dims[2] = 85;
            modelInfo_.outputs[0].Size = 25200 * 85 * sizeof(float);
        } else if (config_.modelType == "yolov8") {
            modelInfo_.outputs[0].dims[0] = 1;
            modelInfo_.outputs[0].dims[1] = 8400;
            modelInfo_.outputs[0].dims[2] = 84;
            modelInfo_.outputs[0].Size = 8400 * 84 * sizeof(float);
        } else {
            modelInfo_.outputs[0].dims[0] = 1;
            modelInfo_.outputs[0].dims[1] = 25200;
            modelInfo_.outputs[0].dims[2] = 85;
            modelInfo_.outputs[0].Size = 25200 * 85 * sizeof(float);
        }
        
        modelInfo_.outputs[0].want_float = 1;
        
        LOG_INFO("Model info (stub): input_num=" + std::to_string(modelInfo_.input_num) +
                 ", output_num=" + std::to_string(modelInfo_.output_num) +
                 ", input_format=NCHW" +
                 ", output_size=" + std::to_string(modelInfo_.outputs[0].Size));
        
        return true;
    }
    
    int32_t outputSize = rknnAdapter_->GetOutputSize(0);
    if (outputSize <= 0) {
        outputSize = 25200 * 85;
    }
    
    modelInfo_.version = 1;
    modelInfo_.input_num = 1;
    modelInfo_.output_num = 1;
    
    modelInfo_.inputs.resize(1);
    modelInfo_.inputs[0].index = 0;
    modelInfo_.inputs[0].Width = inputWidth_;
    modelInfo_.inputs[0].Height = inputHeight_;
    modelInfo_.inputs[0].channel = inputChannel_;
    modelInfo_.inputs[0].format = 1;
    modelInfo_.inputs[0].type = 0;
    modelInfo_.inputs[0].Size = inputWidth_ * inputHeight_ * inputChannel_;
    
    modelInfo_.outputs.resize(1);
    modelInfo_.outputs[0].index = 0;
    modelInfo_.outputs[0].n_dims = 3;
    modelInfo_.outputs[0].dims[0] = 1;
    modelInfo_.outputs[0].dims[1] = outputSize / 85;
    modelInfo_.outputs[0].dims[2] = 85;
    modelInfo_.outputs[0].Size = outputSize * sizeof(float);
    modelInfo_.outputs[0].want_float = 1;
    
    LOG_INFO("Model info: input_num=" + std::to_string(modelInfo_.input_num) +
             ", output_num=" + std::to_string(modelInfo_.output_num) +
             ", input=" + std::to_string(inputChannel_) + "x" +
             std::to_string(inputHeight_) + "x" + std::to_string(inputWidth_) +
             ", output_size=" + std::to_string(modelInfo_.outputs[0].Size));
    
    return true;
}

bool RknnDetector::PrepareInputBuffers() {
    int32_t inputSize = inputWidth_ * inputHeight_ * inputChannel_;
    inputBuffer_.resize(inputSize);
    
    LOG_INFO("Input buffer prepared (NCHW float): Size=" + std::to_string(inputSize) + 
             " floats = " + std::to_string(inputChannel_) + "x" +
             std::to_string(inputHeight_) + "x" + std::to_string(inputWidth_) +
             ", bytes=" + std::to_string(inputSize * sizeof(float)));
    
    return true;
}

bool RknnDetector::PrepareOutputBuffers() {
    int32_t outputSize = 0;
    
    if (rknnAdapter_ && rknnAdapter_->IsInitialized()) {
        outputSize = rknnAdapter_->GetOutputSize(0);
    }
    
    if (outputSize <= 0) {
        if (config_.modelType == "yolov8") {
            outputSize = 8400 * 84;
        } else {
            outputSize = 25200 * 85;
        }
    }
    
    outputBuffer_.resize(outputSize);
    
    LOG_INFO("Output buffer prepared: Size=" + std::to_string(outputSize) + " floats");
    
    return true;
}

void RknnDetector::ReleaseBuffers() {
    inputBuffer_.clear();
    outputBuffer_.clear();
    modelLoaded_ = false;
    
    if (rknnAdapter_) {
        rknnAdapter_->release();
        LOG_INFO("RKNN adapter released");
    }
}

std::vector<BoundingBox> RknnDetector::Detect(const cv::Mat& frame) {
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
        perfStats_->StartTimer("detector_preprocess");
    }
    
    float scaleX, scaleY;
    int32_t offsetX, offsetY;
    cv::Mat preprocessed = Preprocess(frame, scaleX, scaleY, offsetX, offsetY);
    
    if (perfStats_) {
        perfStats_->EndTimer("detector_preprocess");
        perfStats_->StartTimer("detector_inference");
    }
    
    std::vector<float> outputs = RunInference(preprocessed);
    
    if (perfStats_) {
        perfStats_->EndTimer("detector_inference");
        perfStats_->StartTimer("detector_postprocess");
    }
    
    std::vector<BoundingBox> boxes;

    if (!outputs.empty()) {
        if (multiOutputBuffers_.size() > 1 && config_.modelType == "yolov5") {
            boxes = postprocess_->ProcessRknnYolov5(
                multiOutputBuffers_,
                lastFrameWidth_,
                lastFrameHeight_,
                scaleX,
                scaleY,
                offsetX,
                offsetY);
        } else {
            boxes = postprocess_->Process(
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
    }
    
    if (perfStats_) {
        perfStats_->EndTimer("detector_postprocess");
    }
    
    auto endTime = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    lastDetectTime_ = static_cast<double>(duration.count());
    
    totalDetections_++;
    if (perfStats_) {
        perfStats_->IncrementCounter("total_detections");
        perfStats_->IncrementCounter("detection_time_ms", static_cast<int32_t>(lastDetectTime_));
        if (!boxes.empty()) {
            perfStats_->IncrementCounter("detections_with_person", static_cast<int32_t>(boxes.size()));
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

void RknnDetector::SetConfThreshold(float threshold) {
    config_.confThreshold = threshold;
    if (postprocess_) {
        postprocess_->SetConfThreshold(threshold);
    }
    LOG_INFO("Confidence threshold updated to: " + std::to_string(threshold));
}

void RknnDetector::setPerformanceStats(PerformanceStats* stats) {
    perfStats_ = stats;
}

double RknnDetector::GetLastDetectTime() {
    return lastDetectTime_;
}

int32_t RknnDetector::GetTotalDetections() {
    return totalDetections_;
}

cv::Mat RknnDetector::Preprocess(const cv::Mat& frame, 
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

void RknnDetector::FillInputBuffer(const cv::Mat& preprocessed) {
    int32_t expectedPixels = inputWidth_ * inputHeight_ * inputChannel_;
    int32_t actualPixels = static_cast<int32_t>(preprocessed.total() * preprocessed.channels());
    
    if (actualPixels != expectedPixels) {
        LOG_ERROR("Input Size mismatch: expected " + std::to_string(expectedPixels) +
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

std::vector<float> RknnDetector::RunInference(const cv::Mat& preprocessed) {
    if (!rknnAdapter_ || !rknnAdapter_->IsInitialized()) {
        LOG_DEBUG("RKNN adapter not initialized, returning empty results");
        return {};
    }

    std::vector<uint8_t> inputUint8(inputWidth_ * inputHeight_ * inputChannel_);
    int32_t rowSize = inputWidth_ * inputChannel_;
    if (preprocessed.isContinuous()) {
        std::memcpy(inputUint8.data(), preprocessed.data, inputUint8.size());
    } else {
        for (int32_t h = 0; h < inputHeight_; ++h) {
            std::memcpy(inputUint8.data() + h * rowSize, preprocessed.ptr(h), rowSize);
        }
    }

    if (!rknnAdapter_->SetInputBuffer(inputUint8.data(), inputUint8.size())) {
        LOG_ERROR("Failed to set input buffer");
        return {};
    }

    if (!rknnAdapter_->Run()) {
        LOG_ERROR("RKNN inference failed");
        return {};
    }

    int32_t outputNum = rknnAdapter_->GetOutputNum();
    multiOutputBuffers_.resize(outputNum);

    for (int32_t i = 0; i < outputNum; ++i) {
        int32_t outputSize = rknnAdapter_->GetOutputSize(i);
        if (outputSize > 0) {
            multiOutputBuffers_[i].resize(outputSize);
            if (!rknnAdapter_->GetOutputBuffer(i, multiOutputBuffers_[i].data(), outputSize)) {
                LOG_ERROR("Failed to get output buffer for index " + std::to_string(i));
                return {};
            }
        }
    }

    rknnAdapter_->ReleaseOutputs();

    std::vector<float> outputs;
    if (!multiOutputBuffers_.empty()) {
        outputs = multiOutputBuffers_[0];
        outputBuffer_ = outputs;
    }

    LOG_DEBUG("RKNN inference completed: output_num=" + std::to_string(outputNum) +
              ", output0_size=" + std::to_string(outputs.size()));

    return outputs;
}

std::vector<float> RknnDetector::ParseOutputs() {
    std::vector<float> outputs;
    
    if (outputBuffer_.empty()) {
        return outputs;
    }
    
    outputs = outputBuffer_;
    
    LOG_DEBUG("Parsed " + std::to_string(outputs.size()) + " output values");
    
    return outputs;
}

}