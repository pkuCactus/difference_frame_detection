#include "detection/rknn_detector.h"
#include "common/logger.h"
#include <chrono>
#include <fstream>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>

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
        std::cerr << "[RKNN ERROR] Adapter init failed, model=" << config_.modelPath << std::endl;
        return false;
    }

    inputWidth_ = rknnAdapter_->GetInputWidth();
    inputHeight_ = rknnAdapter_->GetInputHeight();
    inputChannel_ = rknnAdapter_->GetInputChannel();

    std::cout << "[RKNN] Adapter init OK" << std::endl;
    std::cout << "[RKNN] Input: " << inputChannel_ << "x" << inputHeight_ << "x" << inputWidth_ << std::endl;
    std::cout << "[RKNN] Input type=" << rknnAdapter_->GetInputType()
              << ", format=" << rknnAdapter_->GetInputFormat() << std::endl;
    std::cout << "[RKNN] Output num=" << rknnAdapter_->GetOutputNum() << std::endl;
    
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

bool RknnDetector::LoadModel() {
    if (!rknnAdapter_ || !rknnAdapter_->IsInitialized()) {
        return false;
    }
    return rknnAdapter_->IsInitialized();
}

bool RknnDetector::QueryModelInfo() {
    if (!rknnAdapter_ || !rknnAdapter_->IsInitialized()) {
        LOG_ERROR("RKNN adapter not initialized, cannot query model info");
        return false;
    }

    modelInfo_.version = 1;
    modelInfo_.input_num = rknnAdapter_->GetInputNum();
    modelInfo_.output_num = rknnAdapter_->GetOutputNum();

    const auto& inputAttrs = rknnAdapter_->GetInputAttrs();
    modelInfo_.inputs.resize(modelInfo_.input_num);
    for (int32_t i = 0; i < modelInfo_.input_num; ++i) {
        modelInfo_.inputs[i].index = static_cast<int32_t>(inputAttrs[i].index);
        modelInfo_.inputs[i].Width = inputWidth_;
        modelInfo_.inputs[i].Height = inputHeight_;
        modelInfo_.inputs[i].channel = inputChannel_;
        modelInfo_.inputs[i].format = static_cast<int32_t>(inputAttrs[i].fmt);
        modelInfo_.inputs[i].type = static_cast<int32_t>(inputAttrs[i].type);
        modelInfo_.inputs[i].Size = static_cast<int32_t>(inputAttrs[i].size);
    }

    const auto& outputAttrs = rknnAdapter_->GetOutputAttrs();
    modelInfo_.outputs.resize(modelInfo_.output_num);
    for (int32_t i = 0; i < modelInfo_.output_num; ++i) {
        modelInfo_.outputs[i].index = static_cast<int32_t>(outputAttrs[i].index);
        modelInfo_.outputs[i].Size = static_cast<int32_t>(outputAttrs[i].size);
        modelInfo_.outputs[i].want_float = 1;
        modelInfo_.outputs[i].fmt = static_cast<int32_t>(outputAttrs[i].fmt);
        modelInfo_.outputs[i].type = static_cast<int32_t>(outputAttrs[i].type);
        modelInfo_.outputs[i].n_dims = static_cast<int32_t>(outputAttrs[i].n_dims);
        for (int32_t d = 0; d < 4; ++d) {
            modelInfo_.outputs[i].dims[d] = static_cast<int32_t>(outputAttrs[i].dims[d]);
        }
    }

    std::ostringstream oss;
    oss << "Model info: input_num=" << modelInfo_.input_num
        << ", output_num=" << modelInfo_.output_num
        << ", input=" << inputChannel_ << "x" << inputHeight_ << "x" << inputWidth_;
    for (int32_t i = 0; i < modelInfo_.output_num; ++i) {
        oss << ", output[" << i << "]dims=[";
        for (int32_t d = 0; d < modelInfo_.outputs[i].n_dims; ++d) {
            oss << modelInfo_.outputs[i].dims[d];
            if (d + 1 < modelInfo_.outputs[i].n_dims) oss << ",";
        }
        oss << "]";
    }
    LOG_INFO(oss.str());

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
            std::cout << "[POST] Using ProcessRknnYolov5, outputs=" << multiOutputBuffers_.size() << std::endl;
            boxes = postprocess_->ProcessRknnYolov5(
                multiOutputBuffers_,
                lastFrameWidth_,
                lastFrameHeight_,
                scaleX,
                scaleY,
                offsetX,
                offsetY,
                inputWidth_,
                inputHeight_);
        } else {
            std::cout << "[POST] Using Process (single output), outputs.size=" << outputs.size() << std::endl;
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
    } else {
        std::cout << "[POST] WARNING: outputs is empty!" << std::endl;
    }

    std::cout << "[POST] Boxes after NMS: " << boxes.size() << std::endl;

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

    cv::Mat rgbResized;
    cv::cvtColor(resized, rgbResized, cv::COLOR_BGR2RGB);

    cv::Mat letterbox(inputHeight_, inputWidth_, CV_8UC3, cv::Scalar(114, 114, 114));

    if (newWidth > 0 && newHeight > 0) {
        cv::Rect roi(offsetX, offsetY, newWidth, newHeight);
        rgbResized.copyTo(letterbox(roi));
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

    int32_t inputType = rknnAdapter_->GetInputType();
    int32_t inputFormat = rknnAdapter_->GetInputFormat();

    bool isFp32 = (inputType == 1);  // RKNN_TENSOR_FLOAT32
    bool isNchw = (inputFormat == 1); // RKNN_TENSOR_NCHW

    if (isFp32) {
        // FP32输入需要归一化到[0,1]
        std::vector<float> inputFloat(inputWidth_ * inputHeight_ * inputChannel_);
        if (isNchw) {
            cv::Mat channels[3];
            cv::split(preprocessed, channels);
            int32_t channelSize = inputWidth_ * inputHeight_;
            for (int32_t c = 0; c < inputChannel_; ++c) {
                for (int32_t i = 0; i < channelSize; ++i) {
                    inputFloat[c * channelSize + i] =
                        static_cast<float>(channels[c].data[i]) / 255.0f;
                }
            }
        } else {
            for (int32_t h = 0; h < inputHeight_; ++h) {
                for (int32_t w = 0; w < inputWidth_; ++w) {
                    cv::Vec3b pixel = preprocessed.at<cv::Vec3b>(h, w);
                    int32_t base = (h * inputWidth_ + w) * inputChannel_;
                    inputFloat[base + 0] = pixel[0] / 255.0f;
                    inputFloat[base + 1] = pixel[1] / 255.0f;
                    inputFloat[base + 2] = pixel[2] / 255.0f;
                }
            }
        }
        int32_t bytes = static_cast<int32_t>(inputFloat.size() * sizeof(float));
        if (!rknnAdapter_->SetInputBuffer(
                reinterpret_cast<const uint8_t*>(inputFloat.data()), bytes)) {
            LOG_ERROR("Failed to set FP32 input buffer");
            return {};
        }
    } else {
        // UINT8/INT8输入直接memcpy
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
            LOG_ERROR("Failed to set UINT8 input buffer");
            return {};
        }
    }

    std::cout << "[RKNN] SetInputBuffer OK, bytes="
              << (isFp32 ? inputWidth_ * inputHeight_ * inputChannel_ * 4 : inputWidth_ * inputHeight_ * inputChannel_)
              << std::endl;

    if (!rknnAdapter_->Run()) {
        std::cerr << "[RKNN ERROR] rknn_run failed" << std::endl;
        return {};
    }

    int32_t outputNum = rknnAdapter_->GetOutputNum();
    multiOutputBuffers_.resize(outputNum);
    std::cout << "[RKNN] Output num=" << outputNum << std::endl;

    for (int32_t i = 0; i < outputNum; ++i) {
        int32_t outputSize = rknnAdapter_->GetOutputSize(i);
        std::cout << "[RKNN] Output[" << i << "] size=" << outputSize << std::endl;
        if (outputSize > 0) {
            multiOutputBuffers_[i].resize(outputSize);
            if (!rknnAdapter_->GetOutputBuffer(i, multiOutputBuffers_[i].data(), outputSize)) {
                std::cerr << "[RKNN ERROR] GetOutputBuffer failed for index " << i << std::endl;
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

    std::cout << "[RKNN] Inference OK, output0_size=" << outputs.size() << std::endl;
    if (!outputs.empty()) {
        std::cout << "[RKNN] Output[0..9]: ";
        for (size_t i = 0; i < std::min(outputs.size(), size_t(10)); ++i) {
            std::cout << outputs[i] << " ";
        }
        std::cout << std::endl;

        // 打印最大置信度（YOLOv5格式：每85个值中第4个是obj_conf）
        if (config_.modelType == "yolov5" || config_.modelType == "yolov3") {
            int stride = 85;
            float maxObjConf = 0.0f;
            for (size_t i = 4; i < outputs.size(); i += stride) {
                maxObjConf = std::max(maxObjConf, outputs[i]);
            }
            std::cout << "[RKNN] Max obj_conf=" << maxObjConf << std::endl;
        } else if (config_.modelType == "yolov8") {
            int stride = 84;
            float maxClassConf = 0.0f;
            for (size_t i = 4; i < outputs.size() && i + 80 < outputs.size(); i += stride) {
                for (int c = 0; c < 80; ++c) {
                    maxClassConf = std::max(maxClassConf, outputs[i + c]);
                }
            }
            std::cout << "[RKNN] Max class_conf=" << maxClassConf << std::endl;
        }
    }

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