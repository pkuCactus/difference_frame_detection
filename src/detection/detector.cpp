#include "detection/detector.h"
#include <iostream>

namespace diff_det {
constexpr int OBJ_CLASS_NUM = 80;
constexpr int PROP_BOX_SIZE = 5 + OBJ_CLASS_NUM;
constexpr int MAX_DETECTIONS = 64;

const std::vector<std::vector<int32_t>> ANCHORS = {
    {10, 13, 16, 30, 33, 23},
    {30, 61, 62, 45, 59, 119},
    {116, 90, 156, 198, 373, 326}
};

inline float Sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

inline float DeqntAffineToF32(int8_t qnt, int32_t zp, float scale) {
    return static_cast<float>(qnt - zp) * scale;
}

inline int8_t QntF32ToAffine(float f32, int32_t zp, float scale) {
    float dstVal = (f32 / scale) + zp;
    return static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, dstVal)));
}

inline int ClampInt(float val, int32_t min, int32_t max) {
    int32_t tVal = static_cast<int32_t>(val);
    if (tVal > max) {
        return max;
    }
    if (tVal < min) {
        return min;
    }
    return tVal;
}

Detector::Detector(const LocalDetectionConfig& config) : config_(config)
{
    LOG_INFO("Detector created with config: " + config_.ToString());
}

bool Detector::Init()
{
    if (initialized_) {
        LOG_WARN("Detector already initialized");
        return true;
    }
    if (!LoadModel()) {
        LOG_ERROR("Failed to load model");
        return false;
    }
    if (!config_.labelPath.empty()) {
        if (!LoadLabels()) {
            LOG_ERROR("Failed to load labels");
            return false;
        }
    }
    initialized_ = true;
    LOG_INFO("Detector initialized successfully");
    return true;
}

void Detector::DeInit()
{
    if (rknnCtx_) {
        rknn_destroy(rknnCtx_);
        rknnCtx_ = 0;
        LOG_INFO("Detector de-initialized and resources released");
    }
    initialized_ = false;
    labelNames_.clear();
    inputAttrs_.clear();
    outputAttrs_.clear();
}

bool Detector::LoadModel()
{
    FILE* fp = fopen(config_.modelPath.c_str(), "rb");
    if (!fp) {
        LOG_ERROR("Failed to open model file: " + config_.modelPath);
        return false;
    }
    fseek(fp, 0, SEEK_END);
    int32_t modelSize = ftell(fp);
    fseek(fp, 0, SEEK_SET);

    uint8_t* modelData = new uint8_t[modelSize];
    fread(modelData, 1, modelSize, fp);
    fclose(fp);

    int32_t ret = rknn_init(&rknnCtx_, modelData, modelSize, 0, nullptr);
    delete[] modelData;
    if (ret != 0) {
        LOG_ERROR("rknn_init failed with error code: " + std::to_string(ret));
        return false;
    }

    rknn_sdk_version version;
    ret = rknn_query(rknnCtx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret != 0) {
        LOG_ERROR("rknn_query SDK version failed with error code: " + std::to_string(ret));
        return false;
    }

    rknn_input_output_num ioNum;
    ret = rknn_query(rknnCtx_, RKNN_QUERY_IN_OUT_NUM, &ioNum, sizeof(ioNum));
    if (ret != 0) {
        LOG_ERROR("rknn_query input/output num failed with error code: " + std::to_string(ret));
        return false;
    }
    inputNum_ = ioNum.n_input;
    outputNum_ = ioNum.n_output;

    inputAttrs_.resize(inputNum_);
    for (int32_t i = 0; i < inputNum_; ++i) {
        inputAttrs_[i].index = i;
        ret = rknn_query(rknnCtx_, RKNN_QUERY_INPUT_ATTR, &inputAttrs_[i], sizeof(inputAttrs_[i]));
        if (ret != 0) {
            LOG_ERROR("rknn_query input attr failed with error code: " + std::to_string(ret));
            return false;
        }
        LOG_INFO("Model inputs[" + std::to_string(i) + "] type: " + std::to_string(inputAttrs_[i].type));
    }

    outputAttrs_.resize(outputNum_);
    for (int32_t i = 0; i < outputNum_; ++i) {
        outputAttrs_[i].index = i;
        ret = rknn_query(rknnCtx_, RKNN_QUERY_OUTPUT_ATTR, &outputAttrs_[i], sizeof(outputAttrs_[i]));
        if (ret != 0) {
            LOG_ERROR("rknn_query output attr failed with error code: " + std::to_string(ret));
            return false;
        }
        LOG_INFO("Model outputs[" + std::to_string(i) + "] type: " + std::to_string(outputAttrs_[i].type));
    }
    if (inputAttrs_[0].fmt == RKNN_TENSOR_NHWC) {
        LOG_INFO("model is NHWC fmt");
        modelInputHeight_ = inputAttrs_[0].dims[1];
        modelInputWidth_ = inputAttrs_[0].dims[2];
        modelInputChannel_ = inputAttrs_[0].dims[3];
    } else {
        LOG_INFO("model is NCHW fmt");
        modelInputChannel_ = inputAttrs_[0].dims[1];
        modelInputHeight_ = inputAttrs_[0].dims[2];
        modelInputWidth_ = inputAttrs_[0].dims[3];
    }
    LOG_INFO("Model Loaded: SDK version " + std::string(version.api_version) +
             ", Driver version " +std::string(version.drv_version) +
             ", Input num=" + std::to_string(inputNum_) + ", Output num=" + std::to_string(outputNum_) +
             ", Input shape=" + std::to_string(modelInputChannel_) + "x" +
             std::to_string(modelInputHeight_) + "x" + std::to_string(modelInputWidth_));
    return true;
}

bool Detector::LoadLabels()
{
    std::ifstream file(config_.labelPath);
    if (!file.is_open()) {
        LOG_ERROR("Failed to open labels file: " + config_.labelPath);
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {
            labelNames_.push_back(line);
        }
    }

    LOG_INFO("Loaded " + std::to_string(labelNames_.size()) + " labels from " + config_.labelPath);
    return true;
}

cv::Mat Detector::PreProcess(const cv::Mat& frame, float& scaleW, float& scaleH, int32_t& offsetW, int32_t& offsetH)
{
    cv::Mat rgbImg;
    cv::cvtColor(frame, rgbImg, cv::COLOR_BGR2RGB);
    cv::Mat resized;
    if (!config_.useLetterBox) {
        cv::resize(rgbImg, resized, cv::Size(modelInputWidth_, modelInputHeight_));
        scaleW = static_cast<float>(modelInputWidth_) / frame.cols;
        scaleH = static_cast<float>(modelInputHeight_) / frame.rows;
        offsetW = 0;
        offsetH = 0;
        return resized;
    }
    scaleW = static_cast<float>(modelInputWidth_) / frame.cols;
    scaleH = static_cast<float>(modelInputHeight_) / frame.rows;
    float scale = std::min(scaleW, scaleH);
    scaleH = scaleW = scale;
    int32_t newW = static_cast<int32_t>(frame.cols * scale);
    int32_t newH = static_cast<int32_t>(frame.rows * scale);
    cv::resize(rgbImg, resized, cv::Size(newW, newH));
    offsetW = (modelInputWidth_ - newW) / 2;
    offsetH = (modelInputHeight_ - newH) / 2;
    cv::Mat letterBoxed(modelInputHeight_, modelInputWidth_, rgbImg.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(letterBoxed(cv::Rect(offsetW, offsetH, newW, newH)));
    return letterBoxed;
}

void Detector::SetupInputs(const cv::Mat& frame)
{
    rknn_input inputs[inputNum_];
    memset(inputs, 0, sizeof(inputs));
    inputs[0].index = 0;
    inputs[0].type = RKNN_TENSOR_UINT8;
    inputs[0].size = modelInputWidth_ * modelInputHeight_ * modelInputChannel_;
    inputs[0].fmt = RKNN_TENSOR_NHWC;
    inputs[0].pass_through = 0;
    inputs[0].buf = frame.data;

    int32_t ret = rknn_inputs_set(rknnCtx_, inputNum_, inputs);
    if (ret != 0) {
        LOG_ERROR("rknn_inputs_set failed with error code: " + std::to_string(ret));
    }
    LOG_DEBUG("Set inputs successful");
}

bool WantFloat(rknn_tensor_type type)
{
    if (type == RKNN_TENSOR_FLOAT16) {
        return true;
    }
    return false;
}

void Detector::PostProcess(const cv::Mat& frame, const std::vector<rknn_output> outputs, float scaleW, float scaleH, int32_t offsetW,
    int32_t offsetH, std::vector<BoundingBox>& boxes)
{
    LOG_DEBUG("PostProcess ...");
    for (int32_t outIdx = 0; outIdx < outputNum_; ++outIdx) {
        int32_t anchorNum = ANCHORS[outIdx].size() / 2;
        rknn_tensor_attr& attr = outputAttrs_[outIdx];
        int8_t confThresh = QntF32ToAffine(config_.confThreshold, attr.zp, attr.scale);
        LOG_DEBUG("Conf threshold: " + std::to_string(config_.confThreshold) + " after quant: " + std::to_string(confThresh));
        int8_t* data = reinterpret_cast<int8_t*>(outputs[outIdx].buf);
        int32_t gridH = attr.dims[attr.n_dims - 2];
        int32_t gridW = attr.dims[attr.n_dims - 1];
        int32_t gridLen = gridH * gridW;
        LOG_DEBUG("Process the outputs [" + std::to_string(outIdx) + "]: " + "achor num: " + std::to_string(anchorNum) + ", " +
            "shape: " + std::to_string(attr.dims[0]) + "x" + std::to_string(attr.dims[1]) + "x" + std::to_string(attr.dims[2]) +
            "x" + std::to_string(attr.dims[3]));
        for (int32_t anchorIdx = 0; anchorIdx < anchorNum; ++anchorIdx) {
            for (int32_t i = 0; i < gridH; ++i) {
                for (int32_t j = 0; j < gridW; ++j) {
                    int32_t idx = anchorIdx * PROP_BOX_SIZE * gridLen + i * attr.dims[3] + j;
                    int8_t objConf = data[idx + 4 * gridLen];
                    // std::cout << anchorIdx << " " << i << " " << j << " " << idx << " " << (int32_t)objConf << " " << (int32_t)confThresh << std::endl;
                    if (objConf < confThresh) {
                        continue;
                    }
                    int32_t label = 0;
                    int8_t classProb = 0.0f;
                    // 当前只检测人
                    for (int32_t c = 0; c < 1; ++c) {
                        if (data[idx + (5 + c) * gridLen] > classProb) {
                            classProb = data[idx + (5 + c) * gridLen];
                            label = c;
                        }
                    }
                    if (classProb < confThresh) {
                        continue;
                    }
                    float xCenter = (DeqntAffineToF32(data[idx], attr.zp, attr.scale) * 2 - 0.5f + j) * (modelInputWidth_ / gridW);
                    float yCenter = (DeqntAffineToF32(data[idx + gridLen], attr.zp, attr.scale) * 2 - 0.5f + i) * (modelInputHeight_ / gridH);
                    float w = powf(DeqntAffineToF32(data[idx + 2 * gridLen], attr.zp, attr.scale) * 2, 2) * ANCHORS[outIdx][anchorIdx * 2];
                    float h = powf(DeqntAffineToF32(data[idx + 3 * gridLen], attr.zp, attr.scale) * 2, 2) * ANCHORS[outIdx][anchorIdx * 2 + 1];
                    // std::cout << xCenter << " " << yCenter << " " << w << " " << h << std::endl;
                    float x1 = xCenter - w / 2;
                    float y1 = yCenter - h / 2;
                    float x2 = xCenter + w / 2;
                    float y2 = yCenter + h / 2;
                    // std::cout << x1 << " " << y1 << " " << x2 << " " << y2 << std::endl;
                    float score = DeqntAffineToF32(objConf, attr.zp, attr.scale) * DeqntAffineToF32(classProb, attr.zp, attr.scale);

                    BoundingBox box(x1, y1, x2, y2, score, label);
                    // std::cout << scaleW << " " << scaleH << " " << offsetW << " " << offsetH << std::endl;
                    // Adjust box to original image scale
                    box.x1 = ClampInt((box.x1 - offsetW) / scaleW, 0, frame.cols - 1);
                    box.y1 = ClampInt((box.y1 - offsetH) / scaleH, 0, frame.rows - 1);
                    box.x2 = ClampInt((box.x2 - offsetW) / scaleW, 0, frame.cols - 1);
                    box.y2 = ClampInt((box.y2 - offsetH) / scaleH, 0, frame.rows - 1);
                    // std::cout << box.x1 << " " << box.y1 << " " << box.x2 << " " << box.y2 << std::endl;
                    boxes.push_back(box);
                }
            }
        }
    }
    LOG_DEBUG("Total boxes before nms: " + std::to_string(boxes.size()));
    boxes = Nms(boxes);
    LOG_DEBUG("Boxes remain after nms: " + std::to_string(boxes.size()));
}

std::vector<BoundingBox> Detector::Nms(std::vector<BoundingBox>& boxes)
{
    std::vector<BoundingBox> result {};
    if (boxes.empty()) {
        return result;
    }
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox &a, const BoundingBox &b){
        return a.conf > b.conf;
    });
    std::vector<int32_t> suppressed(boxes.size(), 0);
    for (int32_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        result.emplace_back(boxes[i]);
        for (int32_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j] || boxes[i].label != boxes[j].label) {
                continue;
            }

            float x1 = std::max(boxes[i].x1, boxes[j].x1);
            float y1 = std::max(boxes[i].y1, boxes[j].y1);
            float x2 = std::min(boxes[i].x2, boxes[j].x2);
            float y2 = std::min(boxes[i].y2, boxes[j].y2);

            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float areaI = boxes[i].Area();
            float areaJ = boxes[j].Area();
            float unionArea = areaI + areaJ - intersection;

            float iou = intersection / unionArea;

            if (iou > config_.nmsThreshold) {
                suppressed[j] = true;
            }
        }
    }
    return result;
}

std::vector<BoundingBox> Detector::Detect(const cv::Mat& frame)
{
    std::vector<BoundingBox> boxes;
    if (!initialized_) {
        LOG_ERROR("Detector not initialized, cannot perform detection");
        return boxes;
    }
    auto startTime = std::chrono::high_resolution_clock::now();

    if (perfStats_) {
        perfStats_->StartTimer("detector_preprocess");
    }
    LOG_DEBUG("Start detecting");
    float scaleW = 1.0f;
    float scaleH = 1.0f;
    int32_t offsetW = 0.0f;
    int32_t offsetH = 0.0f;
    cv::Mat preprocessed = PreProcess(frame, scaleW, scaleH, offsetW, offsetH);
    LOG_DEBUG("preprocess successful: " + std::to_string(scaleW) + " " + std::to_string(scaleH) +
    " " + std::to_string(offsetW) + " " + std::to_string(offsetH));
    SetupInputs(preprocessed);
    std::vector<rknn_output> outputs(outputNum_);
    memset(outputs.data(), 0, sizeof(rknn_output) * outputNum_);
    for (int32_t i = 0; i < outputNum_; ++i) {
        outputs[i].index = i;
        outputs[i].want_float = WantFloat(outputAttrs_[i].type);
    }
    if (perfStats_) {
        perfStats_->EndTimer("detector_preprocess");
        perfStats_->StartTimer("detector_inference");
    }
    int32_t ret = rknn_run(rknnCtx_, nullptr);
    if (ret < 0) {
        LOG_INFO("rknn run failed: " + std::to_string(ret));
        return boxes;
    }
    if (perfStats_) {
        perfStats_->EndTimer("detector_inference");
        perfStats_->StartTimer("detector_postprocess");
    }
    LOG_DEBUG("RKNN inference successful");
    ret = rknn_outputs_get(rknnCtx_, outputNum_, outputs.data(), nullptr);
    if (ret < 0) {
        LOG_INFO("rknn get output failed: " + std::to_string(ret));
        return boxes;
    }
    PostProcess(frame, outputs, scaleW, scaleH, offsetW, offsetH, boxes);
    rknn_outputs_release(rknnCtx_, outputNum_, outputs.data());
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

void Detector::SetConfThreshold(float threshold)
{
    config_.confThreshold = threshold;
}

void Detector::SetNmsThreshold(float threshold)
{
    config_.nmsThreshold = threshold;
}

void Detector::SetPerformanceStats(PerformanceStats* performanceStats)
{
    perfStats_ = performanceStats;
}

} // namespace diff_det