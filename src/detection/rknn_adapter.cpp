#include "detection/rknn_adapter.h"
#include "common/logger.h"
#include <fstream>
#include <cstring>
#include <string>

namespace diff_det {

RknnAdapter::RknnAdapter()
    : rknnCtx_(nullptr)
    , initialized_(false)
    , inputWidth_(832)
    , inputHeight_(448)
    , inputChannel_(3)
    , inputNum_(1)
    , outputNum_(1) {
}

RknnAdapter::~RknnAdapter() {
    release();
}

bool RknnAdapter::checkPlatform() {
#ifdef RK3566_PLATFORM
    LOG_INFO("Running on RK3566 platform with RKNN SDK");
    return true;
#else
    LOG_INFO("Not running on RK3566 platform, RKNN stub mode enabled");
    return false;
#endif
}

bool RknnAdapter::init(const std::string& modelPath) {
    LOG_INFO("Initializing RKNN adapter with model: " + modelPath);
    
    if (!checkPlatform()) {
        LOG_WARN("RK3566 platform not detected, using stub mode");
        initialized_ = true;
        return true;
    }
    
    std::vector<uint8_t> modelData;
    if (!loadModelFile(modelPath, modelData)) {
        LOG_ERROR("Failed to load model file: " + modelPath);
        return false;
    }
    
    LOG_INFO("Model loaded: size=" + std::to_string(modelData.size()) + " bytes");
    
#ifdef RK3566_PLATFORM
    
    int32_t ret = rknn_init(&rknnCtx_, modelData.data(), modelData.size(), 0, nullptr);
    if (ret < 0) {
        LOG_ERROR("rknn_init failed: ret=" + std::to_string(ret));
        return false;
    }
    
    LOG_INFO("RKNN model initialized successfully");
    
    rknn_sdk_version version;
    ret = rknn_query(rknnCtx_, RKNN_QUERY_SDK_VERSION, &version, sizeof(version));
    if (ret == RKNN_SUCC) {
        LOG_INFO("RKNN SDK version: " + std::string(version.api_version));
    }
    
    rknn_input_output_num io_num;
    ret = rknn_query(rknnCtx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret < 0) {
        LOG_ERROR("rknn_query in_out_num failed: ret=" + std::to_string(ret));
        release();
        return false;
    }
    
    inputNum_ = io_num.n_input;
    outputNum_ = io_num.n_output;
    
    LOG_INFO("Model input_num=" + std::to_string(inputNum_) + 
             ", output_num=" + std::to_string(outputNum_));
    
    inputAttrs_.resize(inputNum_);
    for (int32_t i = 0; i < inputNum_; ++i) {
        inputAttrs_[i].index = i;
        ret = rknn_query(rknnCtx_, RKNN_QUERY_INPUT_ATTR, &inputAttrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOG_ERROR("rknn_query input_attr failed for index " + std::to_string(i));
            release();
            return false;
        }
    }
    
    inputWidth_ = inputAttrs_[0].dims[3];
    inputHeight_ = inputAttrs_[0].dims[2];
    inputChannel_ = inputAttrs_[0].dims[1];
    
    LOG_INFO("Model input (NCHW): " + std::to_string(inputChannel_) + "x" +
             std::to_string(inputHeight_) + "x" + std::to_string(inputWidth_));
    LOG_INFO("Input format: NCHW, type: " + std::to_string(inputAttrs_[0].type));
    
    outputAttrs_.resize(outputNum_);
    for (int32_t i = 0; i < outputNum_; ++i) {
        outputAttrs_[i].index = i;
        ret = rknn_query(rknnCtx_, RKNN_QUERY_OUTPUT_ATTR, &outputAttrs_[i], sizeof(rknn_tensor_attr));
        if (ret < 0) {
            LOG_ERROR("rknn_query output_attr failed for index " + std::to_string(i));
            release();
            return false;
        }
    }
    
    int32_t outputSize = outputAttrs_[0].n_elems;
    LOG_INFO("Model output: n_elems=" + std::to_string(outputSize) +
             ", dims: " + std::to_string(outputAttrs_[0].dims[0]) + "x" +
             std::to_string(outputAttrs_[0].dims[1]) + "x" +
             std::to_string(outputAttrs_[0].dims[2]) + "x" +
             std::to_string(outputAttrs_[0].dims[3]));
    
    inputData_.resize(getInputSize());
    outputData_.resize(outputSize);
    
#endif
    
    initialized_ = true;
    return true;
}

bool RknnAdapter::queryInputOutputInfo() {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }
    
#ifdef RK3566_PLATFORM
    
    for (int32_t i = 0; i < inputNum_; ++i) {
        LOG_INFO("Input " + std::to_string(i) + ": format=" + 
                 std::to_string(inputAttrs_[i].fmt) +
                 ", type=" + std::to_string(inputAttrs_[i].type) +
                 ", size=" + std::to_string(inputAttrs_[i].size) +
                 ", dims=[" + std::to_string(inputAttrs_[i].dims[0]) + "," +
                 std::to_string(inputAttrs_[i].dims[1]) + "," +
                 std::to_string(inputAttrs_[i].dims[2]) + "," +
                 std::to_string(inputAttrs_[i].dims[3]) + "]");
    }
    
    for (int32_t i = 0; i < outputNum_; ++i) {
        LOG_INFO("Output " + std::to_string(i) + ": size=" + std::to_string(outputAttrs_[i].size) +
                 ", n_elems=" + std::to_string(outputAttrs_[i].n_elems) +
                 ", dims=[" + std::to_string(outputAttrs_[i].dims[0]) + "," +
                 std::to_string(outputAttrs_[i].dims[1]) + "," +
                 std::to_string(outputAttrs_[i].dims[2]) + "," +
                 std::to_string(outputAttrs_[i].dims[3]) + "]");
    }
    
#endif
    
    return true;
}

bool RknnAdapter::setInputBuffer(const uint8_t* data, int32_t size) {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }
    
    int32_t expectedSize = getInputSize();
    if (size != expectedSize) {
        LOG_ERROR("Input size mismatch: expected " + std::to_string(expectedSize) +
                  ", got " + std::to_string(size));
        return false;
    }
    
#ifdef RK3566_PLATFORM
    
    inputs_.resize(inputNum_);
    for (int32_t i = 0; i < inputNum_; ++i) {
        inputs_[i].index = i;
        inputs_[i].buf = static_cast<int32_t>(reinterpret_cast<uint64_t>(inputData_.data()));
        inputs_[i].pass_through = 0;
        inputs_[i].fmt = RKNN_TENSOR_FORMAT_NCHW;
        
        if (inputAttrs_[i].type == RKNN_TENSOR_TYPE_UINT8) {
            inputs_[i].type = RKNN_TENSOR_TYPE_UINT8;
        } else if (inputAttrs_[i].type == RKNN_TENSOR_TYPE_INT8) {
            inputs_[i].type = RKNN_TENSOR_TYPE_INT8;
        } else {
            inputs_[i].type = RKNN_TENSOR_TYPE_UINT8;
        }
    }
    
    std::memcpy(inputData_.data(), data, size);
    
    int32_t ret = rknn_inputs_set(rknnCtx_, inputNum_, inputs_.data());
    if (ret < 0) {
        LOG_ERROR("rknn_inputs_set failed: ret=" + std::to_string(ret));
        return false;
    }
    
    LOG_DEBUG("Input buffer set: " + std::to_string(size) + " bytes (NCHW format)");
    
#else
    
    LOG_DEBUG("Input buffer set (stub mode): " + std::to_string(size) + " bytes");
    
#endif
    
    return true;
}

bool RknnAdapter::run() {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }
    
#ifdef RK3566_PLATFORM
    
    outputs_.resize(outputNum_);
    for (int32_t i = 0; i < outputNum_; ++i) {
        outputs_[i].index = i;
        outputs_[i].want_float = 1;
        outputs_[i].buf = static_cast<int32_t>(reinterpret_cast<uint64_t>(outputData_.data()));
        outputs_[i].fmt = RKNN_TENSOR_FORMAT_NCHW;
        outputs_[i].type = RKNN_TENSOR_TYPE_FLOAT32;
    }
    
    int32_t ret = rknn_run(rknnCtx_, nullptr);
    if (ret < 0) {
        LOG_ERROR("rknn_run failed: ret=" + std::to_string(ret));
        return false;
    }
    
    ret = rknn_outputs_get(rknnCtx_, outputNum_, outputs_.data(), sizeof(rknn_output));
    if (ret < 0) {
        LOG_ERROR("rknn_outputs_get failed: ret=" + std::to_string(ret));
        return false;
    }
    
    LOG_DEBUG("RKNN inference completed successfully");
    
#else
    
    LOG_DEBUG("RKNN stub mode: inference skipped");
    
#endif
    
    return true;
}

bool RknnAdapter::getOutputBuffer(float* data, int32_t size) {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }
    
#ifdef RK3566_PLATFORM
    
    int32_t outputSize = outputAttrs_[0].n_elems;
    if (size != outputSize) {
        LOG_ERROR("Output size mismatch: expected " + std::to_string(outputSize) +
                  ", got " + std::to_string(size));
        return false;
    }
    
    std::memcpy(data, outputData_.data(), size * sizeof(float));
    
    rknn_outputs_release(rknnCtx_, outputNum_, outputs_.data());
    
    LOG_DEBUG("Output buffer retrieved: " + std::to_string(size) + " floats");
    
#else
    
    LOG_DEBUG("RKNN stub mode: output buffer empty");
    
#endif
    
    return true;
}

int32_t RknnAdapter::getOutputSize(int32_t index) const {
    if (index >= outputNum_) {
        return 0;
    }
    
#ifdef RK3566_PLATFORM
    return outputAttrs_[index].n_elems;
#else
    return 0;
#endif
}

void RknnAdapter::release() {
    if (!initialized_) {
        return;
    }
    
#ifdef RK3566_PLATFORM
    
    if (rknnCtx_) {
        rknn_destroy(rknnCtx_);
        rknnCtx_ = nullptr;
        LOG_INFO("RKNN context destroyed");
    }
    
#endif
    
    inputData_.clear();
    outputData_.clear();
    inputs_.clear();
    outputs_.clear();
    
    initialized_ = false;
    LOG_INFO("RKNN adapter released");
}

bool RknnAdapter::loadModelFile(const std::string& modelPath, std::vector<uint8_t>& modelData) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open model file: " + modelPath);
        return false;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    modelData.resize(static_cast<size_t>(size));
    
    if (!file.read(reinterpret_cast<char*>(modelData.data()), size)) {
        LOG_ERROR("Failed to read model file: " + modelPath);
        return false;
    }
    
    file.close();
    
    LOG_INFO("Model file loaded: " + modelPath + ", size=" + std::to_string(size));
    
    return true;
}

}