#include "detection/rknn_adapter.h"
#include "common/logger.h"
#include <fstream>
#include <cstring>
#include <string>

namespace diff_det {

RknnAdapter::RknnAdapter()
    : rknnCtx_(0)
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

bool RknnAdapter::CheckPlatform() {
    LOG_INFO("Running on RK3566 platform with RKNN SDK");
    return true;
}

bool RknnAdapter::Init(const std::string& modelPath) {
    LOG_INFO("Initializing RKNN adapter with model: " + modelPath);

    std::vector<uint8_t> modelData;
    if (!LoadModelFile(modelPath, modelData)) {
        LOG_ERROR("Failed to load model file: " + modelPath);
        return false;
    }

    LOG_INFO("Model loaded: Size=" + std::to_string(modelData.size()) + " bytes");

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

    if (inputAttrs_[0].fmt == RKNN_TENSOR_NCHW) {
        inputChannel_ = inputAttrs_[0].dims[1];
        inputHeight_ = inputAttrs_[0].dims[2];
        inputWidth_ = inputAttrs_[0].dims[3];
    } else {
        inputHeight_ = inputAttrs_[0].dims[1];
        inputWidth_ = inputAttrs_[0].dims[2];
        inputChannel_ = inputAttrs_[0].dims[3];
    }

    std::string fmtStr = (inputAttrs_[0].fmt == RKNN_TENSOR_NCHW) ? "NCHW" : "NHWC";
    std::string typeStr;
    switch (inputAttrs_[0].type) {
        case RKNN_TENSOR_FLOAT32: typeStr = "FP32"; break;
        case RKNN_TENSOR_FLOAT16: typeStr = "FP16"; break;
        case RKNN_TENSOR_INT8: typeStr = "INT8"; break;
        case RKNN_TENSOR_UINT8: typeStr = "UINT8"; break;
        default: typeStr = "UNKNOWN(" + std::to_string(inputAttrs_[0].type) + ")"; break;
    }

    LOG_INFO("Model input: " + fmtStr + " " + typeStr + " " +
             std::to_string(inputChannel_) + "x" +
             std::to_string(inputHeight_) + "x" + std::to_string(inputWidth_));

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

    int32_t totalOutputSize = 0;
    outputScales_.resize(outputNum_);
    outputZps_.resize(outputNum_);
    for (int32_t i = 0; i < outputNum_; ++i) {
        totalOutputSize += outputAttrs_[i].n_elems;
        outputScales_[i] = outputAttrs_[i].scale;
        outputZps_[i] = outputAttrs_[i].zp;
        LOG_INFO("Model output[" + std::to_string(i) + "]: n_elems=" +
                 std::to_string(outputAttrs_[i].n_elems) +
                 ", type=" + std::to_string(outputAttrs_[i].type) +
                 ", scale=" + std::to_string(outputAttrs_[i].scale) +
                 ", zp=" + std::to_string(outputAttrs_[i].zp) +
                 ", dims=[" + std::to_string(outputAttrs_[i].dims[0]) + "," +
                 std::to_string(outputAttrs_[i].dims[1]) + "," +
                 std::to_string(outputAttrs_[i].dims[2]) + "," +
                 std::to_string(outputAttrs_[i].dims[3]) + "]");
    }

    inputData_.resize(GetInputSize());
    outputData_.resize(totalOutputSize);

    inputs_.resize(inputNum_);
    for (int32_t i = 0; i < inputNum_; ++i) {
        int32_t elemSize = 1;
        if (inputAttrs_[i].type == RKNN_TENSOR_FLOAT32) elemSize = 4;
        else if (inputAttrs_[i].type == RKNN_TENSOR_FLOAT16) elemSize = 2;

        inputData_.resize(inputAttrs_[i].n_elems * elemSize);

        inputs_[i].index = i;
        inputs_[i].buf = inputData_.data();
        inputs_[i].size = inputData_.size();
        inputs_[i].pass_through = 0;
        inputs_[i].fmt = inputAttrs_[i].fmt;
        inputs_[i].type = inputAttrs_[i].type;

        std::string inFmtStr = (inputAttrs_[i].fmt == RKNN_TENSOR_NCHW) ? "NCHW" : "NHWC";
        LOG_INFO("Input[" + std::to_string(i) + "]: fmt=" + inFmtStr +
                 ", type=" + std::to_string(inputAttrs_[i].type) +
                 ", bytes=" + std::to_string(inputs_[i].size));
    }

    outputs_.resize(outputNum_);
    float* outputPtr = outputData_.data();
    outputOffsets_.resize(outputNum_);
    outputQuantizedBuffers_.resize(outputNum_);
    for (int32_t i = 0; i < outputNum_; ++i) {
        bool isQuantized = (outputAttrs_[i].type == RKNN_TENSOR_INT8 ||
                            outputAttrs_[i].type == RKNN_TENSOR_UINT8);

        outputs_[i].index = i;
        outputs_[i].is_prealloc = 1;

        if (isQuantized) {
            // INT8/UINT8 模型：want_float=0，获取原始量化数据后手动反量化
            outputs_[i].want_float = 0;
            outputQuantizedBuffers_[i].resize(outputAttrs_[i].n_elems);
            outputs_[i].buf = outputQuantizedBuffers_[i].data();
            outputs_[i].size = outputAttrs_[i].n_elems * sizeof(int8_t);
            LOG_INFO("Output[" + std::to_string(i) +
                     "]: quantized mode, buf_size=" + std::to_string(outputs_[i].size) + " bytes");
        } else {
            // FP32/FP16 模型：want_float=1，RKNN 自动转为 float32
            outputs_[i].want_float = 1;
            outputs_[i].buf = outputPtr;
            outputs_[i].size = outputAttrs_[i].n_elems * sizeof(float);
            outputOffsets_[i] = static_cast<int32_t>(outputPtr - outputData_.data());
            outputPtr += outputAttrs_[i].n_elems;
            LOG_INFO("Output[" + std::to_string(i) +
                     "]: float mode, buf_size=" + std::to_string(outputs_[i].size) + " bytes");
        }
    }

    initialized_ = true;
    return true;
}

bool RknnAdapter::QueryInputOutputInfo() {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }

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

    return true;
}

bool RknnAdapter::SetInputBuffer(const uint8_t* data, int32_t Size) {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }

    if (Size <= 0 || data == nullptr) {
        LOG_ERROR("Invalid input buffer: size=" + std::to_string(Size));
        return false;
    }

    int32_t expectedBytes = static_cast<int32_t>(inputData_.size());
    if (Size != expectedBytes) {
        LOG_ERROR("Input size mismatch: expected " + std::to_string(expectedBytes) +
                  " bytes, got " + std::to_string(Size) + " bytes");
        return false;
    }

    std::memcpy(inputData_.data(), data, Size);

    int32_t ret = rknn_inputs_set(rknnCtx_, inputNum_, inputs_.data());
    if (ret < 0) {
        LOG_ERROR("rknn_inputs_set failed: ret=" + std::to_string(ret));
        return false;
    }

    LOG_DEBUG("Input buffer set: " + std::to_string(Size) + " bytes");

    return true;
}

bool RknnAdapter::Run() {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }

    int32_t ret = rknn_run(rknnCtx_, nullptr);
    if (ret < 0) {
        LOG_ERROR("rknn_run failed: ret=" + std::to_string(ret));
        return false;
    }

    ret = rknn_outputs_get(rknnCtx_, outputNum_, outputs_.data(), nullptr);
    if (ret < 0) {
        LOG_ERROR("rknn_outputs_get failed: ret=" + std::to_string(ret));
        return false;
    }

    LOG_DEBUG("RKNN inference completed successfully");

    return true;
}

bool RknnAdapter::GetOutputBuffer(int32_t index, float* data, int32_t Size) {
    if (!initialized_) {
        LOG_ERROR("RKNN adapter not initialized");
        return false;
    }

    if (index >= outputNum_) {
        LOG_ERROR("Output index out of range: " + std::to_string(index));
        return false;
    }

    int32_t outputSize = outputAttrs_[index].n_elems;
    if (Size != outputSize) {
        LOG_ERROR("Output Size mismatch: expected " + std::to_string(outputSize) +
                  ", got " + std::to_string(Size));
        return false;
    }

    bool isQuantized = (outputAttrs_[index].type == RKNN_TENSOR_INT8 ||
                        outputAttrs_[index].type == RKNN_TENSOR_UINT8);

    if (isQuantized) {
        // 手动反量化：float = (qnt - zp) * scale
        const int8_t* qntData = outputQuantizedBuffers_[index].data();
        float scale = outputScales_[index];
        int32_t zp = outputZps_[index];
        for (int32_t i = 0; i < Size; ++i) {
            data[i] = (static_cast<float>(qntData[i]) - static_cast<float>(zp)) * scale;
        }
        LOG_DEBUG("Output buffer dequantized: index=" + std::to_string(index) +
                  ", size=" + std::to_string(Size) + " floats, scale=" + std::to_string(scale) +
                  ", zp=" + std::to_string(zp));
    } else {
        std::memcpy(data, outputData_.data() + outputOffsets_[index], Size * sizeof(float));
        LOG_DEBUG("Output buffer retrieved: index=" + std::to_string(index) +
                  ", size=" + std::to_string(Size) + " floats");
    }

    return true;
}

void RknnAdapter::ReleaseOutputs() {
    if (rknnCtx_ && !outputs_.empty()) {
        rknn_outputs_release(rknnCtx_, outputNum_, outputs_.data());
        LOG_DEBUG("RKNN outputs released");
    }
}

int32_t RknnAdapter::GetInputType() const {
    if (!inputAttrs_.empty()) {
        return inputAttrs_[0].type;
    }
    return RKNN_TENSOR_UINT8;
}

int32_t RknnAdapter::GetInputFormat() const {
    if (!inputAttrs_.empty()) {
        return inputAttrs_[0].fmt;
    }
    return RKNN_TENSOR_NHWC;
}

int32_t RknnAdapter::GetOutputSize(int32_t index) const {
    if (index >= outputNum_) {
        return 0;
    }
    return outputAttrs_[index].n_elems;
}

void RknnAdapter::release() {
    if (!initialized_) {
        return;
    }

    if (rknnCtx_) {
        rknn_destroy(rknnCtx_);
        rknnCtx_ = 0;
        LOG_INFO("RKNN context destroyed");
    }

    inputData_.clear();
    outputData_.clear();
    outputQuantizedBuffers_.clear();
    outputScales_.clear();
    outputZps_.clear();
    inputs_.clear();
    outputs_.clear();

    initialized_ = false;
    LOG_INFO("RKNN adapter released");
}

bool RknnAdapter::LoadModelFile(const std::string& modelPath, std::vector<uint8_t>& modelData) {
    std::ifstream file(modelPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open model file: " + modelPath);
        return false;
    }

    std::streamsize Size = file.tellg();
    file.seekg(0, std::ios::beg);

    modelData.resize(static_cast<size_t>(Size));

    if (!file.read(reinterpret_cast<char*>(modelData.data()), Size)) {
        LOG_ERROR("Failed to read model file: " + modelPath);
        return false;
    }

    file.close();

    LOG_INFO("Model file loaded: " + modelPath + ", Size=" + std::to_string(Size));

    return true;
}

}
