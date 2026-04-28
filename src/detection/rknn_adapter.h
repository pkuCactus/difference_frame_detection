#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace diff_det {

#include <rknn_api.h>

#define RKNN_SDK_VERSION "RKNN-SDK-for-RK3566"

class RknnAdapter {
public:
    RknnAdapter();
    ~RknnAdapter();
    
    bool Init(const std::string& modelPath);
    bool QueryInputOutputInfo();
    bool SetInputBuffer(const uint8_t* data, int32_t Size);
    bool Run();
    bool GetOutputBuffer(int32_t index, float* data, int32_t Size);
    void ReleaseOutputs();

    int32_t GetInputWidth() const { return inputWidth_; }
    int32_t GetInputHeight() const { return inputHeight_; }
    int32_t GetInputChannel() const { return inputChannel_; }
    int32_t GetInputSize() const { return inputWidth_ * inputHeight_ * inputChannel_; }
    int32_t GetInputType() const;
    int32_t GetInputFormat() const;
    
    int32_t GetInputNum() const { return inputNum_; }
    int32_t GetOutputNum() const { return outputNum_; }
    int32_t GetOutputSize(int32_t index) const;

    const std::vector<rknn_tensor_attr>& GetInputAttrs() const { return inputAttrs_; }
    const std::vector<rknn_tensor_attr>& GetOutputAttrs() const { return outputAttrs_; }

    void release();
    
    bool IsInitialized() const { return initialized_; }
    
    static bool CheckPlatform();
    
private:
    bool LoadModelFile(const std::string& modelPath, std::vector<uint8_t>& modelData);
    
    rknn_context rknnCtx_;
    bool initialized_;
    
    int32_t inputWidth_;
    int32_t inputHeight_;
    int32_t inputChannel_;
    int32_t inputNum_;
    
    int32_t outputNum_;
    std::vector<rknn_tensor_attr> inputAttrs_;
    std::vector<rknn_tensor_attr> outputAttrs_;
    
    std::vector<rknn_input> inputs_;
    std::vector<rknn_output> outputs_;
    std::vector<int32_t> outputOffsets_;

    std::vector<uint8_t> inputData_;
    std::vector<float> outputData_;

    // INT8/UINT8 量化输出 buffer 和参数，FP16/FP32 时不使用
    std::vector<std::vector<int8_t>> outputQuantizedBuffers_;
    std::vector<float> outputScales_;
    std::vector<int32_t> outputZps_;
};

}