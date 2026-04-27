#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace diff_det {

#ifdef RK3566_PLATFORM

#include <rknn_api.h>

#define RKNN_SDK_VERSION "RKNN-SDK-for-RK3566"

#else

#define RKNN_SDK_VERSION "Stub-Mode"

typedef uint64_t rknn_context;

enum rknn_tensor_format {
    RKNN_TENSOR_FORMAT_AUTO = 0,
    RKNN_TENSOR_FORMAT_NCHW = 1,
    RKNN_TENSOR_FORMAT_NHWC = 2,
};

enum rknn_tensor_type {
    RKNN_TENSOR_TYPE_AUTO = 0,
    RKNN_TENSOR_TYPE_FLOAT32 = 1,
    RKNN_TENSOR_TYPE_FLOAT16 = 2,
    RKNN_TENSOR_TYPE_INT8 = 3,
    RKNN_TENSOR_TYPE_UINT8 = 4,
};

enum rknn_tensor_qnt_type {
    RKNN_TENSOR_QNT_TYPE_NONE = 0,
    RKNN_TENSOR_QNT_TYPE_AFFINE = 1,
};

struct rknn_tensor_attr {
    int32_t index;
    int32_t n_dims;
    int32_t dims[4];
    int32_t n_elems;
    int32_t Size;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
    rknn_tensor_qnt_type qnt_type;
    int8_t fl;
    int32_t zp;
    float scale;
    char name[16];
};

struct rknn_input {
    int32_t index;
    int32_t buf;
    int32_t pass_through;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
};

struct rknn_output {
    int32_t index;
    int32_t want_float;
    int32_t buf;
    rknn_tensor_format fmt;
    rknn_tensor_type type;
};

#endif

class RknnAdapter {
public:
    RknnAdapter();
    ~RknnAdapter();
    
    bool Init(const std::string& modelPath);
    bool QueryInputOutputInfo();
    bool SetInputBuffer(const uint8_t* data, int32_t Size);
    bool Run();
    bool GetOutputBuffer(float* data, int32_t Size);
    
    int32_t GetInputWidth() const { return inputWidth_; }
    int32_t GetInputHeight() const { return inputHeight_; }
    int32_t GetInputChannel() const { return inputChannel_; }
    int32_t GetInputSize() const { return inputWidth_ * inputHeight_ * inputChannel_; }
    
    int32_t GetOutputNum() const { return outputNum_; }
    int32_t GetOutputSize(int32_t index) const;
    
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
    
    std::vector<uint8_t> inputData_;
    std::vector<float> outputData_;
};

}