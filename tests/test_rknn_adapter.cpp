#include <gtest/gtest.h>
#include "detection/rknn_adapter.h"

using namespace diff_det;

TEST(RknnAdapterTest, PlatformCheck) {
    bool isRK3566 = RknnAdapter::CheckPlatform();
    
    EXPECT_FALSE(isRK3566);
}

TEST(RknnAdapterTest, InitStubMode) {
    RknnAdapter adapter;
    
    EXPECT_FALSE(adapter.IsInitialized());
    
    bool result = adapter.Init("/tmp/nonexistent.rknn");
    EXPECT_TRUE(result);
    EXPECT_TRUE(adapter.IsInitialized());
    
    adapter.release();
    EXPECT_FALSE(adapter.IsInitialized());
}

TEST(RknnAdapterTest, InputSize) {
    RknnAdapter adapter;
    adapter.Init("/tmp/test.rknn");
    
    int Width = adapter.GetInputWidth();
    int Height = adapter.GetInputHeight();
    int channel = adapter.GetInputChannel();
    int Size = adapter.GetInputSize();
    
    EXPECT_EQ(Width, 832);
    EXPECT_EQ(Height, 448);
    EXPECT_EQ(channel, 3);
    EXPECT_EQ(Size, Width * Height * channel);
    
    adapter.release();
}

TEST(RknnAdapterTest, SetInputBuffer) {
    RknnAdapter adapter;
    adapter.Init("/tmp/test.rknn");
    
    int Size = adapter.GetInputSize();
    std::vector<uint8_t> buffer(Size, 128);
    
    bool result = adapter.SetInputBuffer(buffer.data(), Size);
    EXPECT_TRUE(result);
    
    adapter.release();
}

TEST(RknnAdapterTest, RunInference) {
    RknnAdapter adapter;
    adapter.Init("/tmp/test.rknn");
    
    int Size = adapter.GetInputSize();
    std::vector<uint8_t> buffer(Size, 128);
    adapter.SetInputBuffer(buffer.data(), Size);
    
    bool result = adapter.Run();
    EXPECT_TRUE(result);
    
    adapter.release();
}

TEST(RknnAdapterTest, GetOutput) {
    RknnAdapter adapter;
    adapter.Init("/tmp/test.rknn");
    
    int inputSize = adapter.GetInputSize();
    std::vector<uint8_t> inputBuffer(inputSize, 128);
    adapter.SetInputBuffer(inputBuffer.data(), inputSize);
    adapter.Run();
    
    int outputSize = adapter.GetOutputSize(0);
    EXPECT_EQ(outputSize, 0);
    
    std::vector<float> outputBuffer(outputSize);
    bool result = adapter.GetOutputBuffer(outputBuffer.data(), outputSize);
    EXPECT_TRUE(result);
    
    adapter.release();
}