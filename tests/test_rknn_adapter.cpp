#include <gtest/gtest.h>
#include "detection/rknn_adapter.h"

using namespace diff_det;

TEST(RknnAdapterTest, PlatformCheck) {
    bool isRK3566 = RknnAdapter::checkPlatform();
    
    EXPECT_FALSE(isRK3566);
}

TEST(RknnAdapterTest, InitStubMode) {
    RknnAdapter adapter;
    
    EXPECT_FALSE(adapter.isInitialized());
    
    bool result = adapter.init("/tmp/nonexistent.rknn");
    EXPECT_TRUE(result);
    EXPECT_TRUE(adapter.isInitialized());
    
    adapter.release();
    EXPECT_FALSE(adapter.isInitialized());
}

TEST(RknnAdapterTest, InputSize) {
    RknnAdapter adapter;
    adapter.init("/tmp/test.rknn");
    
    int width = adapter.getInputWidth();
    int height = adapter.getInputHeight();
    int channel = adapter.getInputChannel();
    int size = adapter.getInputSize();
    
    EXPECT_EQ(width, 832);
    EXPECT_EQ(height, 448);
    EXPECT_EQ(channel, 3);
    EXPECT_EQ(size, width * height * channel);
    
    adapter.release();
}

TEST(RknnAdapterTest, SetInputBuffer) {
    RknnAdapter adapter;
    adapter.init("/tmp/test.rknn");
    
    int size = adapter.getInputSize();
    std::vector<uint8_t> buffer(size, 128);
    
    bool result = adapter.setInputBuffer(buffer.data(), size);
    EXPECT_TRUE(result);
    
    adapter.release();
}

TEST(RknnAdapterTest, RunInference) {
    RknnAdapter adapter;
    adapter.init("/tmp/test.rknn");
    
    int size = adapter.getInputSize();
    std::vector<uint8_t> buffer(size, 128);
    adapter.setInputBuffer(buffer.data(), size);
    
    bool result = adapter.run();
    EXPECT_TRUE(result);
    
    adapter.release();
}

TEST(RknnAdapterTest, GetOutput) {
    RknnAdapter adapter;
    adapter.init("/tmp/test.rknn");
    
    int inputSize = adapter.getInputSize();
    std::vector<uint8_t> inputBuffer(inputSize, 128);
    adapter.setInputBuffer(inputBuffer.data(), inputSize);
    adapter.run();
    
    int outputSize = adapter.getOutputSize(0);
    EXPECT_EQ(outputSize, 0);
    
    std::vector<float> outputBuffer(outputSize);
    bool result = adapter.getOutputBuffer(outputBuffer.data(), outputSize);
    EXPECT_TRUE(result);
    
    adapter.release();
}