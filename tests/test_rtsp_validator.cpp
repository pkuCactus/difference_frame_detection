#include <gtest/gtest.h>
#include "rtsp/rtsp_validator.h"
#include "rtsp/rtsp_client.h"
#include <opencv2/opencv.hpp>

using namespace diff_det;

class MockRtspClient : public IRtspClient {
public:
    bool Connect(const std::string& url) override {
        connected_ = true;
        url_ = url;
        return connectResult_;
    }
    void Disconnect() override {
        connected_ = false;
    }
    bool IsConnected() override {
        return connected_;
    }
    bool GetFrame(cv::Mat& frame, int& frameId, int64_t& timestamp) override {
        if (!connected_ || !getFrameResult_) {
            return false;
        }
        frame = cv::Mat::zeros(height_, width_, CV_8UC3);
        frameId = frameId_++;
        timestamp = timestamp_++;
        if (callback_) {
            callback_(frame, frameId, timestamp);
        }
        return true;
    }
    void SetFrameCallback(FrameCallback callback) override {
        callback_ = callback;
    }
    bool Reconnect() override {
        return Connect(url_);
    }
    double GetFps() const override { return fps_; }
    int GetWidth() const override { return width_; }
    int GetHeight() const override { return height_; }

    void SetConnectResult(bool result) { connectResult_ = result; }
    void SetGetFrameResult(bool result) { getFrameResult_ = result; }
    void SetResolution(int w, int h) { width_ = w; height_ = h; }
    void SetFps(double fps) { fps_ = fps; }

private:
    bool connectResult_ = true;
    bool getFrameResult_ = true;
    bool connected_ = false;
    int frameId_ = 0;
    int64_t timestamp_ = 0;
    int width_ = 640;
    int height_ = 480;
    double fps_ = 30.0;
    std::string url_;
    FrameCallback callback_;
};

TEST(RtspValidatorTest, ConnectFailure) {
    MockRtspClient client;
    client.SetConnectResult(false);

    auto result = RtspValidator::Validate(&client, "rtsp://test", 1);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.framesReceived, 0);
    EXPECT_FALSE(result.errorMessage.empty());
}

TEST(RtspValidatorTest, ReceiveFramesSuccessfully) {
    MockRtspClient client;
    client.SetConnectResult(true);
    client.SetGetFrameResult(true);
    client.SetResolution(1920, 1080);

    auto result = RtspValidator::Validate(&client, "rtsp://test", 0);

    EXPECT_TRUE(result.success);
    EXPECT_GT(result.framesReceived, 0);
    EXPECT_EQ(result.width, 1920);
    EXPECT_EQ(result.height, 1080);
}

TEST(RtspValidatorTest, GetFrameFailure) {
    MockRtspClient client;
    client.SetConnectResult(true);
    client.SetGetFrameResult(false);

    auto result = RtspValidator::Validate(&client, "rtsp://test", 0);

    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.framesReceived, 0);
}

TEST(RtspValidatorTest, ResultStructDefaults) {
    RtspValidationResult result;
    EXPECT_FALSE(result.success);
    EXPECT_EQ(result.framesReceived, 0);
    EXPECT_EQ(result.fps, 0.0);
    EXPECT_EQ(result.width, 0);
    EXPECT_EQ(result.height, 0);
}
