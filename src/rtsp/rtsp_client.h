#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <functional>
#include <string>

namespace diff_det {

class IRtspClient {
public:
    using FrameCallback = std::function<void(const cv::Mat& frame, int frameId, int64_t timestamp)>;
    
    virtual ~IRtspClient() = default;
    
    virtual bool Connect(const std::string& url) = 0;
    virtual void Disconnect() = 0;
    virtual bool IsConnected() = 0;
    virtual bool GetFrame(cv::Mat& frame, int& frameId, int64_t& timestamp) = 0;
    virtual void SetFrameCallback(FrameCallback callback) = 0;
    virtual bool Reconnect() = 0;
};

class RtspClient : public IRtspClient {
public:
    RtspClient();
    ~RtspClient();
    
    bool Connect(const std::string& url) override;
    void Disconnect() override;
    bool IsConnected() override;
    bool GetFrame(cv::Mat& frame, int& frameId, int64_t& timestamp) override;
    void SetFrameCallback(FrameCallback callback) override;
    bool Reconnect() override;
    
    double GetFps() const;
    int GetWidth() const;
    int GetHeight() const;
    
private:
    std::string url_;
    bool connected_;
    int frameId_;
    int64_t timestamp_;
    int64_t startTimestamp_;
    FrameCallback callback_;
    
    cv::VideoCapture cap_;
    double fps_;
    int width_;
    int height_;
};

}