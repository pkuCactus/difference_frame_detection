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
    
    virtual bool connect(const std::string& url) = 0;
    virtual void disconnect() = 0;
    virtual bool isConnected() = 0;
    virtual bool getFrame(cv::Mat& frame, int& frameId, int64_t& timestamp) = 0;
    virtual void setFrameCallback(FrameCallback callback) = 0;
    virtual bool reconnect() = 0;
};

class RtspClient : public IRtspClient {
public:
    RtspClient();
    ~RtspClient();
    
    bool connect(const std::string& url) override;
    void disconnect() override;
    bool isConnected() override;
    bool getFrame(cv::Mat& frame, int& frameId, int64_t& timestamp) override;
    void setFrameCallback(FrameCallback callback) override;
    bool reconnect() override;
    
    double getFps() const;
    int getWidth() const;
    int getHeight() const;
    
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