#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace diff_det {

class IFrameDecoder {
public:
    virtual ~IFrameDecoder() = default;
    
    virtual bool Init(const std::string& rtspUrl) = 0;
    virtual bool DecodeNext(cv::Mat& frame, int& frameId, int64_t& timestamp) = 0;
    virtual void reset() = 0;
    virtual bool isOpened() = 0;
};

class FrameDecoder : public IFrameDecoder {
public:
    FrameDecoder();
    ~FrameDecoder();
    
    bool Init(const std::string& rtspUrl) override;
    bool DecodeNext(cv::Mat& frame, int& frameId, int64_t& timestamp) override;
    void reset() override;
    bool isOpened() override;
    
    double GetFps() const;
    int GetWidth() const;
    int GetHeight() const;
    
private:
    cv::VideoCapture cap_;
    std::string url_;
    bool opened_;
    int frameId_;
    int64_t timestamp_;
    double fps_;
    int width_;
    int height_;
};

}