#include "decoder/frame_decoder.h"
#include "common/logger.h"
#include <chrono>

namespace diff_det {

FrameDecoder::FrameDecoder()
    : opened_(false)
    , frameId_(0)
    , timestamp_(0)
    , fps_(30.0)
    , width_(0)
    , height_(0) {
}

FrameDecoder::~FrameDecoder() {
    reset();
}

bool FrameDecoder::init(const std::string& rtspUrl) {
    url_ = rtspUrl;
    LOG_INFO("FrameDecoder initializing with RTSP URL: " + rtspUrl);
    
    try {
        cap_.open(rtspUrl, cv::CAP_ANY);
        
        if (!cap_.isOpened()) {
            LOG_ERROR("FrameDecoder failed to open RTSP stream");
            opened_ = false;
            return false;
        }
        
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        if (fps_ <= 0 || fps_ > 100) {
            fps_ = 30.0;
            LOG_WARN("Invalid FPS detected, using default 30");
        }
        
        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        frameId_ = 0;
        
        auto now = std::chrono::system_clock::now();
        timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        
        opened_ = true;
        
        LOG_INFO("FrameDecoder initialized: fps=" + std::to_string(fps_) +
                 ", resolution=" + std::to_string(width_) + "x" + std::to_string(height_));
        
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception in FrameDecoder: " + std::string(e.what()));
        opened_ = false;
        return false;
    }
}

bool FrameDecoder::decodeNext(cv::Mat& frame, int& frameId, int64_t& timestamp) {
    if (!opened_ || !cap_.isOpened()) {
        LOG_WARN("FrameDecoder not opened");
        return false;
    }
    
    try {
        if (!cap_.read(frame)) {
            LOG_WARN("FrameDecoder failed to read frame");
            opened_ = false;
            return false;
        }
        
        if (frame.empty()) {
            LOG_WARN("FrameDecoder received empty frame");
            return false;
        }
        
        frameId = frameId_;
        timestamp = timestamp_;
        
        frameId_++;
        
        int64_t frameInterval = static_cast<int64_t>(1000.0 / fps_);
        timestamp_ += frameInterval;
        
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception while decoding: " + std::string(e.what()));
        opened_ = false;
        return false;
    }
}

void FrameDecoder::reset() {
    if (cap_.isOpened()) {
        cap_.release();
    }
    opened_ = false;
    frameId_ = 0;
    timestamp_ = 0;
    LOG_INFO("FrameDecoder reset");
}

bool FrameDecoder::isOpened() {
    return opened_ && cap_.isOpened();
}

double FrameDecoder::getFps() const {
    return fps_;
}

int FrameDecoder::getWidth() const {
    return width_;
}

int FrameDecoder::getHeight() const {
    return height_;
}

}