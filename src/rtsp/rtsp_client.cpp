#include "rtsp/rtsp_client.h"
#include "common/logger.h"
#include <chrono>
#include <thread>

namespace diff_det {

RtspClient::RtspClient() 
    : connected_(false)
    , frameId_(0)
    , timestamp_(0)
    , startTimestamp_(0)
    , fps_(30.0)
    , width_(0)
    , height_(0) {
}

RtspClient::~RtspClient() {
    disconnect();
}

bool RtspClient::connect(const std::string& url) {
    url_ = url;
    LOG_INFO("Connecting to RTSP: " + url);
    
    try {
        cap_.open(url);
        
        if (!cap_.isOpened()) {
            LOG_ERROR("Failed to open RTSP stream: " + url);
            connected_ = false;
            return false;
        }
        
        fps_ = cap_.get(cv::CAP_PROP_FPS);
        if (fps_ <= 0 || fps_ > 100) {
            fps_ = 30.0;
        }
        
        width_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_WIDTH));
        height_ = static_cast<int>(cap_.get(cv::CAP_PROP_FRAME_HEIGHT));
        
        frameId_ = 0;
        auto now = std::chrono::system_clock::now();
        startTimestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()).count();
        timestamp_ = startTimestamp_;
        
        connected_ = true;
        
        LOG_INFO("RTSP connected successfully: fps=" + std::to_string(fps_) +
                 ", width=" + std::to_string(width_) +
                 ", height=" + std::to_string(height_));
        
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception while connecting to RTSP: " + std::string(e.what()));
        connected_ = false;
        return false;
    }
}

void RtspClient::disconnect() {
    if (cap_.isOpened()) {
        cap_.release();
    }
    connected_ = false;
    LOG_INFO("Disconnected from RTSP");
}

bool RtspClient::isConnected() {
    return connected_ && cap_.isOpened();
}

bool RtspClient::getFrame(cv::Mat& frame, int& frameId, int64_t& timestamp) {
    if (!isConnected()) {
        LOG_WARN("RTSP not connected");
        return false;
    }
    
    try {
        if (!cap_.read(frame)) {
            LOG_WARN("Failed to read frame from RTSP, stream may have ended");
            connected_ = false;
            return false;
        }
        
        if (frame.empty()) {
            LOG_WARN("Empty frame received");
            return false;
        }
        
        frameId = frameId_;
        timestamp = timestamp_;
        
        frameId_++;
        
        int64_t frameInterval = static_cast<int64_t>(1000.0 / fps_);
        timestamp_ += frameInterval;
        
        if (callback_) {
            callback_(frame, frameId, timestamp);
        }
        
        return true;
        
    } catch (const cv::Exception& e) {
        LOG_ERROR("OpenCV exception while reading frame: " + std::string(e.what()));
        connected_ = false;
        return false;
    }
}

void RtspClient::setFrameCallback(FrameCallback callback) {
    callback_ = callback;
}

bool RtspClient::reconnect() {
    LOG_WARN("Attempting to reconnect to RTSP: " + url_);
    
    disconnect();
    
    int maxRetries = 5;
    int retryDelayMs = 1000;
    
    for (int i = 0; i < maxRetries; ++i) {
        LOG_INFO("Reconnect attempt " + std::to_string(i + 1) + "/" + std::to_string(maxRetries));
        
        if (connect(url_)) {
            LOG_INFO("Reconnected successfully");
            return true;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(retryDelayMs));
    }
    
    LOG_ERROR("Failed to reconnect after " + std::to_string(maxRetries) + " attempts");
    return false;
}

double RtspClient::getFps() const {
    return fps_;
}

int RtspClient::getWidth() const {
    return width_;
}

int RtspClient::getHeight() const {
    return height_;
}

}