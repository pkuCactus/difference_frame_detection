#include "analysis/event_analyzer.h"
#include "common/logger.h"
#include <chrono>
#include <sstream>
#include <iomanip>

namespace diff_det {

EventAnalyzer::EventAnalyzer(const EventAnalysisConfig& config)
    : mode_(config.mode)
    , videoDurationSec_(config.videoDurationSec)
    , videoBuffer_(nullptr)
    , eventCount_(0)
    , lastEventId_(0) {
    
    LOG_INFO("EventAnalyzer initialized: mode=" + mode_ +
             ", videoDuration=" + std::to_string(videoDurationSec_) + "s");
}

EventAnalyzer::~EventAnalyzer() {
}

bool EventAnalyzer::ValidateInput(const cv::Mat& frame,
                                    const std::vector<BoundingBox>& boxes) {
    if (frame.empty()) {
        LOG_ERROR("Event analysis received empty frame");
        return false;
    }
    if (boxes.empty()) {
        LOG_WARN("Event analysis received empty boxes, skipping");
        return false;
    }
    return true;
}

bool EventAnalyzer::ValidateInput(const std::vector<cv::Mat>& frames,
                                    const std::vector<BoundingBox>& boxes) {
    if (frames.empty()) {
        LOG_ERROR("Event analysis received empty frames");
        return false;
    }
    if (boxes.empty()) {
        LOG_WARN("Event analysis received empty boxes, skipping");
        return false;
    }
    return true;
}

void EventAnalyzer::AnalyzeImage(const cv::Mat& frame,
                                   const std::vector<BoundingBox>& boxes) {
    if (!ValidateInput(frame, boxes)) {
        return;
    }

    std::string eventId = generateEventId();
    eventCount_++;
    
    cv::Mat annotatedFrame = frame.clone();
    drawBoxes(annotatedFrame, boxes);
    
    LOG_INFO("Event analysis (image mode): eventId=" + eventId +
             ", boxes=" + std::to_string(boxes.size()) +
             ", totalEvents=" + std::to_string(eventCount_));
    
    if (callback_) {
        callback_(annotatedFrame, boxes, eventCount_, 
                  std::chrono::duration_cast<std::chrono::milliseconds>(
                      std::chrono::system_clock::now().time_since_epoch()).count());
    }
}

void EventAnalyzer::AnalyzeVideo(const std::vector<cv::Mat>& frames,
                                   const std::vector<BoundingBox>& boxes) {
    if (!ValidateInput(frames, boxes)) {
        return;
    }

    std::string eventId = generateEventId();
    eventCount_++;
    
    LOG_INFO("Event analysis (video mode): eventId=" + eventId +
             ", frames=" + std::to_string(frames.size()) +
             ", boxes=" + std::to_string(boxes.size()) +
             ", totalEvents=" + std::to_string(eventCount_));
    
    cv::Mat annotatedFrame = frames[0].clone();
    drawBoxes(annotatedFrame, boxes);
    
    if (callback_) {
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat annotated = frames[i].clone();
            if (i == 0) {
                drawBoxes(annotated, boxes);
            }
            callback_(annotated, boxes, static_cast<int>(i),
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count());
        }
    }
}

void EventAnalyzer::setEventCallback(EventCallback callback) {
    callback_ = callback;
}

void EventAnalyzer::setVideoBuffer(std::deque<cv::Mat>* buffer) {
    videoBuffer_ = buffer;
}

void EventAnalyzer::drawBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes) {
    for (const auto& box : boxes) {
        cv::Rect rect(static_cast<int>(box.x1), static_cast<int>(box.y1),
                      static_cast<int>(box.x2 - box.x1), static_cast<int>(box.y2 - box.y1));
        
        cv::rectangle(frame, rect, cv::Scalar(0, 255, 0), 2);
        
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << box.conf;
        std::string text = "person: " + oss.str();
        
        cv::Point textPos(static_cast<int>(box.x1), static_cast<int>(box.y1) - 5);
        cv::putText(frame, text, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(0, 255, 0), 1);
    }
}

std::string EventAnalyzer::generateEventId() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    std::ostringstream oss;
    oss << "event_" << std::setw(12) << std::setfill('0') << ms;
    
    return oss.str();
}

VideoBuffer::VideoBuffer(int maxSize)
    : maxSize_(maxSize) {
    LOG_INFO("VideoBuffer initialized: maxSize=" + std::to_string(maxSize_));
}

void VideoBuffer::addFrame(const cv::Mat& frame, int frameId, int64_t timestamp) {
    if (frame.empty()) {
        LOG_WARN("VideoBuffer received empty frame");
        return;
    }
    
    frames_.push_back(frame.clone());
    frameIds_.push_back(frameId);
    timestamps_.push_back(timestamp);
    
    while (frames_.size() > static_cast<size_t>(maxSize_)) {
        frames_.pop_front();
        frameIds_.pop_front();
        timestamps_.pop_front();
    }
    
    LOG_DEBUG("VideoBuffer: added frame " + std::to_string(frameId) +
              ", bufferSize=" + std::to_string(frames_.size()));
}

std::vector<cv::Mat> VideoBuffer::getFrames(int count) {
    std::vector<cv::Mat> result;
    
    int actualCount = std::min(count, static_cast<int>(frames_.size()));
    
    auto it = frames_.end();
    for (int i = 0; i < actualCount; ++i) {
        --it;
        result.push_back(*it);
    }
    
    std::reverse(result.begin(), result.end());
    
    LOG_DEBUG("VideoBuffer: retrieved " + std::to_string(result.size()) + " frames");
    
    return result;
}

std::vector<cv::Mat> VideoBuffer::getFramesByDuration(int durationSec, double fps) {
    int frameCount = static_cast<int>(durationSec * fps);
    return getFrames(frameCount);
}

void VideoBuffer::Clear() {
    frames_.clear();
    frameIds_.clear();
    timestamps_.clear();
    LOG_INFO("VideoBuffer cleared");
}

}