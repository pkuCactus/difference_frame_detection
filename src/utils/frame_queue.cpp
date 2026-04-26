#include "utils/frame_queue.h"
#include "common/logger.h"

namespace diff_det {

FrameQueue::FrameQueue(int32_t maxSize)
    : maxSize_(maxSize)
    , stopped_(false) {
    LOG_INFO("FrameQueue created with maxSize=" + std::to_string(maxSize));
}

void FrameQueue::Push(const cv::Mat& frame, int32_t frameId, int64_t timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    FrameWithMeta meta(frame, frameId, timestamp);
    
    queue_.push_back(meta);
    
    while (queue_.size() > static_cast<size_t>(maxSize_)) {
        LOG_WARN("FrameQueue overflow, dropping oldest frame: " + std::to_string(queue_.front().frameId));
        queue_.pop_front();
    }
    
    cv_.notify_one();
    
    LOG_DEBUG("FrameQueue pushed frame " + std::to_string(frameId) + 
              ", Size=" + std::to_string(queue_.size()));
}

bool FrameQueue::Pop(cv::Mat& frame, int32_t& frameId, int64_t& timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (queue_.empty()) {
        return false;
    }
    
    FrameWithMeta meta = queue_.front();
    queue_.pop_front();
    
    frame = meta.frame;
    frameId = meta.frameId;
    timestamp = meta.timestamp;
    
    return true;
}

bool FrameQueue::Pop(FrameWithMeta& frameMeta) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (queue_.empty()) {
        return false;
    }
    
    frameMeta = queue_.front();
    queue_.pop_front();
    
    return true;
}

int32_t FrameQueue::Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(queue_.size());
}

bool FrameQueue::Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void FrameQueue::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    LOG_INFO("FrameQueue cleared");
}

void FrameQueue::SetMaxSize(int32_t maxSize) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxSize_ = maxSize;
    
    while (queue_.size() > static_cast<size_t>(maxSize_)) {
        queue_.pop_front();
    }
}

int32_t FrameQueue::GetMaxSize() {
    return maxSize_;
}

void FrameQueue::WaitForFrame(int32_t timeoutMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (!queue_.empty()) {
        return;
    }
    
    cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs));
}

VideoFrameBuffer::VideoFrameBuffer(int32_t maxFrames)
    : maxFrames_(maxFrames) {
    LOG_INFO("VideoFrameBuffer created with maxFrames=" + std::to_string(maxFrames));
}

void VideoFrameBuffer::AddFrame(const cv::Mat& frame, int32_t frameId, int64_t timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    FrameWithMeta meta(frame, frameId, timestamp);
    buffer_.push_back(meta);
    
    while (buffer_.size() > static_cast<size_t>(maxFrames_)) {
        buffer_.pop_front();
    }
    
    LOG_DEBUG("VideoFrameBuffer added frame " + std::to_string(frameId) +
              ", Size=" + std::to_string(buffer_.size()));
}

std::vector<cv::Mat> VideoFrameBuffer::GetFrames(int32_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<cv::Mat> frames;
    
    int32_t actualCount = std::min(count, static_cast<int32_t>(buffer_.size()));
    
    auto it = buffer_.rbegin();
    for (int32_t i = 0; i < actualCount && it != buffer_.rend(); ++i, ++it) {
        frames.push_back(it->frame.clone());
    }
    
    std::reverse(frames.begin(), frames.end());
    
    LOG_DEBUG("VideoFrameBuffer retrieved " + std::to_string(frames.size()) + " frames");
    
    return frames;
}

std::vector<cv::Mat> VideoFrameBuffer::GetFramesByDuration(int32_t durationSec, double fps) {
    int32_t frameCount = static_cast<int32_t>(durationSec * fps);
    return GetFrames(frameCount);
}

std::vector<FrameWithMeta> VideoFrameBuffer::GetFramesWithMeta(int32_t count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<FrameWithMeta> frames;
    
    int32_t actualCount = std::min(count, static_cast<int32_t>(buffer_.size()));
    
    auto it = buffer_.rbegin();
    for (int32_t i = 0; i < actualCount && it != buffer_.rend(); ++i, ++it) {
        frames.push_back(*it);
    }
    
    std::reverse(frames.begin(), frames.end());
    
    return frames;
}

void VideoFrameBuffer::Clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    LOG_INFO("VideoFrameBuffer cleared");
}

int32_t VideoFrameBuffer::Size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int32_t>(buffer_.size());
}

bool VideoFrameBuffer::Empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

int32_t VideoFrameBuffer::GetOldestFrameId() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return -1;
    return buffer_.front().frameId;
}

int32_t VideoFrameBuffer::GetNewestFrameId() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return -1;
    return buffer_.back().frameId;
}

int64_t VideoFrameBuffer::GetOldestTimestamp() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return 0;
    return buffer_.front().timestamp;
}

int64_t VideoFrameBuffer::GetNewestTimestamp() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return 0;
    return buffer_.back().timestamp;
}

void VideoFrameBuffer::SetMaxSize(int32_t maxFrames) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxFrames_ = maxFrames;
    
    while (buffer_.size() > static_cast<size_t>(maxFrames_)) {
        buffer_.pop_front();
    }
}

}