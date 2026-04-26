#include "utils/frame_queue.h"
#include "common/logger.h"

namespace diff_det {

FrameQueue::FrameQueue(int maxSize)
    : maxSize_(maxSize)
    , stopped_(false) {
    LOG_INFO("FrameQueue created with maxSize=" + std::to_string(maxSize));
}

void FrameQueue::push(const cv::Mat& frame, int frameId, int64_t timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    FrameWithMeta meta(frame, frameId, timestamp);
    
    queue_.push_back(meta);
    
    while (queue_.size() > static_cast<size_t>(maxSize_)) {
        LOG_WARN("FrameQueue overflow, dropping oldest frame: " + std::to_string(queue_.front().frameId));
        queue_.pop_front();
    }
    
    cv_.notify_one();
    
    LOG_DEBUG("FrameQueue pushed frame " + std::to_string(frameId) + 
              ", size=" + std::to_string(queue_.size()));
}

bool FrameQueue::pop(cv::Mat& frame, int& frameId, int64_t& timestamp) {
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

bool FrameQueue::pop(FrameWithMeta& frameMeta) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (queue_.empty()) {
        return false;
    }
    
    frameMeta = queue_.front();
    queue_.pop_front();
    
    return true;
}

int FrameQueue::size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(queue_.size());
}

bool FrameQueue::empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.empty();
}

void FrameQueue::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    queue_.clear();
    LOG_INFO("FrameQueue cleared");
}

void FrameQueue::setMaxSize(int size) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxSize_ = size;
    
    while (queue_.size() > static_cast<size_t>(maxSize_)) {
        queue_.pop_front();
    }
}

int FrameQueue::getMaxSize() {
    return maxSize_;
}

void FrameQueue::waitForFrame(int timeoutMs) {
    std::unique_lock<std::mutex> lock(mutex_);
    
    if (!queue_.empty()) {
        return;
    }
    
    cv_.wait_for(lock, std::chrono::milliseconds(timeoutMs));
}

VideoFrameBuffer::VideoFrameBuffer(int maxFrames)
    : maxFrames_(maxFrames) {
    LOG_INFO("VideoFrameBuffer created with maxFrames=" + std::to_string(maxFrames));
}

void VideoFrameBuffer::addFrame(const cv::Mat& frame, int frameId, int64_t timestamp) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    FrameWithMeta meta(frame, frameId, timestamp);
    buffer_.push_back(meta);
    
    while (buffer_.size() > static_cast<size_t>(maxFrames_)) {
        buffer_.pop_front();
    }
    
    LOG_DEBUG("VideoFrameBuffer added frame " + std::to_string(frameId) +
              ", size=" + std::to_string(buffer_.size()));
}

std::vector<cv::Mat> VideoFrameBuffer::getFrames(int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<cv::Mat> frames;
    
    int actualCount = std::min(count, static_cast<int>(buffer_.size()));
    
    auto it = buffer_.rbegin();
    for (int i = 0; i < actualCount && it != buffer_.rend(); ++i, ++it) {
        frames.push_back(it->frame.clone());
    }
    
    std::reverse(frames.begin(), frames.end());
    
    LOG_DEBUG("VideoFrameBuffer retrieved " + std::to_string(frames.size()) + " frames");
    
    return frames;
}

std::vector<cv::Mat> VideoFrameBuffer::getFramesByDuration(int durationSec, double fps) {
    int frameCount = static_cast<int>(durationSec * fps);
    return getFrames(frameCount);
}

std::vector<FrameWithMeta> VideoFrameBuffer::getFramesWithMeta(int count) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<FrameWithMeta> frames;
    
    int actualCount = std::min(count, static_cast<int>(buffer_.size()));
    
    auto it = buffer_.rbegin();
    for (int i = 0; i < actualCount && it != buffer_.rend(); ++i, ++it) {
        frames.push_back(*it);
    }
    
    std::reverse(frames.begin(), frames.end());
    
    return frames;
}

void VideoFrameBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
    LOG_INFO("VideoFrameBuffer cleared");
}

int VideoFrameBuffer::size() {
    std::lock_guard<std::mutex> lock(mutex_);
    return static_cast<int>(buffer_.size());
}

bool VideoFrameBuffer::empty() {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

int VideoFrameBuffer::getOldestFrameId() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return -1;
    return buffer_.front().frameId;
}

int VideoFrameBuffer::getNewestFrameId() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return -1;
    return buffer_.back().frameId;
}

int64_t VideoFrameBuffer::getOldestTimestamp() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return 0;
    return buffer_.front().timestamp;
}

int64_t VideoFrameBuffer::getNewestTimestamp() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (buffer_.empty()) return 0;
    return buffer_.back().timestamp;
}

void VideoFrameBuffer::setMaxSize(int size) {
    std::lock_guard<std::mutex> lock(mutex_);
    maxFrames_ = size;
    
    while (buffer_.size() > static_cast<size_t>(maxFrames_)) {
        buffer_.pop_front();
    }
}

}