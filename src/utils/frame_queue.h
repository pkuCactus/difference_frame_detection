#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace diff_det {

struct FrameWithMeta {
    cv::Mat frame;
    int frameId;
    int64_t timestamp;
    std::vector<float> detectionTimes;
    
    FrameWithMeta() : frameId(-1), timestamp(0) {}
    
    FrameWithMeta(const cv::Mat& f, int id, int64_t ts)
        : frame(f.clone()), frameId(id), timestamp(ts) {}
};

class FrameQueue {
public:
    FrameQueue(int maxSize = 30);
    
    void push(const cv::Mat& frame, int frameId, int64_t timestamp);
    bool pop(cv::Mat& frame, int& frameId, int64_t& timestamp);
    bool pop(FrameWithMeta& frameMeta);
    
    int size();
    bool empty();
    void clear();
    
    void setMaxSize(int size);
    int getMaxSize();
    
    void waitForFrame(int timeoutMs = 1000);
    
private:
    std::deque<FrameWithMeta> queue_;
    int maxSize_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_;
};

class VideoFrameBuffer {
public:
    VideoFrameBuffer(int maxFrames = 150);
    
    void addFrame(const cv::Mat& frame, int frameId, int64_t timestamp);
    
    std::vector<cv::Mat> getFrames(int count);
    std::vector<cv::Mat> getFramesByDuration(int durationSec, double fps);
    std::vector<FrameWithMeta> getFramesWithMeta(int count);
    
    void clear();
    int size();
    bool empty();
    
    int getOldestFrameId();
    int getNewestFrameId();
    int64_t getOldestTimestamp();
    int64_t getNewestTimestamp();
    
    void setMaxSize(int size);
    
private:
    std::deque<FrameWithMeta> buffer_;
    int maxFrames_;
    std::mutex mutex_;
};

}