#pragma once

#include <opencv2/opencv.hpp>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace diff_det {

struct FrameWithMeta {
    cv::Mat frame;
    int32_t frameId;
    int64_t timestamp;
    std::vector<float> detectionTimes;
    
    FrameWithMeta() : frameId(-1), timestamp(0) {}
    
    FrameWithMeta(const cv::Mat& f, int32_t id, int64_t ts)
        : frame(f.clone()), frameId(id), timestamp(ts) {}
};

class FrameQueue {
public:
    FrameQueue(int32_t maxSize = 30);
    
    void Push(const cv::Mat& frame, int32_t frameId, int64_t timestamp);
    bool Pop(cv::Mat& frame, int32_t& frameId, int64_t& timestamp);
    bool Pop(FrameWithMeta& frameMeta);
    
    int32_t Size();
    bool Empty();
    void Clear();
    
    void SetMaxSize(int32_t maxSize);
    int32_t GetMaxSize();
    
    void WaitForFrame(int32_t timeoutMs = 1000);
    
private:
    std::deque<FrameWithMeta> queue_;
    int32_t maxSize_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> stopped_;
};

class VideoFrameBuffer {
public:
    VideoFrameBuffer(int32_t maxFrames = 150);
    
    void AddFrame(const cv::Mat& frame, int32_t frameId, int64_t timestamp);
    
    std::vector<cv::Mat> GetFrames(int32_t count);
    std::vector<cv::Mat> GetFramesByDuration(int32_t durationSec, double fps);
    std::vector<FrameWithMeta> GetFramesWithMeta(int32_t count);
    
    void Clear();
    int32_t Size();
    bool Empty();
    
    int32_t GetOldestFrameId();
    int32_t GetNewestFrameId();
    int64_t GetOldestTimestamp();
    int64_t GetNewestTimestamp();
    
    void SetMaxSize(int32_t maxFrames);
    
private:
    std::deque<FrameWithMeta> buffer_;
    int32_t maxFrames_;
    std::mutex mutex_;
};

}