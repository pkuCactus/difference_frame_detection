#pragma once

#include "common/types.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <deque>

namespace diff_det {

class IEventAnalyzer {
public:
    virtual ~IEventAnalyzer() = default;
    
    virtual void analyzeImage(const cv::Mat& frame, 
                              const std::vector<BoundingBox>& boxes) = 0;
    virtual void analyzeVideo(const std::vector<cv::Mat>& frames, 
                              const std::vector<BoundingBox>& boxes) = 0;
};

using EventCallback = std::function<void(const cv::Mat& frame, 
                                          const std::vector<BoundingBox>& boxes,
                                          int frameId, int64_t timestamp)>;

class EventAnalyzer : public IEventAnalyzer {
public:
    EventAnalyzer(const EventAnalysisConfig& config);
    ~EventAnalyzer();
    
    void analyzeImage(const cv::Mat& frame, 
                      const std::vector<BoundingBox>& boxes) override;
    void analyzeVideo(const std::vector<cv::Mat>& frames, 
                      const std::vector<BoundingBox>& boxes) override;
    
    void setEventCallback(EventCallback callback);
    void setVideoBuffer(std::deque<cv::Mat>* buffer);
    
    int getEventCount() const { return eventCount_; }
    
private:
    void drawBoxes(cv::Mat& frame, const std::vector<BoundingBox>& boxes);
    std::string generateEventId();
    
    std::string mode_;
    int videoDurationSec_;
    EventCallback callback_;
    std::deque<cv::Mat>* videoBuffer_;
    
    int eventCount_;
    int lastEventId_;
};

class VideoBuffer {
public:
    VideoBuffer(int maxSize = 150);
    
    void addFrame(const cv::Mat& frame, int frameId, int64_t timestamp);
    std::vector<cv::Mat> getFrames(int count);
    std::vector<cv::Mat> getFramesByDuration(int durationSec, double fps);
    void clear();
    int size() const { return static_cast<int>(frames_.size()); }
    
private:
    std::deque<cv::Mat> frames_;
    std::deque<int> frameIds_;
    std::deque<int64_t> timestamps_;
    int maxSize_;
};

}