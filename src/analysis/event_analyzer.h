#pragma once

#include "common/types.h"
#include "common/config.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <functional>
#include <deque>

namespace diff_det {

enum class EventAnalysisMode {
    kImage,
    kVideo
};

class IEventAnalyzer {
public:
    virtual ~IEventAnalyzer() = default;
    
    virtual void AnalyzeImage(const cv::Mat& frame, 
                              const std::vector<BoundingBox>& boxes) = 0;
    virtual void AnalyzeVideo(const std::vector<cv::Mat>& frames, 
                              const std::vector<BoundingBox>& boxes) = 0;
};

using EventCallback = std::function<void(const cv::Mat& frame, 
                                          const std::vector<BoundingBox>& boxes,
                                          int frameId, int64_t timestamp)>;

class EventAnalyzer : public IEventAnalyzer {
public:
    EventAnalyzer(const EventAnalysisConfig& config);
    ~EventAnalyzer();
    
    void AnalyzeImage(const cv::Mat& frame, 
                      const std::vector<BoundingBox>& boxes) override;
    void AnalyzeVideo(const std::vector<cv::Mat>& frames, 
                      const std::vector<BoundingBox>& boxes) override;
    
    void setEventCallback(EventCallback callback);
    void setVideoBuffer(std::deque<cv::Mat>* buffer);
    
    int GetEventCount() const { return eventCount_; }

private:
    bool ValidateBoxes(const std::vector<BoundingBox>& boxes);
    bool ValidateInput(const cv::Mat& frame, const std::vector<BoundingBox>& boxes);
    bool ValidateInput(const std::vector<cv::Mat>& frames, const std::vector<BoundingBox>& boxes);
    std::string generateEventId();

    EventAnalysisMode mode_;
    int videoDurationSec_;
    std::string webhookUrl_;
    bool webhookEnabled_;
    bool saveImg_;
    bool withBox_;
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
    void Clear();
    int Size() const { return static_cast<int>(frames_.size()); }
    
private:
    std::deque<cv::Mat> frames_;
    std::deque<int> frameIds_;
    std::deque<int64_t> timestamps_;
    int maxSize_;
};

}