#pragma once

#include "common/types.h"
#include "common/config.h"
#include "common/logger.h"
#include "common/performance_stats.h"
#include "core/state.h"
#include "detection/rknn_detector.h"
#include "tracking/byte_tracker.h"
#include "rtsp/rtsp_client.h"
#include "camera/capability_checker.h"
#include "camera/detection_reader.h"
#include "analysis/frame_diff.h"
#include "analysis/event_analyzer.h"
#include "utils/frame_queue.h"

#include <opencv2/opencv.hpp>
#include <memory>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace diff_det {

struct PipelineStats {
    int totalFramesProcessed;
    int framesWithPerson;
    int framesSimilar;
    int framesDifferent;
    int eventsGenerated;
    int reconnectionAttempts;
    
    double avgDetectionTime;
    double avgFrameDiffTime;
    double avgTotalProcessTime;
    
    PipelineStats() : totalFramesProcessed(0), framesWithPerson(0), 
                      framesSimilar(0), framesDifferent(0), 
                      eventsGenerated(0), reconnectionAttempts(0),
                      avgDetectionTime(0), avgFrameDiffTime(0), 
                      avgTotalProcessTime(0) {}
    
    std::string ToString() const;
};

class StateMachine {
public:
    StateMachine(const Config& config);
    ~StateMachine();
    
    void Run();
    void Stop();
    void Pause();
    void Resume();
    
    State CurrentState() const;
    
    bool IsRunning() const { return running_; }
    bool isPaused() const { return paused_; }
    int GetFrameCount() const { return frameCounter_; }
    int GetEventCount() const;
    
    PipelineStats GetStats() const;
    PerformanceStats& getPerformanceStats() { return perfStats_; }
    
private:
    void Transition(State newState);
    
    void HandleInit();
    void HandleConnecting();
    void HandleCheckCapability();
    void HandleCameraDetectionMode();
    void HandleLocalDetectionMode();
    void HandleDifferenceAnalysis();
    void HandleEventAnalysis();
    void HandleUpdateRef();
    void HandleReconnecting();
    void HandleError();
    
    bool CheckConnection();
    void ProcessCameraDetection();
    void ProcessLocalDetection();
    bool CheckForPerson(const std::vector<BoundingBox>& boxes);
    bool MatchFrame(int frameId, int64_t timestamp, const CameraDetectionResult& result);
    
    void InitializeComponents();
    void CleanupComponents();
    
    void StoreFrameForVideo(const cv::Mat& frame, int frameId, int64_t timestamp);
    std::vector<FrameWithMeta> GetVideoFrames();
    
    void UpdateStats(bool hasPerson, bool IsSimilar);
    void LogFrameInfo(int frameId, int64_t timestamp, const std::vector<BoundingBox>& boxes);
    
    Config config_;
    State state_;
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
    
    cv::Mat currentFrame_;
    int currentFrameId_;
    int64_t currentTimestamp_;
    
    std::vector<BoundingBox> currentBoxes_;
    std::vector<Track> currentTracks_;
    
    bool useCameraDetection_;
    
    std::unique_ptr<RtspClient> rtspClient_;
    std::unique_ptr<CameraCapabilityChecker> capabilityChecker_;
    std::unique_ptr<CameraDetectionReader> detectionReader_;
    std::unique_ptr<RknnDetector> detector_;
    std::unique_ptr<ByteTracker> tracker_;
    std::unique_ptr<FrameDiffAnalyzer> frameDiffAnalyzer_;
    std::unique_ptr<EventAnalyzer> eventAnalyzer_;
    
    std::unique_ptr<VideoFrameBuffer> videoBuffer_;
    
    PerformanceStats perfStats_;
    PipelineStats pipelineStats_;
    
    int frameCounter_;
    bool hasDifference_;
    RefUpdateStrategy updateStrategy_;
    EventAnalysisMode eventMode_;

    std::mutex frameMutex_;
    std::condition_variable frameCv_;
    
    int64_t lastReconnectAttemptTime_;
    int reconnectAttempts_;
    
    int64_t pipelineStartTime_;
};

}