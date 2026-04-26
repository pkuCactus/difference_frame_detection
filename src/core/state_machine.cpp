#include "core/state_machine.h"
#include "common/logger.h"
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

namespace diff_det {

std::string PipelineStats::ToString() const {
    std::ostringstream oss;
    oss << "=== Pipeline Statistics ===\n";
    oss << "Frames processed: " << totalFramesProcessed << "\n";
    oss << "Frames with person: " << framesWithPerson << "\n";
    oss << "Frames similar: " << framesSimilar << "\n";
    oss << "Frames different: " << framesDifferent << "\n";
    oss << "Events generated: " << eventsGenerated << "\n";
    oss << "Reconnection attempts: " << reconnectionAttempts << "\n";
    oss << "Avg detection time: " << std::fixed << std::setprecision(2) << avgDetectionTime << "ms\n";
    oss << "Avg frame diff time: " << avgFrameDiffTime << "ms\n";
    oss << "Avg total Process time: " << avgTotalProcessTime << "ms\n";
    return oss.str();
}

namespace {

RefUpdateStrategy ParseUpdateStrategy(const std::string& strategy) {
    if (strategy == "newest") return RefUpdateStrategy::kNewest;
    return RefUpdateStrategy::kDefault;
}

EventAnalysisMode ParseEventMode(const std::string& mode) {
    if (mode == "video") return EventAnalysisMode::kVideo;
    return EventAnalysisMode::kImage;
}

} // namespace

StateMachine::StateMachine(const Config& config)
    : config_(config)
    , state_(State::INIT)
    , running_(false)
    , paused_(false)
    , currentFrameId_(-1)
    , currentTimestamp_(0)
    , useCameraDetection_(false)
    , frameCounter_(0)
    , hasDifference_(false)
    , updateStrategy_(ParseUpdateStrategy(config_.refFrame.updateStrategy))
    , eventMode_(ParseEventMode(config_.eventAnalysis.mode))
    , lastReconnectAttemptTime_(0)
    , reconnectAttempts_(0) {

    auto now = std::chrono::system_clock::now();
    pipelineStartTime_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

StateMachine::~StateMachine() {
    Stop();
    CleanupComponents();
    
    LOG_INFO(pipelineStats_.ToString());
}

void StateMachine::Run() {
    running_ = true;
    LOG_INFO("StateMachine started");
    
    while (running_) {
        if (paused_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        
        ScopedTimer timer(perfStats_, "frame_total");
        
        switch (state_) {
            case State::INIT:
                HandleInit();
                break;
            case State::CONNECTING:
                HandleConnecting();
                break;
            case State::CHECK_CAPABILITY:
                HandleCheckCapability();
                break;
            case State::CAMERA_DETECTION_MODE:
                HandleCameraDetectionMode();
                break;
            case State::LOCAL_DETECTION_MODE:
                HandleLocalDetectionMode();
                break;
            case State::DIFFERENCE_ANALYSIS:
                HandleDifferenceAnalysis();
                break;
            case State::EVENT_ANALYSIS:
                HandleEventAnalysis();
                break;
            case State::UPDATE_REF:
                HandleUpdateRef();
                break;
            case State::RECONNECTING:
                HandleReconnecting();
                break;
            case State::ERROR:
                HandleError();
                break;
            default:
                LOG_ERROR("Unknown state: " + std::to_string(static_cast<int>(state_)));
                Transition(State::ERROR);
                break;
        }
    }
    
    LOG_INFO("StateMachine stopped");
    LOG_INFO(pipelineStats_.ToString());
}

void StateMachine::Stop() {
    running_ = false;
    paused_ = false;
}

void StateMachine::Pause() {
    paused_ = true;
    LOG_INFO("StateMachine paused");
}

void StateMachine::Resume() {
    paused_ = false;
    LOG_INFO("StateMachine resumed");
}

State StateMachine::CurrentState() const {
    return state_;
}

int StateMachine::GetEventCount() const {
    if (eventAnalyzer_) {
        return eventAnalyzer_->GetEventCount();
    }
    return 0;
}

PipelineStats StateMachine::GetStats() const {
    return pipelineStats_;
}

void StateMachine::Transition(State newState) {
    LOG_INFO("State Transition: " + std::string(StateToString(state_)) + 
             " -> " + std::string(StateToString(newState)));
    state_ = newState;
    perfStats_.IncrementCounter("state_transitions");
}

void StateMachine::HandleInit() {
    LOG_INFO("Initializing StateMachine...");
    
    Logger::GetInstance().Init(config_.logging.filePath, config_.logging.level);
    
    InitializeComponents();
    
    int maxBufferFrames = static_cast<int>(config_.eventAnalysis.videoDurationSec * 30) + 10;
    videoBuffer_ = std::make_unique<VideoFrameBuffer>(maxBufferFrames);
    
    Transition(State::CONNECTING);
}

void StateMachine::HandleConnecting() {
    LOG_INFO("Connecting to RTSP: " + config_.rtsp.url);
    
    perfStats_.StartTimer("rtsp_connect");
    
    if (CheckConnection()) {
        perfStats_.EndTimer("rtsp_connect");
        
        double fps = rtspClient_->GetFps();
        int Width = rtspClient_->GetWidth();
        int Height = rtspClient_->GetHeight();
        
        LOG_INFO("RTSP connected: fps=" + std::to_string(fps) +
                 ", resolution=" + std::to_string(Width) + "x" + std::to_string(Height));
        
        Transition(State::CHECK_CAPABILITY);
    } else {
        perfStats_.EndTimer("rtsp_connect");
        LOG_WARN("RTSP connection failed");
        Transition(State::RECONNECTING);
    }
}

void StateMachine::HandleCheckCapability() {
    LOG_INFO("Checking camera detection capability");
    
    if (!capabilityChecker_) {
        LOG_ERROR("CapabilityChecker not initialized");
        Transition(State::ERROR);
        return;
    }
    
    perfStats_.StartTimer("capability_check");
    CapabilityResult result = capabilityChecker_->Check(config_.cameraDetection);
    perfStats_.EndTimer("capability_check");
    
    if (result.supported) {
        useCameraDetection_ = true;
        
        if (detectionReader_) {
            detectionReader_->Init(config_.cameraDetection);
        }
        
        LOG_INFO("camera_id=" + config_.cameraDetection.cameraId + 
                 ", detection_supported=true, mode=camera_detection");
        Transition(State::CAMERA_DETECTION_MODE);
    } else {
        useCameraDetection_ = false;
        LOG_INFO("camera_id=" + config_.cameraDetection.cameraId + 
                 ", detection_supported=false, reason=" + result.reason + 
                 ", mode=local_detection");
        Transition(State::LOCAL_DETECTION_MODE);
    }
}

void StateMachine::HandleCameraDetectionMode() {
    ProcessCameraDetection();
}

void StateMachine::HandleLocalDetectionMode() {
    ProcessLocalDetection();
}

void StateMachine::HandleDifferenceAnalysis() {
    if (!frameDiffAnalyzer_) {
        LOG_ERROR("FrameDiffAnalyzer not initialized");
        Transition(State::ERROR);
        return;
    }
    
    perfStats_.StartTimer("frame_diff_analysis");
    
    frameDiffAnalyzer_->SetBoxesForRoi(currentBoxes_);
    
    if (!frameDiffAnalyzer_->HasRef()) {
        perfStats_.EndTimer("frame_diff_analysis");
        
        LOG_INFO("No Ref frame, first frame must enter event analysis");
        hasDifference_ = true;
        Transition(State::EVENT_ANALYSIS);
        return;
    }
    
    cv::Mat refFrame = frameDiffAnalyzer_->GetRef();
    bool IsSimilar = frameDiffAnalyzer_->IsSimilar(currentFrame_, refFrame);
    
    perfStats_.EndTimer("frame_diff_analysis");
    
    UpdateStats(!currentBoxes_.empty(), IsSimilar);
    
    if (IsSimilar) {
        LOG_INFO("Frame similar to Ref, skip event analysis");
        hasDifference_ = false;
        pipelineStats_.framesSimilar++;
        
        if (updateStrategy_ == RefUpdateStrategy::kNewest && !currentBoxes_.empty()) {
            frameDiffAnalyzer_->UpdateRef(currentFrame_);
        }
        
        if (useCameraDetection_) {
            Transition(State::CAMERA_DETECTION_MODE);
        } else {
            Transition(State::LOCAL_DETECTION_MODE);
        }
    } else {
        LOG_INFO("Frame different from Ref, enter event analysis");
        hasDifference_ = true;
        pipelineStats_.framesDifferent++;
        Transition(State::EVENT_ANALYSIS);
    }
}

void StateMachine::HandleEventAnalysis() {
    if (!eventAnalyzer_) {
        LOG_ERROR("EventAnalyzer not initialized");
        Transition(State::ERROR);
        return;
    }
    
    perfStats_.StartTimer("event_analysis");
    
    LOG_INFO("Entering event analysis, frameId=" + std::to_string(currentFrameId_) +
             ", boxes=" + std::to_string(currentBoxes_.size()));
    
    if (eventMode_ == EventAnalysisMode::kImage) {
        eventAnalyzer_->AnalyzeImage(currentFrame_, currentBoxes_);
    } else {
        std::vector<FrameWithMeta> frames = GetVideoFrames();
        if (frames.empty()) {
            FrameWithMeta meta(currentFrame_, currentFrameId_, currentTimestamp_);
            frames.push_back(meta);
        }
        
        std::vector<cv::Mat> frameImages;
        for (auto& f : frames) {
            frameImages.push_back(f.frame);
        }
        eventAnalyzer_->AnalyzeVideo(frameImages, currentBoxes_);
    }
    
    perfStats_.EndTimer("event_analysis");
    
    pipelineStats_.eventsGenerated++;
    perfStats_.IncrementCounter("events_generated");
    
    Transition(State::UPDATE_REF);
}

void StateMachine::HandleUpdateRef() {
    if (!frameDiffAnalyzer_) {
        LOG_ERROR("FrameDiffAnalyzer not initialized");
        Transition(State::ERROR);
        return;
    }
    
    LOG_INFO("Updating Ref frame");
    
    if (updateStrategy_ == RefUpdateStrategy::kDefault || hasDifference_) {
        frameDiffAnalyzer_->UpdateRef(currentFrame_);
    }
    
    if (useCameraDetection_) {
        Transition(State::CAMERA_DETECTION_MODE);
    } else {
        Transition(State::LOCAL_DETECTION_MODE);
    }
}

void StateMachine::HandleReconnecting() {
    auto now = std::chrono::system_clock::now();
    int64_t currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    if (currentTime - lastReconnectAttemptTime_ < config_.rtsp.reconnectIntervalMs) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        return;
    }
    
    lastReconnectAttemptTime_ = currentTime;
    reconnectAttempts_++;
    pipelineStats_.reconnectionAttempts++;
    
    LOG_WARN("Attempting to Reconnect (" + std::to_string(reconnectAttempts_) + " attempts)...");
    
    if (rtspClient_ && rtspClient_->Reconnect()) {
        LOG_INFO("Reconnected successfully after " + std::to_string(reconnectAttempts_) + " attempts");
        reconnectAttempts_ = 0;
        Transition(State::CHECK_CAPABILITY);
    } else {
        LOG_ERROR("Reconnect failed, will retry after " + 
                  std::to_string(config_.rtsp.reconnectIntervalMs) + "ms");
    }
}

void StateMachine::HandleError() {
    LOG_ERROR("Error state encountered");
    LOG_INFO(pipelineStats_.ToString());
    CleanupComponents();
    running_ = false;
}

bool StateMachine::CheckConnection() {
    if (!rtspClient_) {
        LOG_ERROR("RtspClient not initialized");
        return false;
    }
    
    return rtspClient_->Connect(config_.rtsp.url);
}

void StateMachine::ProcessCameraDetection() {
    if (!rtspClient_ || !detectionReader_) {
        LOG_ERROR("Components not initialized for camera detection mode");
        Transition(State::ERROR);
        return;
    }
    
    if (!rtspClient_->IsConnected()) {
        LOG_WARN("RTSP disconnected in camera detection mode");
        Transition(State::RECONNECTING);
        return;
    }
    
    perfStats_.StartTimer("camera_detection_fetch");
    CameraDetectionResult result = detectionReader_->GetDetectionResult();
    perfStats_.EndTimer("camera_detection_fetch");
    
    if (!rtspClient_->GetFrame(currentFrame_, currentFrameId_, currentTimestamp_)) {
        LOG_WARN("Failed to get frame in camera detection mode");
        Transition(State::RECONNECTING);
        return;
    }
    
    perfStats_.StartTimer("camera_frame_decode");
    StoreFrameForVideo(currentFrame_, currentFrameId_, currentTimestamp_);
    perfStats_.EndTimer("camera_frame_decode");
    
    if (result.objNum > 0) {
        if (MatchFrame(currentFrameId_, currentTimestamp_, result)) {
            currentBoxes_ = result.objs;
            
            perfStats_.IncrementCounter("camera_detections_with_person", result.objNum);
            pipelineStats_.framesWithPerson++;
            
            LogFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
            
            Transition(State::DIFFERENCE_ANALYSIS);
        } else {
            LOG_WARN("Frame match failed, continue");
            perfStats_.IncrementCounter("frame_match_failures");
        }
    } else {
        currentBoxes_.clear();
        LogFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
    }
    
    pipelineStats_.totalFramesProcessed++;
    frameCounter_++;
}

void StateMachine::ProcessLocalDetection() {
    if (!rtspClient_ || !detector_) {
        LOG_ERROR("Components not initialized for local detection mode");
        Transition(State::ERROR);
        return;
    }
    
    if (!rtspClient_->IsConnected()) {
        LOG_WARN("RTSP disconnected in local detection mode");
        Transition(State::RECONNECTING);
        return;
    }
    
    if (frameCounter_ % config_.localDetection.detectInterval != 0) {
        frameCounter_++;
        
        if (!rtspClient_->GetFrame(currentFrame_, currentFrameId_, currentTimestamp_)) {
            LOG_WARN("Failed to get frame in skip mode");
            Transition(State::RECONNECTING);
            return;
        }
        
        StoreFrameForVideo(currentFrame_, currentFrameId_, currentTimestamp_);
        currentBoxes_.clear();
        
        perfStats_.IncrementCounter("frames_skipped");
        return;
    }
    
    if (!rtspClient_->GetFrame(currentFrame_, currentFrameId_, currentTimestamp_)) {
        LOG_WARN("Failed to get frame in local detection mode");
        Transition(State::RECONNECTING);
        return;
    }
    
    StoreFrameForVideo(currentFrame_, currentFrameId_, currentTimestamp_);
    
    currentBoxes_ = detector_->Detect(currentFrame_);
    
    if (config_.tracker.enabled && !currentBoxes_.empty()) {
        if (!tracker_) {
            LOG_ERROR("Tracker not initialized but enabled in config");
        } else {
            perfStats_.StartTimer("tracker_update");
            currentTracks_ = tracker_->Update(currentFrame_, currentBoxes_);
            perfStats_.EndTimer("tracker_update");
            
            currentBoxes_.clear();
            for (const auto& track : currentTracks_) {
                if (track.state == TrackState::Tracked) {
                    currentBoxes_.push_back(track.ToBoundingBox());
                }
            }
            
            perfStats_.IncrementCounter("tracked_objects", currentBoxes_.size());
        }
    }
    
    frameCounter_++;
    pipelineStats_.totalFramesProcessed++;
    
    if (!currentBoxes_.empty()) {
        pipelineStats_.framesWithPerson++;
        LogFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
        Transition(State::DIFFERENCE_ANALYSIS);
    } else {
        LogFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
    }
    
    pipelineStats_.avgDetectionTime = detector_->GetLastDetectTime();
}

bool StateMachine::CheckForPerson(const std::vector<BoundingBox>& boxes) {
    return !boxes.empty();
}

bool StateMachine::MatchFrame(int frameId, int64_t timestamp, 
                               const CameraDetectionResult& result) {
    if (!detectionReader_) {
        return false;
    }
    
    perfStats_.StartTimer("frame_matching");
    bool matched = detectionReader_->MatchFrame(frameId, timestamp, result);
    perfStats_.EndTimer("frame_matching");
    
    return matched;
}

void StateMachine::InitializeComponents() {
    LOG_INFO("Initializing components...");
    
    rtspClient_ = std::make_unique<RtspClient>();
    
    decoder_ = std::make_unique<FrameDecoder>();
    
    capabilityChecker_ = std::make_unique<CameraCapabilityChecker>();
    
    detectionReader_ = std::make_unique<CameraDetectionReader>();
    
    detector_ = std::make_unique<RknnDetector>(config_.localDetection);
    detector_->Init();
    detector_->setPerformanceStats(&perfStats_);
    
    if (config_.tracker.enabled) {
        tracker_ = std::make_unique<ByteTracker>(config_.tracker);
    }
    
    frameDiffAnalyzer_ = std::make_unique<FrameDiffAnalyzer>(config_.refFrame);
    
    eventAnalyzer_ = std::make_unique<EventAnalyzer>(config_.eventAnalysis);
    
    LOG_INFO("All components initialized successfully");
}

void StateMachine::CleanupComponents() {
    LOG_INFO("Cleaning up components...");
    
    if (rtspClient_) {
        rtspClient_->Disconnect();
    }
    
    if (videoBuffer_) {
        videoBuffer_->Clear();
    }
    
    currentBoxes_.clear();
    currentTracks_.clear();
    currentFrame_.release();
    
    LOG_INFO("Components cleaned up");
}

void StateMachine::StoreFrameForVideo(const cv::Mat& frame, int frameId, int64_t timestamp) {
    if (eventMode_ == EventAnalysisMode::kVideo && videoBuffer_) {
        videoBuffer_->AddFrame(frame, frameId, timestamp);
    }
}

std::vector<FrameWithMeta> StateMachine::GetVideoFrames() {
    if (!videoBuffer_) {
        return {};
    }
    
    int frameCount = static_cast<int>(config_.eventAnalysis.videoDurationSec * 30);
    return videoBuffer_->GetFramesWithMeta(frameCount);
}

void StateMachine::UpdateStats(bool hasPerson, bool IsSimilar) {
    perfStats_.IncrementCounter("total_frames_processed");
    
    if (hasPerson) {
        perfStats_.IncrementCounter("frames_with_person");
    }
    
    if (IsSimilar) {
        perfStats_.IncrementCounter("frames_similar");
    } else {
        perfStats_.IncrementCounter("frames_different");
    }
}

void StateMachine::LogFrameInfo(int frameId, int64_t timestamp, 
                                  const std::vector<BoundingBox>& boxes) {
    std::ostringstream oss;
    oss << "Frame " << frameId << " (ts=" << timestamp << "): ";
    
    if (boxes.empty()) {
        oss << "no person detected";
    } else {
        oss << boxes.size() << " persons";
        for (size_t i = 0; i < std::min(boxes.size(), size_t(2)); ++i) {
            oss << " [" << (int)boxes[i].x1 << "," << (int)boxes[i].y1 << "-" 
                << (int)boxes[i].x2 << "," << (int)boxes[i].y2 << "]";
        }
    }
    
    LOG_INFO(oss.str());
}

}