#include "core/state_machine.h"
#include "common/logger.h"
#include <chrono>
#include <thread>
#include <sstream>
#include <iomanip>

namespace diff_det {

std::string PipelineStats::toString() const {
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
    oss << "Avg total process time: " << avgTotalProcessTime << "ms\n";
    return oss.str();
}

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
    , lastReconnectAttemptTime_(0)
    , reconnectAttempts_(0) {
    
    auto now = std::chrono::system_clock::now();
    pipelineStartTime_ = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

StateMachine::~StateMachine() {
    stop();
    cleanupComponents();
    
    LOG_INFO(pipelineStats_.toString());
}

void StateMachine::run() {
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
                handleInit();
                break;
            case State::CONNECTING:
                handleConnecting();
                break;
            case State::CHECK_CAPABILITY:
                handleCheckCapability();
                break;
            case State::CAMERA_DETECTION_MODE:
                handleCameraDetectionMode();
                break;
            case State::LOCAL_DETECTION_MODE:
                handleLocalDetectionMode();
                break;
            case State::DIFFERENCE_ANALYSIS:
                handleDifferenceAnalysis();
                break;
            case State::EVENT_ANALYSIS:
                handleEventAnalysis();
                break;
            case State::UPDATE_REF:
                handleUpdateRef();
                break;
            case State::RECONNECTING:
                handleReconnecting();
                break;
            case State::ERROR:
                handleError();
                break;
            default:
                LOG_ERROR("Unknown state: " + std::to_string(static_cast<int>(state_)));
                transition(State::ERROR);
                break;
        }
    }
    
    LOG_INFO("StateMachine stopped");
    LOG_INFO(pipelineStats_.toString());
}

void StateMachine::stop() {
    running_ = false;
    paused_ = false;
}

void StateMachine::pause() {
    paused_ = true;
    LOG_INFO("StateMachine paused");
}

void StateMachine::resume() {
    paused_ = false;
    LOG_INFO("StateMachine resumed");
}

State StateMachine::currentState() const {
    return state_;
}

int StateMachine::getEventCount() const {
    if (eventAnalyzer_) {
        return eventAnalyzer_->getEventCount();
    }
    return 0;
}

PipelineStats StateMachine::getStats() const {
    return pipelineStats_;
}

void StateMachine::transition(State newState) {
    LOG_INFO("State transition: " + std::string(stateToString(state_)) + 
             " -> " + std::string(stateToString(newState)));
    state_ = newState;
    perfStats_.incrementCounter("state_transitions");
}

void StateMachine::handleInit() {
    LOG_INFO("Initializing StateMachine...");
    
    Logger::getInstance().init(config_.logging.filePath, config_.logging.level);
    
    initializeComponents();
    
    int maxBufferFrames = static_cast<int>(config_.eventAnalysis.videoDurationSec * 30) + 10;
    videoBuffer_ = std::make_unique<VideoFrameBuffer>(maxBufferFrames);
    
    transition(State::CONNECTING);
}

void StateMachine::handleConnecting() {
    LOG_INFO("Connecting to RTSP: " + config_.rtsp.url);
    
    perfStats_.startTimer("rtsp_connect");
    
    if (checkConnection()) {
        perfStats_.endTimer("rtsp_connect");
        
        double fps = rtspClient_->getFps();
        int width = rtspClient_->getWidth();
        int height = rtspClient_->getHeight();
        
        LOG_INFO("RTSP connected: fps=" + std::to_string(fps) +
                 ", resolution=" + std::to_string(width) + "x" + std::to_string(height));
        
        transition(State::CHECK_CAPABILITY);
    } else {
        perfStats_.endTimer("rtsp_connect");
        LOG_WARN("RTSP connection failed");
        transition(State::RECONNECTING);
    }
}

void StateMachine::handleCheckCapability() {
    LOG_INFO("Checking camera detection capability");
    
    if (!capabilityChecker_) {
        LOG_ERROR("CapabilityChecker not initialized");
        transition(State::ERROR);
        return;
    }
    
    perfStats_.startTimer("capability_check");
    CapabilityResult result = capabilityChecker_->check(config_.cameraDetection);
    perfStats_.endTimer("capability_check");
    
    if (result.supported) {
        useCameraDetection_ = true;
        
        if (detectionReader_) {
            detectionReader_->init(config_.cameraDetection);
        }
        
        LOG_INFO("camera_id=" + config_.cameraDetection.cameraId + 
                 ", detection_supported=true, mode=camera_detection");
        transition(State::CAMERA_DETECTION_MODE);
    } else {
        useCameraDetection_ = false;
        LOG_INFO("camera_id=" + config_.cameraDetection.cameraId + 
                 ", detection_supported=false, reason=" + result.reason + 
                 ", mode=local_detection");
        transition(State::LOCAL_DETECTION_MODE);
    }
}

void StateMachine::handleCameraDetectionMode() {
    processCameraDetection();
}

void StateMachine::handleLocalDetectionMode() {
    processLocalDetection();
}

void StateMachine::handleDifferenceAnalysis() {
    if (!frameDiffAnalyzer_) {
        LOG_ERROR("FrameDiffAnalyzer not initialized");
        transition(State::ERROR);
        return;
    }
    
    perfStats_.startTimer("frame_diff_analysis");
    
    frameDiffAnalyzer_->setBoxesForRoi(currentBoxes_);
    
    if (!frameDiffAnalyzer_->hasRef()) {
        perfStats_.endTimer("frame_diff_analysis");
        
        LOG_INFO("No Ref frame, first frame must enter event analysis");
        hasDifference_ = true;
        transition(State::EVENT_ANALYSIS);
        return;
    }
    
    cv::Mat refFrame = frameDiffAnalyzer_->getRef();
    bool isSimilar = frameDiffAnalyzer_->isSimilar(currentFrame_, refFrame);
    
    perfStats_.endTimer("frame_diff_analysis");
    
    updateStats(!currentBoxes_.empty(), isSimilar);
    
    if (isSimilar) {
        LOG_INFO("Frame similar to Ref, skip event analysis");
        hasDifference_ = false;
        pipelineStats_.framesSimilar++;
        
        if (config_.refFrame.updateStrategy == "newest" && !currentBoxes_.empty()) {
            frameDiffAnalyzer_->updateRef(currentFrame_);
        }
        
        if (useCameraDetection_) {
            transition(State::CAMERA_DETECTION_MODE);
        } else {
            transition(State::LOCAL_DETECTION_MODE);
        }
    } else {
        LOG_INFO("Frame different from Ref, enter event analysis");
        hasDifference_ = true;
        pipelineStats_.framesDifferent++;
        transition(State::EVENT_ANALYSIS);
    }
}

void StateMachine::handleEventAnalysis() {
    if (!eventAnalyzer_) {
        LOG_ERROR("EventAnalyzer not initialized");
        transition(State::ERROR);
        return;
    }
    
    perfStats_.startTimer("event_analysis");
    
    LOG_INFO("Entering event analysis, frameId=" + std::to_string(currentFrameId_) +
             ", boxes=" + std::to_string(currentBoxes_.size()));
    
    if (config_.eventAnalysis.mode == "image") {
        eventAnalyzer_->analyzeImage(currentFrame_, currentBoxes_);
    } else {
        std::vector<FrameWithMeta> frames = getVideoFrames();
        if (frames.empty()) {
            FrameWithMeta meta(currentFrame_, currentFrameId_, currentTimestamp_);
            frames.push_back(meta);
        }
        
        std::vector<cv::Mat> frameImages;
        for (auto& f : frames) {
            frameImages.push_back(f.frame);
        }
        eventAnalyzer_->analyzeVideo(frameImages, currentBoxes_);
    }
    
    perfStats_.endTimer("event_analysis");
    
    pipelineStats_.eventsGenerated++;
    perfStats_.incrementCounter("events_generated");
    
    transition(State::UPDATE_REF);
}

void StateMachine::handleUpdateRef() {
    if (!frameDiffAnalyzer_) {
        LOG_ERROR("FrameDiffAnalyzer not initialized");
        transition(State::ERROR);
        return;
    }
    
    LOG_INFO("Updating Ref frame");
    
    if (config_.refFrame.updateStrategy == "default" || hasDifference_) {
        frameDiffAnalyzer_->updateRef(currentFrame_);
    }
    
    if (useCameraDetection_) {
        transition(State::CAMERA_DETECTION_MODE);
    } else {
        transition(State::LOCAL_DETECTION_MODE);
    }
}

void StateMachine::handleReconnecting() {
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
    
    LOG_WARN("Attempting to reconnect (" + std::to_string(reconnectAttempts_) + " attempts)...");
    
    if (rtspClient_ && rtspClient_->reconnect()) {
        LOG_INFO("Reconnected successfully after " + std::to_string(reconnectAttempts_) + " attempts");
        reconnectAttempts_ = 0;
        transition(State::CHECK_CAPABILITY);
    } else {
        LOG_ERROR("Reconnect failed, will retry after " + 
                  std::to_string(config_.rtsp.reconnectIntervalMs) + "ms");
    }
}

void StateMachine::handleError() {
    LOG_ERROR("Error state encountered");
    LOG_INFO(pipelineStats_.toString());
    cleanupComponents();
    running_ = false;
}

bool StateMachine::checkConnection() {
    if (!rtspClient_) {
        LOG_ERROR("RtspClient not initialized");
        return false;
    }
    
    return rtspClient_->connect(config_.rtsp.url);
}

void StateMachine::processCameraDetection() {
    if (!rtspClient_ || !detectionReader_) {
        LOG_ERROR("Components not initialized for camera detection mode");
        transition(State::ERROR);
        return;
    }
    
    if (!rtspClient_->isConnected()) {
        LOG_WARN("RTSP disconnected in camera detection mode");
        transition(State::RECONNECTING);
        return;
    }
    
    perfStats_.startTimer("camera_detection_fetch");
    CameraDetectionResult result = detectionReader_->getDetectionResult();
    perfStats_.endTimer("camera_detection_fetch");
    
    if (!rtspClient_->getFrame(currentFrame_, currentFrameId_, currentTimestamp_)) {
        LOG_WARN("Failed to get frame in camera detection mode");
        transition(State::RECONNECTING);
        return;
    }
    
    perfStats_.startTimer("camera_frame_decode");
    storeFrameForVideo(currentFrame_, currentFrameId_, currentTimestamp_);
    perfStats_.endTimer("camera_frame_decode");
    
    if (result.objNum > 0) {
        if (matchFrame(currentFrameId_, currentTimestamp_, result)) {
            currentBoxes_ = result.objs;
            
            perfStats_.incrementCounter("camera_detections_with_person", result.objNum);
            pipelineStats_.framesWithPerson++;
            
            logFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
            
            transition(State::DIFFERENCE_ANALYSIS);
        } else {
            LOG_WARN("Frame match failed, continue");
            perfStats_.incrementCounter("frame_match_failures");
        }
    } else {
        currentBoxes_.clear();
        logFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
    }
    
    pipelineStats_.totalFramesProcessed++;
    frameCounter_++;
}

void StateMachine::processLocalDetection() {
    if (!rtspClient_ || !detector_) {
        LOG_ERROR("Components not initialized for local detection mode");
        transition(State::ERROR);
        return;
    }
    
    if (!rtspClient_->isConnected()) {
        LOG_WARN("RTSP disconnected in local detection mode");
        transition(State::RECONNECTING);
        return;
    }
    
    if (frameCounter_ % config_.localDetection.detectInterval != 0) {
        frameCounter_++;
        
        if (!rtspClient_->getFrame(currentFrame_, currentFrameId_, currentTimestamp_)) {
            LOG_WARN("Failed to get frame in skip mode");
            transition(State::RECONNECTING);
            return;
        }
        
        storeFrameForVideo(currentFrame_, currentFrameId_, currentTimestamp_);
        currentBoxes_.clear();
        
        perfStats_.incrementCounter("frames_skipped");
        return;
    }
    
    if (!rtspClient_->getFrame(currentFrame_, currentFrameId_, currentTimestamp_)) {
        LOG_WARN("Failed to get frame in local detection mode");
        transition(State::RECONNECTING);
        return;
    }
    
    storeFrameForVideo(currentFrame_, currentFrameId_, currentTimestamp_);
    
    currentBoxes_ = detector_->detect(currentFrame_);
    
    if (config_.tracker.enabled && !currentBoxes_.empty()) {
        if (!tracker_) {
            LOG_ERROR("Tracker not initialized but enabled in config");
        } else {
            perfStats_.startTimer("tracker_update");
            currentTracks_ = tracker_->update(currentFrame_, currentBoxes_);
            perfStats_.endTimer("tracker_update");
            
            currentBoxes_.clear();
            for (const auto& track : currentTracks_) {
                if (track.state == TrackState::Tracked) {
                    currentBoxes_.push_back(track.toBoundingBox());
                }
            }
            
            perfStats_.incrementCounter("tracked_objects", currentBoxes_.size());
        }
    }
    
    frameCounter_++;
    pipelineStats_.totalFramesProcessed++;
    
    if (!currentBoxes_.empty()) {
        pipelineStats_.framesWithPerson++;
        logFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
        transition(State::DIFFERENCE_ANALYSIS);
    } else {
        logFrameInfo(currentFrameId_, currentTimestamp_, currentBoxes_);
    }
    
    pipelineStats_.avgDetectionTime = detector_->getLastDetectTime();
}

bool StateMachine::checkForPerson(const std::vector<BoundingBox>& boxes) {
    return !boxes.empty();
}

bool StateMachine::matchFrame(int frameId, int64_t timestamp, 
                               const CameraDetectionResult& result) {
    if (!detectionReader_) {
        return false;
    }
    
    perfStats_.startTimer("frame_matching");
    bool matched = detectionReader_->matchFrame(frameId, timestamp, result);
    perfStats_.endTimer("frame_matching");
    
    return matched;
}

void StateMachine::initializeComponents() {
    LOG_INFO("Initializing components...");
    
    rtspClient_ = std::make_unique<RtspClient>();
    
    decoder_ = std::make_unique<FrameDecoder>();
    
    capabilityChecker_ = std::make_unique<CameraCapabilityChecker>();
    
    detectionReader_ = std::make_unique<CameraDetectionReader>();
    
    detector_ = std::make_unique<RknnDetector>(config_.localDetection);
    detector_->init();
    detector_->setPerformanceStats(&perfStats_);
    
    if (config_.tracker.enabled) {
        tracker_ = std::make_unique<ByteTracker>(config_.tracker);
    }
    
    frameDiffAnalyzer_ = std::make_unique<FrameDiffAnalyzer>(config_.refFrame);
    
    eventAnalyzer_ = std::make_unique<EventAnalyzer>(config_.eventAnalysis);
    
    LOG_INFO("All components initialized successfully");
}

void StateMachine::cleanupComponents() {
    LOG_INFO("Cleaning up components...");
    
    if (rtspClient_) {
        rtspClient_->disconnect();
    }
    
    if (videoBuffer_) {
        videoBuffer_->clear();
    }
    
    currentBoxes_.clear();
    currentTracks_.clear();
    currentFrame_.release();
    
    LOG_INFO("Components cleaned up");
}

void StateMachine::storeFrameForVideo(const cv::Mat& frame, int frameId, int64_t timestamp) {
    if (config_.eventAnalysis.mode == "video" && videoBuffer_) {
        videoBuffer_->addFrame(frame, frameId, timestamp);
    }
}

std::vector<FrameWithMeta> StateMachine::getVideoFrames() {
    if (!videoBuffer_) {
        return {};
    }
    
    int frameCount = static_cast<int>(config_.eventAnalysis.videoDurationSec * 30);
    return videoBuffer_->getFramesWithMeta(frameCount);
}

void StateMachine::updateStats(bool hasPerson, bool isSimilar) {
    perfStats_.incrementCounter("total_frames_processed");
    
    if (hasPerson) {
        perfStats_.incrementCounter("frames_with_person");
    }
    
    if (isSimilar) {
        perfStats_.incrementCounter("frames_similar");
    } else {
        perfStats_.incrementCounter("frames_different");
    }
}

void StateMachine::logFrameInfo(int frameId, int64_t timestamp, 
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