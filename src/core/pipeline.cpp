#include "core/pipeline.h"
#include "common/logger.h"
#include <chrono>

namespace diff_det {

Pipeline::Pipeline(const Config& config)
    : config_(config)
    , running_(false)
    , paused_(false) {
    LOG_INFO("Pipeline created: rtsp_url=" + config_.rtsp.url +
             ", detection_mode=" + (config_.cameraDetection.enabled ? "camera" : "local") +
             ", tracker_enabled=" + (config_.tracker.enabled ? "true" : "false"));
}

Pipeline::~Pipeline() {
    Stop();
    LOG_INFO("Pipeline destroyed");
}

void Pipeline::Start() {
    if (running_) {
        LOG_WARN("Pipeline already running");
        return;
    }
    
    running_ = true;
    paused_ = false;
    
    LOG_INFO("Pipeline starting...");
    
    stateMachine_ = std::make_unique<StateMachine>(config_);
    
    workerThread_ = std::make_unique<std::thread>(&Pipeline::runThread, this);
    
    LOG_INFO("Pipeline started successfully");
}

void Pipeline::Stop() {
    if (!running_) {
        return;
    }
    
    LOG_INFO("Pipeline stopping...");
    
    running_ = false;
    paused_ = false;
    
    if (stateMachine_) {
        stateMachine_->Stop();
    }
    
    if (workerThread_ && workerThread_->joinable()) {
        workerThread_->join();
    }
    
    stateMachine_.reset();
    workerThread_.reset();
    
    LOG_INFO("Pipeline stopped");
}

void Pipeline::Pause() {
    if (!running_) {
        LOG_WARN("Pipeline not running, cannot Pause");
        return;
    }
    
    paused_ = true;
    
    if (stateMachine_) {
        stateMachine_->Pause();
    }
    
    LOG_INFO("Pipeline paused");
}

void Pipeline::Resume() {
    if (!running_) {
        LOG_WARN("Pipeline not running, cannot Resume");
        return;
    }
    
    paused_ = false;
    
    if (stateMachine_) {
        stateMachine_->Resume();
    }
    
    LOG_INFO("Pipeline resumed");
}

bool Pipeline::IsRunning() const {
    return running_;
}

bool Pipeline::isPaused() const {
    return paused_;
}

State Pipeline::GetCurrentState() const {
    if (stateMachine_) {
        return stateMachine_->CurrentState();
    }
    return State::INIT;
}

int Pipeline::GetFrameCount() const {
    if (stateMachine_) {
        return stateMachine_->GetFrameCount();
    }
    return 0;
}

int Pipeline::GetEventCount() const {
    if (stateMachine_) {
        return stateMachine_->GetEventCount();
    }
    return 0;
}

PipelineStats Pipeline::GetStats() const {
    if (stateMachine_) {
        return stateMachine_->GetStats();
    }
    return PipelineStats();
}

void Pipeline::setConfig(const Config& config) {
    if (running_) {
        LOG_WARN("Cannot change config while pipeline is running");
        return;
    }
    
    config_ = config;
    LOG_INFO("Pipeline config updated");
}

Config Pipeline::getConfig() const {
    return config_;
}

void Pipeline::runThread() {
    LOG_INFO("Pipeline worker thread started");
    
    if (stateMachine_) {
        stateMachine_->Run();
    }
    
    LOG_INFO("Pipeline worker thread finished");
}

}