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
    stop();
    LOG_INFO("Pipeline destroyed");
}

void Pipeline::start() {
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

void Pipeline::stop() {
    if (!running_) {
        return;
    }
    
    LOG_INFO("Pipeline stopping...");
    
    running_ = false;
    paused_ = false;
    
    if (stateMachine_) {
        stateMachine_->stop();
    }
    
    if (workerThread_ && workerThread_->joinable()) {
        workerThread_->join();
    }
    
    stateMachine_.reset();
    workerThread_.reset();
    
    LOG_INFO("Pipeline stopped");
}

void Pipeline::pause() {
    if (!running_) {
        LOG_WARN("Pipeline not running, cannot pause");
        return;
    }
    
    paused_ = true;
    
    if (stateMachine_) {
        stateMachine_->pause();
    }
    
    LOG_INFO("Pipeline paused");
}

void Pipeline::resume() {
    if (!running_) {
        LOG_WARN("Pipeline not running, cannot resume");
        return;
    }
    
    paused_ = false;
    
    if (stateMachine_) {
        stateMachine_->resume();
    }
    
    LOG_INFO("Pipeline resumed");
}

bool Pipeline::isRunning() const {
    return running_;
}

bool Pipeline::isPaused() const {
    return paused_;
}

State Pipeline::getCurrentState() const {
    if (stateMachine_) {
        return stateMachine_->currentState();
    }
    return State::INIT;
}

int Pipeline::getFrameCount() const {
    if (stateMachine_) {
        return stateMachine_->getFrameCount();
    }
    return 0;
}

int Pipeline::getEventCount() const {
    if (stateMachine_) {
        return stateMachine_->getEventCount();
    }
    return 0;
}

PipelineStats Pipeline::getStats() const {
    if (stateMachine_) {
        return stateMachine_->getStats();
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
        stateMachine_->run();
    }
    
    LOG_INFO("Pipeline worker thread finished");
}

}