#pragma once

#include "common/config.h"
#include "core/state_machine.h"
#include <memory>
#include <atomic>
#include <thread>

namespace diff_det {

class Pipeline {
public:
    Pipeline(const Config& config);
    ~Pipeline();
    
    void start();
    void stop();
    void pause();
    void resume();
    
    bool isRunning() const;
    bool isPaused() const;
    
    State getCurrentState() const;
    int getFrameCount() const;
    int getEventCount() const;
    PipelineStats getStats() const;
    
    void setConfig(const Config& config);
    Config getConfig() const;
    
private:
    void runThread();
    
    Config config_;
    std::unique_ptr<StateMachine> stateMachine_;
    std::unique_ptr<std::thread> workerThread_;
    
    std::atomic<bool> running_;
    std::atomic<bool> paused_;
};

}