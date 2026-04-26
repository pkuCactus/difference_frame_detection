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
    
    void Start();
    void Stop();
    void Pause();
    void Resume();
    
    bool IsRunning() const;
    bool isPaused() const;
    
    State GetCurrentState() const;
    int GetFrameCount() const;
    int GetEventCount() const;
    PipelineStats GetStats() const;
    
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