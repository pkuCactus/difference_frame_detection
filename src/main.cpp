#include "core/pipeline.h"
#include "common/config.h"
#include "common/logger.h"

#include <iostream>
#include <signal.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <iomanip>

using namespace diff_det;

std::atomic<bool> g_running(true);
Pipeline* g_pipeline = nullptr;

void signalHandler(int signum) {
    std::cout << "\nReceived signal " << signum << ", stopping gracefully..." << std::endl;
    g_running = false;
    
    if (g_pipeline) {
        g_pipeline->stop();
    }
}

void printStats(const PipelineStats& stats, int intervalSec) {
    std::cout << "\n=== Statistics (last " << intervalSec << "s) ===" << std::endl;
    std::cout << "Frames processed: " << stats.totalFramesProcessed << std::endl;
    std::cout << "Frames with person: " << stats.framesWithPerson << std::endl;
    std::cout << "Frames similar: " << stats.framesSimilar << std::endl;
    std::cout << "Frames different: " << stats.framesDifferent << std::endl;
    std::cout << "Events generated: " << stats.eventsGenerated << std::endl;
    std::cout << "Avg detection time: " << std::fixed << std::setprecision(2) 
              << stats.avgDetectionTime << "ms" << std::endl;
    std::cout << "Avg frame diff time: " << stats.avgFrameDiffTime << "ms" << std::endl;
    std::cout << "=========================" << std::endl;
}

int main(int argc, char* argv[]) {
    std::string configPath = "config/config.yaml";
    
    if (argc > 1) {
        configPath = argv[1];
    }
    
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    std::cout << "========================================" << std::endl;
    std::cout << "Difference Detection Pipeline v1.0" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Config file: " << configPath << std::endl;
    
    try {
        Config config = Config::fromFile(configPath);
        
        std::cout << "RTSP URL: " << config.rtsp.url << std::endl;
        std::cout << "Detection mode: " << (config.cameraDetection.enabled ? "camera" : "local") << std::endl;
        std::cout << "Tracker: " << (config.tracker.enabled ? "enabled" : "disabled") << std::endl;
        std::cout << "Event mode: " << config.eventAnalysis.mode << std::endl;
        std::cout << "Ref update strategy: " << config.refFrame.updateStrategy << std::endl;
        std::cout << "Similarity method: " << config.refFrame.compareMethod << std::endl;
        std::cout << "Similarity threshold: " << config.refFrame.similarityThreshold << std::endl;
        std::cout << "Detect interval: " << config.localDetection.detectInterval << " frames" << std::endl;
        std::cout << "Log file: " << config.logging.filePath << std::endl;
        std::cout << "========================================" << std::endl;
        
        Pipeline pipeline(config);
        g_pipeline = &pipeline;
        pipeline.start();
        
        int statsInterval = 10;
        auto lastStatsTime = std::chrono::system_clock::now();
        int prevFrameCount = 0;
        
        while (g_running && pipeline.isRunning()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            State state = pipeline.getCurrentState();
            int frameCount = pipeline.getFrameCount();
            int eventCount = pipeline.getEventCount();
            
            std::cout << "[" << stateToString(state) << "] "
                      << "Frames: " << frameCount
                      << ", Events: " << eventCount
                      << ", FPS: " << (frameCount - prevFrameCount)
                      << std::endl;
            
            prevFrameCount = frameCount;
            
            auto now = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastStatsTime);
            
            if (elapsed.count() >= statsInterval) {
                printStats(pipeline.getStats(), statsInterval);
                lastStatsTime = now;
            }
        }
        
        pipeline.stop();
        g_pipeline = nullptr;
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Pipeline Final Statistics:" << std::endl;
        printStats(pipeline.getStats(), 0);
        std::cout << "Pipeline stopped gracefully" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML config error: " << e.what() << std::endl;
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}