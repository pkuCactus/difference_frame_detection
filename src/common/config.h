#pragma once

#include <yaml-cpp/yaml.h>
#include <string>
#include <vector>
#include <functional>

namespace diff_det {

struct ConfigValidationError {
    std::string field;
    std::string message;
    std::string suggestion;
    
    ConfigValidationError(const std::string& f, const std::string& m, const std::string& s = "")
        : field(f), message(m), suggestion(s) {}
};

struct RtspConfig {
    std::string url;
    int reconnectIntervalMs;
    
    RtspConfig() : reconnectIntervalMs(3000) {}
    
    bool isValid() const;
    std::string toString() const;
};

struct CameraDetectionConfig {
    bool enabled;
    std::string protocol;
    std::string endpoint;
    int pollIntervalMs;
    std::string cameraId;
    std::string cameraHost;
    int cameraPort;
    std::string capabilityUrl;
    int timeoutMs;
    
    CameraDetectionConfig() : enabled(true), protocol("REST"), pollIntervalMs(100), 
                               cameraPort(80), timeoutMs(3000) {}
    
    bool isValid() const;
    std::string toString() const;
};

struct LocalDetectionConfig {
    std::string modelPath;
    std::string modelType;
    float confThreshold;
    int detectInterval;
    int timeoutMs;
    
    LocalDetectionConfig() : modelType("yolov8"), confThreshold(0.5f), 
                              detectInterval(3), timeoutMs(500) {}
    
    bool isValid() const;
    std::string toString() const;
};

struct TrackerConfig {
    bool enabled;
    int confirmFrames;
    int maxLostFrames;
    float highThreshold;
    float lowThreshold;
    float matchThreshold;
    
    TrackerConfig() : enabled(false), confirmFrames(3), maxLostFrames(30),
                       highThreshold(0.5f), lowThreshold(0.1f), matchThreshold(0.5f) {}
    
    bool isValid() const;
    std::string toString() const;
};

struct RefFrameConfig {
    float similarityThreshold;
    std::string compareMethod;
    std::string updateStrategy;
    bool compareRoiOnly;
    
    RefFrameConfig() : similarityThreshold(0.85f), compareMethod("ssim"),
                        updateStrategy("newest"), compareRoiOnly(false) {}
    
    bool isValid() const;
    std::string toString() const;
};

struct EventAnalysisConfig {
    std::string mode;
    int videoDurationSec;
    
    EventAnalysisConfig() : mode("image"), videoDurationSec(5) {}
    
    bool isValid() const;
    std::string toString() const;
};

struct LoggingConfig {
    std::string level;
    std::string filePath;
    
    LoggingConfig() : level("INFO"), filePath("/data/logs/pipeline.log") {}
    
    bool isValid() const;
    std::string toString() const;
};

class Config {
public:
    RtspConfig rtsp;
    CameraDetectionConfig cameraDetection;
    LocalDetectionConfig localDetection;
    TrackerConfig tracker;
    RefFrameConfig refFrame;
    EventAnalysisConfig eventAnalysis;
    LoggingConfig logging;
    
    static Config fromFile(const std::string& path);
    static Config fromYaml(const YAML::Node& node);
    
    bool isValid() const;
    std::vector<ConfigValidationError> validate() const;
    
    std::string toString() const;
    std::string toSummary() const;
    
    void saveToFile(const std::string& path) const;
    YAML::Node toYaml() const;
    
    std::string getConfigPath() const { return configPath_; }
    
    using ConfigChangeCallback = std::function<void(const Config&)>;
    void setChangeCallback(ConfigChangeCallback callback);
    
private:
    std::string configPath_;
    ConfigChangeCallback changeCallback_;
};

}