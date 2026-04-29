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

    ConfigValidationError(const std::string& fieldVal, const std::string& msgVal, const std::string& suggestVal = "")
        : field(fieldVal), message(msgVal), suggestion(suggestVal) {}
};

struct RtspConfig {
    std::string url;
    int32_t reconnectIntervalMs;

    RtspConfig() : reconnectIntervalMs(3000) {}

    bool IsValid() const;
    std::string ToString() const;
};

struct CameraDetectionConfig {
    bool enabled;
    std::string protocol;
    std::string endpoint;
    int32_t pollIntervalMs;
    std::string cameraId;
    std::string cameraHost;
    int32_t cameraPort;
    std::string capabilityUrl;
    int32_t timeoutMs;

    CameraDetectionConfig() : enabled(true), protocol("REST"), pollIntervalMs(100),
                               cameraPort(80), timeoutMs(3000) {}

    bool IsValid() const;
    std::string ToString() const;
};

struct LocalDetectionConfig {
    float confThreshold;
    float nmsThreshold;
    bool useLetterBox;
    int32_t detectInterval;
    int32_t timeoutMs;
    std::string modelPath;
    std::string modelType;
    std::string labelPath;

    LocalDetectionConfig() : modelType("yolov8"), confThreshold(0.5f),
                              nmsThreshold(0.3f), detectInterval(3), timeoutMs(500) {}

    bool IsValid() const;
    std::string ToString() const;
};

struct TrackerConfig {
    bool enabled;
    int32_t confirmFrames;
    int32_t maxLostFrames;
    float highThreshold;
    float lowThreshold;
    float matchThreshold;

    TrackerConfig() : enabled(false), confirmFrames(3), maxLostFrames(30),
                       highThreshold(0.5f), lowThreshold(0.1f), matchThreshold(0.5f) {}

    bool IsValid() const;
    std::string ToString() const;
};

struct RefFrameConfig {
    float similarityThreshold;
    std::string compareMethod;
    std::string updateStrategy;
    bool compareRoiOnly;

    RefFrameConfig() : similarityThreshold(0.85f), compareMethod("ssim"),
                        updateStrategy("newest"), compareRoiOnly(false) {}

    bool IsValid() const;
    std::string ToString() const;
};

struct EventAnalysisConfig {
    std::string mode;
    int32_t videoDurationSec;
    std::string webhookUrl;
    bool webhookEnabled;
    bool saveImg;
    bool withBox;

    EventAnalysisConfig() : mode("image"), videoDurationSec(5), webhookUrl("http://localhost:8080/api/vision"), webhookEnabled(false), saveImg(true), withBox(true) {}

    bool IsValid() const;
    std::string ToString() const;
};

struct LoggingConfig {
    std::string level;
    std::string filePath;

    LoggingConfig() : level("INFO"), filePath("/data/logs/pipeline.Log") {}

    bool IsValid() const;
    std::string ToString() const;
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

    static Config FromFile(const std::string& path);
    static Config FromYaml(const YAML::Node& node);

    bool IsValid() const;
    std::vector<ConfigValidationError> Validate() const;

    std::string ToString() const;
    std::string ToSummary() const;

    void SaveToFile(const std::string& path) const;
    YAML::Node ToYaml() const;

    std::string GetConfigPath() const { return configPath_; }

    using ConfigChangeCallback = std::function<void(const Config&)>;
    void SetChangeCallback(ConfigChangeCallback callback);

private:
    std::string configPath_;
    ConfigChangeCallback changeCallback_;
};

}