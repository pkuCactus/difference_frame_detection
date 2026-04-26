#include "common/config.h"
#include "common/logger.h"
#include <fstream>
#include <sstream>
#include <iomanip>

namespace diff_det {

bool RtspConfig::isValid() const {
    return !url.empty() && reconnectIntervalMs > 0;
}

std::string RtspConfig::toString() const {
    std::ostringstream oss;
    oss << "RtspConfig:\n";
    oss << "  url: " << url << "\n";
    oss << "  reconnect_interval_ms: " << reconnectIntervalMs << "\n";
    return oss.str();
}

bool CameraDetectionConfig::isValid() const {
    if (!enabled) return true;
    return !protocol.empty() && 
           (protocol == "REST" || protocol == "ONVIF") &&
           pollIntervalMs > 0 && timeoutMs > 0;
}

std::string CameraDetectionConfig::toString() const {
    std::ostringstream oss;
    oss << "CameraDetectionConfig:\n";
    oss << "  enabled: " << (enabled ? "true" : "false") << "\n";
    oss << "  protocol: " << protocol << "\n";
    oss << "  endpoint: " << endpoint << "\n";
    oss << "  poll_interval_ms: " << pollIntervalMs << "\n";
    oss << "  camera_id: " << cameraId << "\n";
    oss << "  camera_host: " << cameraHost << "\n";
    oss << "  camera_port: " << cameraPort << "\n";
    oss << "  capability_url: " << capabilityUrl << "\n";
    oss << "  timeout_ms: " << timeoutMs << "\n";
    return oss.str();
}

bool LocalDetectionConfig::isValid() const {
    return !modelType.empty() &&
           (modelType == "yolov3" || modelType == "yolov5" || modelType == "yolov8") &&
           confThreshold >= 0.0f && confThreshold <= 1.0f &&
           detectInterval > 0 && timeoutMs > 0;
}

std::string LocalDetectionConfig::toString() const {
    std::ostringstream oss;
    oss << "LocalDetectionConfig:\n";
    oss << "  model_path: " << modelPath << "\n";
    oss << "  model_type: " << modelType << "\n";
    oss << "  conf_threshold: " << std::fixed << std::setprecision(2) << confThreshold << "\n";
    oss << "  detect_interval: " << detectInterval << "\n";
    oss << "  timeout_ms: " << timeoutMs << "\n";
    return oss.str();
}

bool TrackerConfig::isValid() const {
    if (!enabled) return true;
    return confirmFrames > 0 && maxLostFrames > 0 &&
           highThreshold >= 0.0f && highThreshold <= 1.0f &&
           lowThreshold >= 0.0f && lowThreshold <= 1.0f &&
           matchThreshold >= 0.0f && matchThreshold <= 1.0f &&
           highThreshold > lowThreshold;
}

std::string TrackerConfig::toString() const {
    std::ostringstream oss;
    oss << "TrackerConfig:\n";
    oss << "  enabled: " << (enabled ? "true" : "false") << "\n";
    oss << "  confirm_frames: " << confirmFrames << "\n";
    oss << "  max_lost_frames: " << maxLostFrames << "\n";
    oss << "  high_threshold: " << std::fixed << std::setprecision(2) << highThreshold << "\n";
    oss << "  low_threshold: " << lowThreshold << "\n";
    oss << "  match_threshold: " << matchThreshold << "\n";
    return oss.str();
}

bool RefFrameConfig::isValid() const {
    return similarityThreshold >= 0.0f && similarityThreshold <= 1.0f &&
           (compareMethod == "ssim" || compareMethod == "pixel_diff" || compareMethod == "phash") &&
           (updateStrategy == "newest" || updateStrategy == "default");
}

std::string RefFrameConfig::toString() const {
    std::ostringstream oss;
    oss << "RefFrameConfig:\n";
    oss << "  similarity_threshold: " << std::fixed << std::setprecision(2) << similarityThreshold << "\n";
    oss << "  compare_method: " << compareMethod << "\n";
    oss << "  update_strategy: " << updateStrategy << "\n";
    oss << "  compare_roi_only: " << (compareRoiOnly ? "true" : "false") << "\n";
    return oss.str();
}

bool EventAnalysisConfig::isValid() const {
    return (mode == "image" || mode == "video") && videoDurationSec > 0;
}

std::string EventAnalysisConfig::toString() const {
    std::ostringstream oss;
    oss << "EventAnalysisConfig:\n";
    oss << "  mode: " << mode << "\n";
    oss << "  video_duration_sec: " << videoDurationSec << "\n";
    return oss.str();
}

bool LoggingConfig::isValid() const {
    return (level == "DEBUG" || level == "INFO" || level == "WARNING" || level == "ERROR") &&
           !filePath.empty();
}

std::string LoggingConfig::toString() const {
    std::ostringstream oss;
    oss << "LoggingConfig:\n";
    oss << "  level: " << level << "\n";
    oss << "  file_path: " << filePath << "\n";
    return oss.str();
}

Config Config::fromFile(const std::string& path) {
    try {
        YAML::Node node = YAML::LoadFile(path);
        Config config = fromYaml(node);
        config.configPath_ = path;
        
        LOG_INFO("Config loaded from: " + path);
        
        auto errors = config.validate();
        if (!errors.empty()) {
            LOG_WARN("Config validation warnings:");
            for (auto& err : errors) {
                LOG_WARN("  " + err.field + ": " + err.message);
                if (!err.suggestion.empty()) {
                    LOG_WARN("    Suggestion: " + err.suggestion);
                }
            }
        }
        
        return config;
    } catch (const YAML::Exception& e) {
        LOG_ERROR("YAML parse error: " + std::string(e.what()));
        throw;
    }
}

Config Config::fromYaml(const YAML::Node& node) {
    Config config;
    
    if (node["rtsp"]) {
        auto rtspNode = node["rtsp"];
        if (rtspNode["url"]) {
            config.rtsp.url = rtspNode["url"].as<std::string>();
        }
        if (rtspNode["reconnect_interval_ms"]) {
            config.rtsp.reconnectIntervalMs = rtspNode["reconnect_interval_ms"].as<int>();
        }
        if (rtspNode["width"]) {
        }
        if (rtspNode["height"]) {
        }
        if (rtspNode["fps"]) {
        }
    }
    
    if (node["camera_detection"]) {
        auto camNode = node["camera_detection"];
        if (camNode["enabled"]) {
            config.cameraDetection.enabled = camNode["enabled"].as<bool>();
        }
        if (camNode["protocol"]) {
            config.cameraDetection.protocol = camNode["protocol"].as<std::string>();
        }
        if (camNode["endpoint"]) {
            config.cameraDetection.endpoint = camNode["endpoint"].as<std::string>();
        }
        if (camNode["poll_interval_ms"]) {
            config.cameraDetection.pollIntervalMs = camNode["poll_interval_ms"].as<int>();
        }
        if (camNode["camera_id"]) {
            config.cameraDetection.cameraId = camNode["camera_id"].as<std::string>();
        }
        if (camNode["camera_host"]) {
            config.cameraDetection.cameraHost = camNode["camera_host"].as<std::string>();
        }
        if (camNode["camera_port"]) {
            config.cameraDetection.cameraPort = camNode["camera_port"].as<int>();
        }
        if (camNode["capability_url"]) {
            config.cameraDetection.capabilityUrl = camNode["capability_url"].as<std::string>();
        }
        if (camNode["timeout_ms"]) {
            config.cameraDetection.timeoutMs = camNode["timeout_ms"].as<int>();
        }
    }
    
    if (node["local_detection"]) {
        auto detNode = node["local_detection"];
        if (detNode["model_path"]) {
            config.localDetection.modelPath = detNode["model_path"].as<std::string>();
        }
        if (detNode["model_type"]) {
            config.localDetection.modelType = detNode["model_type"].as<std::string>();
        }
        if (detNode["conf_threshold"]) {
            config.localDetection.confThreshold = detNode["conf_threshold"].as<float>();
        }
        if (detNode["detect_interval"]) {
            config.localDetection.detectInterval = detNode["detect_interval"].as<int>();
        }
        if (detNode["timeout_ms"]) {
            config.localDetection.timeoutMs = detNode["timeout_ms"].as<int>();
        }
        if (detNode["nms_threshold"]) {
        }
    }
    
    if (node["tracker"]) {
        auto trackerNode = node["tracker"];
        if (trackerNode["enabled"]) {
            config.tracker.enabled = trackerNode["enabled"].as<bool>();
        }
        if (trackerNode["confirm_frames"]) {
            config.tracker.confirmFrames = trackerNode["confirm_frames"].as<int>();
        }
        if (trackerNode["max_lost_frames"]) {
            config.tracker.maxLostFrames = trackerNode["max_lost_frames"].as<int>();
        }
        if (trackerNode["high_threshold"]) {
            config.tracker.highThreshold = trackerNode["high_threshold"].as<float>();
        }
        if (trackerNode["low_threshold"]) {
            config.tracker.lowThreshold = trackerNode["low_threshold"].as<float>();
        }
        if (trackerNode["match_threshold"]) {
            config.tracker.matchThreshold = trackerNode["match_threshold"].as<float>();
        }
    }
    
    if (node["ref_frame"]) {
        auto refNode = node["ref_frame"];
        if (refNode["similarity_threshold"]) {
            config.refFrame.similarityThreshold = refNode["similarity_threshold"].as<float>();
        }
        if (refNode["compare_method"]) {
            config.refFrame.compareMethod = refNode["compare_method"].as<std::string>();
        }
        if (refNode["update_strategy"]) {
            config.refFrame.updateStrategy = refNode["update_strategy"].as<std::string>();
        }
        if (refNode["compare_roi_only"]) {
            config.refFrame.compareRoiOnly = refNode["compare_roi_only"].as<bool>();
        }
    }
    
    if (node["event_analysis"]) {
        auto eventNode = node["event_analysis"];
        if (eventNode["mode"]) {
            config.eventAnalysis.mode = eventNode["mode"].as<std::string>();
        }
        if (eventNode["video_duration_sec"]) {
            config.eventAnalysis.videoDurationSec = eventNode["video_duration_sec"].as<int>();
        }
    }
    
    if (node["logging"]) {
        auto logNode = node["logging"];
        if (logNode["level"]) {
            config.logging.level = logNode["level"].as<std::string>();
        }
        if (logNode["file_path"]) {
            config.logging.filePath = logNode["file_path"].as<std::string>();
        }
    }
    
    return config;
}

bool Config::isValid() const {
    return validate().empty();
}

std::vector<ConfigValidationError> Config::validate() const {
    std::vector<ConfigValidationError> errors;
    
    if (rtsp.url.empty()) {
        errors.emplace_back("rtsp.url", "RTSP URL is required", "e.g., rtsp://192.168.1.100:554/stream");
    }
    if (rtsp.reconnectIntervalMs <= 0) {
        errors.emplace_back("rtsp.reconnect_interval_ms", "Must be positive", "default: 3000");
    }
    
    if (cameraDetection.enabled) {
        if (cameraDetection.protocol.empty() || 
            (cameraDetection.protocol != "REST" && cameraDetection.protocol != "ONVIF")) {
            errors.emplace_back("camera_detection.protocol", "Must be REST or ONVIF", "default: REST");
        }
        if (cameraDetection.pollIntervalMs <= 0) {
            errors.emplace_back("camera_detection.poll_interval_ms", "Must be positive", "default: 100");
        }
        if (cameraDetection.timeoutMs <= 0) {
            errors.emplace_back("camera_detection.timeout_ms", "Must be positive", "default: 3000");
        }
    }
    
    if (localDetection.modelType.empty() ||
        (localDetection.modelType != "yolov3" && 
         localDetection.modelType != "yolov5" &&
         localDetection.modelType != "yolov8")) {
        errors.emplace_back("local_detection.model_type", "Must be yolov3, yolov5, or yolov8", "default: yolov8");
    }
    if (localDetection.confThreshold < 0.0f || localDetection.confThreshold > 1.0f) {
        errors.emplace_back("local_detection.conf_threshold", "Must be between 0 and 1", "default: 0.5");
    }
    if (localDetection.detectInterval <= 0) {
        errors.emplace_back("local_detection.detect_interval", "Must be positive", "default: 3");
    }
    
    if (tracker.enabled) {
        if (tracker.confirmFrames <= 0) {
            errors.emplace_back("tracker.confirm_frames", "Must be positive", "default: 3");
        }
        if (tracker.maxLostFrames <= 0) {
            errors.emplace_back("tracker.max_lost_frames", "Must be positive", "default: 30");
        }
        if (tracker.highThreshold <= tracker.lowThreshold) {
            errors.emplace_back("tracker.high_threshold", "Must be greater than low_threshold", "high > low");
        }
    }
    
    if (refFrame.similarityThreshold < 0.0f || refFrame.similarityThreshold > 1.0f) {
        errors.emplace_back("ref_frame.similarity_threshold", "Must be between 0 and 1", "default: 0.85");
    }
    if (refFrame.compareMethod.empty() ||
        (refFrame.compareMethod != "ssim" && 
         refFrame.compareMethod != "pixel_diff" &&
         refFrame.compareMethod != "phash")) {
        errors.emplace_back("ref_frame.compare_method", "Must be ssim, pixel_diff, or phash", "default: ssim");
    }
    if (refFrame.updateStrategy.empty() ||
        (refFrame.updateStrategy != "newest" && refFrame.updateStrategy != "default")) {
        errors.emplace_back("ref_frame.update_strategy", "Must be newest or default", "default: newest");
    }
    
    if (eventAnalysis.mode.empty() ||
        (eventAnalysis.mode != "image" && eventAnalysis.mode != "video")) {
        errors.emplace_back("event_analysis.mode", "Must be image or video", "default: image");
    }
    if (eventAnalysis.videoDurationSec <= 0) {
        errors.emplace_back("event_analysis.video_duration_sec", "Must be positive", "default: 5");
    }
    
    if (logging.level.empty() ||
        (logging.level != "DEBUG" && 
         logging.level != "INFO" &&
         logging.level != "WARNING" &&
         logging.level != "ERROR")) {
        errors.emplace_back("logging.level", "Must be DEBUG, INFO, WARNING, or ERROR", "default: INFO");
    }
    if (logging.filePath.empty()) {
        errors.emplace_back("logging.file_path", "Log file path is required", "default: /data/logs/pipeline.log");
    }
    
    return errors;
}

std::string Config::toString() const {
    std::ostringstream oss;
    oss << "=== Full Configuration ===\n";
    oss << rtsp.toString();
    oss << cameraDetection.toString();
    oss << localDetection.toString();
    oss << tracker.toString();
    oss << refFrame.toString();
    oss << eventAnalysis.toString();
    oss << logging.toString();
    oss << "==========================\n";
    return oss.str();
}

std::string Config::toSummary() const {
    std::ostringstream oss;
    oss << "RTSP: " << rtsp.url << "\n";
    oss << "Detection: " << (cameraDetection.enabled ? "camera (" + cameraDetection.protocol + ")" : "local (" + localDetection.modelType + ")") << "\n";
    oss << "Tracker: " << (tracker.enabled ? "enabled" : "disabled") << "\n";
    oss << "Event mode: " << eventAnalysis.mode << "\n";
    oss << "Log: " << logging.filePath << " (" << logging.level << ")\n";
    return oss.str();
}

YAML::Node Config::toYaml() const {
    YAML::Node node;
    
    node["rtsp"]["url"] = rtsp.url;
    node["rtsp"]["reconnect_interval_ms"] = rtsp.reconnectIntervalMs;
    
    node["camera_detection"]["enabled"] = cameraDetection.enabled;
    node["camera_detection"]["protocol"] = cameraDetection.protocol;
    node["camera_detection"]["endpoint"] = cameraDetection.endpoint;
    node["camera_detection"]["poll_interval_ms"] = cameraDetection.pollIntervalMs;
    node["camera_detection"]["camera_id"] = cameraDetection.cameraId;
    node["camera_detection"]["camera_host"] = cameraDetection.cameraHost;
    node["camera_detection"]["camera_port"] = cameraDetection.cameraPort;
    node["camera_detection"]["capability_url"] = cameraDetection.capabilityUrl;
    node["camera_detection"]["timeout_ms"] = cameraDetection.timeoutMs;
    
    node["local_detection"]["model_path"] = localDetection.modelPath;
    node["local_detection"]["model_type"] = localDetection.modelType;
    node["local_detection"]["conf_threshold"] = localDetection.confThreshold;
    node["local_detection"]["detect_interval"] = localDetection.detectInterval;
    node["local_detection"]["timeout_ms"] = localDetection.timeoutMs;
    
    node["tracker"]["enabled"] = tracker.enabled;
    node["tracker"]["confirm_frames"] = tracker.confirmFrames;
    node["tracker"]["max_lost_frames"] = tracker.maxLostFrames;
    node["tracker"]["high_threshold"] = tracker.highThreshold;
    node["tracker"]["low_threshold"] = tracker.lowThreshold;
    node["tracker"]["match_threshold"] = tracker.matchThreshold;
    
    node["ref_frame"]["similarity_threshold"] = refFrame.similarityThreshold;
    node["ref_frame"]["compare_method"] = refFrame.compareMethod;
    node["ref_frame"]["update_strategy"] = refFrame.updateStrategy;
    node["ref_frame"]["compare_roi_only"] = refFrame.compareRoiOnly;
    
    node["event_analysis"]["mode"] = eventAnalysis.mode;
    node["event_analysis"]["video_duration_sec"] = eventAnalysis.videoDurationSec;
    
    node["logging"]["level"] = logging.level;
    node["logging"]["file_path"] = logging.filePath;
    
    return node;
}

void Config::saveToFile(const std::string& path) const {
    YAML::Node node = toYaml();
    
    std::ofstream file(path);
    if (!file.is_open()) {
        LOG_ERROR("Cannot open file for writing: " + path);
        return;
    }
    
    file << node;
    file.close();
    
    LOG_INFO("Config saved to: " + path);
}

void Config::setChangeCallback(ConfigChangeCallback callback) {
    changeCallback_ = callback;
}

}