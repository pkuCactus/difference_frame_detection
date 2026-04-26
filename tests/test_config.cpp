#include <gtest/gtest.h>
#include "common/config.h"
#include <yaml-cpp/yaml.h>
#include <fstream>

using namespace diff_det;

TEST(ConfigValidationTest, ValidConfig) {
    Config config;
    config.rtsp.url = "rtsp://192.168.1.100:554/stream";
    
    auto errors = config.validate();
    EXPECT_TRUE(errors.empty());
}

TEST(ConfigValidationTest, MissingRtspUrl) {
    Config config;
    config.rtsp.url = "";
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    
    bool foundUrlError = false;
    for (auto& err : errors) {
        if (err.field == "rtsp.url") {
            foundUrlError = true;
            EXPECT_TRUE(err.message.find("required") != std::string::npos);
        }
    }
    EXPECT_TRUE(foundUrlError);
}

TEST(ConfigValidationTest, InvalidModelType) {
    Config config;
    config.rtsp.url = "rtsp://test";
    config.localDetection.modelType = "invalid_model";
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
    
    bool foundModelError = false;
    for (auto& err : errors) {
        if (err.field == "local_detection.model_type") {
            foundModelError = true;
        }
    }
    EXPECT_TRUE(foundModelError);
}

TEST(ConfigValidationTest, InvalidThreshold) {
    Config config;
    config.rtsp.url = "rtsp://test";
    config.localDetection.confThreshold = 1.5f;
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(ConfigValidationTest, InvalidCompareMethod) {
    Config config;
    config.rtsp.url = "rtsp://test";
    config.refFrame.compareMethod = "invalid";
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(ConfigValidationTest, InvalidLogLevel) {
    Config config;
    config.rtsp.url = "rtsp://test";
    config.logging.level = "INVALID";
    
    auto errors = config.validate();
    EXPECT_FALSE(errors.empty());
}

TEST(ConfigSerializationTest, ToString) {
    Config config;
    config.rtsp.url = "rtsp://test";
    
    std::string str = config.toString();
    EXPECT_TRUE(str.find("RtspConfig") != std::string::npos);
    EXPECT_TRUE(str.find("rtsp://test") != std::string::npos);
}

TEST(ConfigSerializationTest, ToSummary) {
    Config config;
    config.rtsp.url = "rtsp://test";
    config.cameraDetection.enabled = true;
    config.tracker.enabled = true;
    
    std::string summary = config.toSummary();
    EXPECT_TRUE(summary.find("rtsp://test") != std::string::npos);
    EXPECT_TRUE(summary.find("camera") != std::string::npos);
    EXPECT_TRUE(summary.find("Tracker: enabled") != std::string::npos);
}

TEST(ConfigSerializationTest, ToYaml) {
    Config config;
    config.rtsp.url = "rtsp://test";
    config.tracker.enabled = true;
    
    YAML::Node node = config.toYaml();
    EXPECT_EQ(node["rtsp"]["url"].as<std::string>(), "rtsp://test");
    EXPECT_EQ(node["tracker"]["enabled"].as<bool>(), true);
}

TEST(ConfigSerializationTest, SaveToFile) {
    Config config;
    config.rtsp.url = "rtsp://test";
    
    std::string path = "/tmp/test_config.yaml";
    config.saveToFile(path);
    
    std::ifstream file(path);
    EXPECT_TRUE(file.is_open());
    file.close();
    
    Config loaded = Config::fromFile(path);
    EXPECT_EQ(loaded.rtsp.url, "rtsp://test");
}

TEST(ConfigDefaultsTest, AllDefaults) {
    Config config;
    
    EXPECT_EQ(config.rtsp.reconnectIntervalMs, 3000);
    EXPECT_EQ(config.cameraDetection.enabled, true);
    EXPECT_EQ(config.cameraDetection.protocol, "REST");
    EXPECT_EQ(config.localDetection.modelType, "yolov8");
    EXPECT_EQ(config.localDetection.confThreshold, 0.5f);
    EXPECT_EQ(config.tracker.enabled, false);
    EXPECT_EQ(config.refFrame.similarityThreshold, 0.85f);
    EXPECT_EQ(config.refFrame.compareMethod, "ssim");
    EXPECT_EQ(config.eventAnalysis.mode, "image");
    EXPECT_EQ(config.logging.level, "INFO");
}