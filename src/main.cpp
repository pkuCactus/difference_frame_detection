#include "core/pipeline.h"
#include "common/config.h"
#include "common/logger.h"
#include "rtsp/rtsp_validator.h"
#include "rtsp/rtsp_client.h"

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
        g_pipeline->Stop();
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

void printUsage(const char* programName) {
    std::cout << "Usage: " << programName << " [options] [config.yaml]" << std::endl;
    std::cout << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --test-rtsp           单独验证RTSP收流" << std::endl;
    std::cout << "  --visualize-rtsp      验证RTSP收流并可视化显示" << std::endl;
    std::cout << "  --rtsp-url <url>      指定RTSP流地址(覆盖配置文件)" << std::endl;
    std::cout << "  --test-duration <sec> 测试持续时间,默认10秒,0表示只收一帧" << std::endl;
    std::cout << "  -h, --help            显示帮助信息" << std::endl;
    std::cout << std::endl;
    std::cout << "Examples:" << std::endl;
    std::cout << "  " << programName << " config/config.yaml" << std::endl;
    std::cout << "  " << programName << " --test-rtsp --rtsp-url rtsp://localhost:8554/stream1" << std::endl;
    std::cout << "  " << programName << " --visualize-rtsp --rtsp-url rtsp://localhost:8554/stream1 --test-duration 30" << std::endl;
}

struct CmdLineArgs {
    std::string configPath = "config/config.yaml";
    std::string rtspUrl;
    bool testRtsp = false;
    bool visualizeRtsp = false;
    int testDuration = 10;
    bool showHelp = false;
};

CmdLineArgs parseArgs(int argc, char* argv[]) {
    CmdLineArgs args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test-rtsp") {
            args.testRtsp = true;
        } else if (arg == "--visualize-rtsp") {
            args.visualizeRtsp = true;
        } else if (arg == "--rtsp-url" && i + 1 < argc) {
            args.rtspUrl = argv[++i];
        } else if (arg == "--test-duration" && i + 1 < argc) {
            args.testDuration = std::atoi(argv[++i]);
        } else if (arg == "-h" || arg == "--help") {
            args.showHelp = true;
        } else if (arg[0] != '-') {
            args.configPath = arg;
        }
    }
    return args;
}

int runPipeline(const CmdLineArgs& args) {
    try {
        Config config = Config::FromFile(args.configPath);

        std::cout << "========================================" << std::endl;
        std::cout << "Difference Detection Pipeline v1.0" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Config file: " << args.configPath << std::endl;
        std::cout << "RTSP URL: " << config.rtsp.url << std::endl;
        std::cout << "Detection mode: " << (config.cameraDetection.enabled ? "camera" : "local") << std::endl;
        std::cout << "Tracker: " << (config.tracker.enabled ? "enabled" : "disabled") << std::endl;
        std::cout << "Event mode: " << config.eventAnalysis.mode << std::endl;
        std::cout << "Ref Update strategy: " << config.refFrame.updateStrategy << std::endl;
        std::cout << "Similarity method: " << config.refFrame.compareMethod << std::endl;
        std::cout << "Similarity threshold: " << config.refFrame.similarityThreshold << std::endl;
        std::cout << "Detect interval: " << config.localDetection.detectInterval << " frames" << std::endl;
        std::cout << "Log file: " << config.logging.filePath << std::endl;
        std::cout << "========================================" << std::endl;

        Pipeline pipeline(config);
        g_pipeline = &pipeline;
        pipeline.Start();

        int statsInterval = 10;
        auto lastStatsTime = std::chrono::system_clock::now();
        int prevFrameCount = 0;

        while (g_running && pipeline.IsRunning()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));

            State state = pipeline.GetCurrentState();
            int frameCount = pipeline.GetFrameCount();
            int eventCount = pipeline.GetEventCount();

            std::cout << "[" << StateToString(state) << "] "
                      << "Frames: " << frameCount
                      << ", Events: " << eventCount
                      << ", FPS: " << (frameCount - prevFrameCount)
                      << std::endl;

            prevFrameCount = frameCount;

            auto now = std::chrono::system_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - lastStatsTime);

            if (elapsed.count() >= statsInterval) {
                printStats(pipeline.GetStats(), statsInterval);
                lastStatsTime = now;
            }
        }

        pipeline.Stop();
        g_pipeline = nullptr;

        std::cout << "\n========================================" << std::endl;
        std::cout << "Pipeline Final Statistics:" << std::endl;
        printStats(pipeline.GetStats(), 0);
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

int runRtspTest(const CmdLineArgs& args) {
    std::string url = args.rtspUrl;
    if (url.empty()) {
        try {
            Config config = Config::FromFile(args.configPath);
            url = config.rtsp.url;
        } catch (const std::exception& e) {
            std::cerr << "无法读取配置文件获取RTSP地址: " << e.what() << std::endl;
            std::cerr << "请使用 --rtsp-url 指定RTSP地址" << std::endl;
            return 1;
        }
    }

    if (url.empty()) {
        std::cerr << "错误: 未指定RTSP地址" << std::endl;
        return 1;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "RTSP Stream Verification" << std::endl;
    std::cout << "========================================" << std::endl;

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    RtspClient client;
    RtspValidationResult result;

    if (args.visualizeRtsp) {
        result = RtspValidator::ValidateWithVisualization(&client, url, args.testDuration);
    } else {
        result = RtspValidator::Validate(&client, url, args.testDuration);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "验证结果:" << std::endl;
    std::cout << "  状态: " << (result.success ? "成功" : "失败") << std::endl;
    std::cout << "  接收帧数: " << result.framesReceived << std::endl;
    std::cout << "  分辨率: " << result.width << "x" << result.height << std::endl;
    std::cout << "  帧率: " << std::fixed << std::setprecision(1) << result.fps << std::endl;
    if (!result.errorMessage.empty()) {
        std::cout << "  错误: " << result.errorMessage << std::endl;
    }
    std::cout << "========================================" << std::endl;

    return result.success ? 0 : 1;
}

int main(int argc, char* argv[]) {
    CmdLineArgs args = parseArgs(argc, argv);

    if (args.showHelp) {
        printUsage(argv[0]);
        return 0;
    }

    if (args.testRtsp || args.visualizeRtsp) {
        return runRtspTest(args);
    }

    return runPipeline(args);
}
