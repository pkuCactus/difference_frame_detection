#include "rtsp/rtsp_validator.h"
#include "common/logger.h"
#include <chrono>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <thread>

namespace diff_det {

RtspValidationResult RtspValidator::Validate(IRtspClient* client, const std::string& url, int durationSec) {
    RtspValidationResult result;

    if (!client) {
        result.errorMessage = "RTSP client is null";
        return result;
    }

    LOG_INFO("开始RTSP收流验证: " + url);
    std::cout << "开始RTSP收流验证: " << url << std::endl;

    if (!client->Connect(url)) {
        result.errorMessage = "连接RTSP失败: " + url;
        LOG_ERROR(result.errorMessage);
        std::cout << "错误: " << result.errorMessage << std::endl;
        return result;
    }

    result.width = static_cast<int>(client->GetWidth());
    result.height = static_cast<int>(client->GetHeight());
    result.fps = client->GetFps();

    std::cout << "连接成功: " << result.width << "x" << result.height
              << ", FPS=" << result.fps << std::endl;

    auto startTime = std::chrono::steady_clock::now();
    cv::Mat frame;
    int frameId = 0;
    int64_t timestamp = 0;
    int consecutiveFailures = 0;
    const int maxConsecutiveFailures = 10;

    while (true) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime).count();
        if (durationSec > 0 && elapsed >= durationSec) {
            break;
        }

        if (!client->GetFrame(frame, frameId, timestamp)) {
            consecutiveFailures++;
            LOG_WARN("读取帧失败, 连续失败次数: " + std::to_string(consecutiveFailures));
            if (consecutiveFailures >= maxConsecutiveFailures) {
                result.errorMessage = "连续读取帧失败超过" + std::to_string(maxConsecutiveFailures) + "次";
                LOG_ERROR(result.errorMessage);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        consecutiveFailures = 0;
        result.framesReceived++;

        if (result.framesReceived % 30 == 0) {
            auto now = std::chrono::steady_clock::now();
            auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count();
            double actualFps = (totalMs > 0) ? (result.framesReceived * 1000.0 / totalMs) : 0.0;
            std::cout << "已接收 " << result.framesReceived << " 帧, 实际FPS="
                      << std::fixed << std::setprecision(1) << actualFps << std::endl;
        }

        if (durationSec <= 0) {
            break;
        }
    }

    client->Disconnect();

    if (result.framesReceived > 0) {
        result.success = true;
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count();
        if (totalMs > 0) {
            result.fps = result.framesReceived * 1000.0 / totalMs;
        }
        std::cout << "验证完成: 共接收 " << result.framesReceived << " 帧"
                  << ", 平均FPS=" << std::fixed << std::setprecision(1) << result.fps << std::endl;
    } else if (result.errorMessage.empty()) {
        result.errorMessage = "未收到任何帧";
        std::cout << "错误: " << result.errorMessage << std::endl;
    }

    return result;
}

RtspValidationResult RtspValidator::ValidateWithVisualization(IRtspClient* client, const std::string& url, int durationSec) {
    RtspValidationResult result;

    if (!client) {
        result.errorMessage = "RTSP client is null";
        return result;
    }

    LOG_INFO("开始RTSP收流验证(可视化): " + url);
    std::cout << "开始RTSP收流验证(可视化): " << url << std::endl;
    std::cout << "按 'q' 键退出, 按 's' 键保存当前帧" << std::endl;

    if (!client->Connect(url)) {
        result.errorMessage = "连接RTSP失败: " + url;
        LOG_ERROR(result.errorMessage);
        std::cout << "错误: " << result.errorMessage << std::endl;
        return result;
    }

    result.width = static_cast<int>(client->GetWidth());
    result.height = static_cast<int>(client->GetHeight());
    result.fps = client->GetFps();

    std::cout << "连接成功: " << result.width << "x" << result.height
              << ", FPS=" << result.fps << std::endl;

    auto startTime = std::chrono::steady_clock::now();
    cv::Mat frame;
    int frameId = 0;
    int64_t timestamp = 0;
    int savedCount = 0;
    int consecutiveFailures = 0;
    const int maxConsecutiveFailures = 10;

    while (true) {
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - startTime).count();
        if (durationSec > 0 && elapsed >= durationSec) {
            break;
        }

        if (!client->GetFrame(frame, frameId, timestamp)) {
            consecutiveFailures++;
            LOG_WARN("读取帧失败, 连续失败次数: " + std::to_string(consecutiveFailures));
            if (consecutiveFailures >= maxConsecutiveFailures) {
                result.errorMessage = "连续读取帧失败超过" + std::to_string(maxConsecutiveFailures) + "次";
                LOG_ERROR(result.errorMessage);
                break;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        consecutiveFailures = 0;
        result.framesReceived++;

        cv::imshow("RTSP Stream", frame);
        int key = cv::waitKey(1) & 0xFF;
        if (key == 'q') {
            std::cout << "用户主动退出" << std::endl;
            break;
        }
        if (key == 's') {
            std::string filename = "/tmp/rtsp_frame_" + std::to_string(frameId) + ".jpg";
            cv::imwrite(filename, frame);
            savedCount++;
            std::cout << "已保存帧: " << filename << std::endl;
        }
    }

    cv::destroyWindow("RTSP Stream");
    client->Disconnect();

    if (result.framesReceived > 0) {
        result.success = true;
        auto totalMs = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count();
        if (totalMs > 0) {
            result.fps = result.framesReceived * 1000.0 / totalMs;
        }
        std::cout << "验证完成: 共接收 " << result.framesReceived << " 帧"
                  << ", 保存 " << savedCount << " 帧"
                  << ", 平均FPS=" << std::fixed << std::setprecision(1) << result.fps << std::endl;
    } else if (result.errorMessage.empty()) {
        result.errorMessage = "未收到任何帧";
        std::cout << "错误: " << result.errorMessage << std::endl;
    }

    return result;
}

} // namespace diff_det
