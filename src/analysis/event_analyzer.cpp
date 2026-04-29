#include "analysis/event_analyzer.h"
#include "common/logger.h"
#include "common/visualization.h"
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <ctime>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

namespace diff_det {

namespace {

const char* kOutputDir = "outputs";

std::string EncodeBase64(const cv::Mat& frame) {
    std::vector<uint8_t> buf;
    std::vector<int> params = {cv::IMWRITE_JPEG_QUALITY, 90};
    if (!cv::imencode(".jpg", frame, buf, params)) {
        LOG_ERROR("Failed to encode frame to JPEG");
        return "";
    }
    static const char* base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string result;
    result.reserve(buf.size() * 4 / 3 + 4);
    int i = 0;
    int n = static_cast<int>(buf.size());
    while (i < n) {
        uint32_t octet_a = i < n ? buf[i++] : 0;
        uint32_t octet_b = i < n ? buf[i++] : 0;
        uint32_t octet_c = i < n ? buf[i++] : 0;
        uint32_t triple = (octet_a << 16) | (octet_b << 8) | octet_c;
        result += base64Chars[(triple >> 18) & 0x3F];
        result += base64Chars[(triple >> 12) & 0x3F];
        result += base64Chars[(triple >> 6) & 0x3F];
        result += base64Chars[triple & 0x3F];
    }
    int mod = n % 3;
    if (mod == 1) {
        result[result.size() - 1] = '=';
        result[result.size() - 2] = '=';
    } else if (mod == 2) {
        result[result.size() - 1] = '=';
    }
    return result;
}

std::string FormatTimestampISO(int64_t timestampMs) {
    time_t t = timestampMs / 1000;
    struct tm* tm_info = localtime(&t);
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%Y-%m-%dT%H:%M:%S");
    return oss.str();
}

bool SendWebhook(const std::string& url, const cv::Mat& frame, int64_t timestamp) {
    std::string base64Data = EncodeBase64(frame);
    if (base64Data.empty()) {
        return false;
    }
    std::string timestampStr = FormatTimestampISO(timestamp);
    nlohmann::json jsonData;
    jsonData["image_base64"] = base64Data;
    jsonData["timestamp"] = timestampStr;
    std::string jsonStr = jsonData.dump();
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR("Failed to init CURL for webhook");
        return false;
    }
    struct curl_slist* headers = nullptr;
    headers = curl_slist_append(headers, "Content-Type: application/json");
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, jsonStr.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 10L);
    CURLcode res = curl_easy_perform(curl);
    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);
    if (res != CURLE_OK) {
        LOG_ERROR("Webhook CURL failed: " + std::string(curl_easy_strerror(res)));
        return false;
    }
    LOG_INFO("Webhook sent successfully to: " + url + ", timestamp=" + timestampStr);
    return true;
}

void SaveFrame(const cv::Mat& frame, const std::string& filename) {
    try {
        std::filesystem::create_directories(kOutputDir);
        std::string path = std::string(kOutputDir) + "/" + filename;
        if (cv::imwrite(path, frame)) {
            LOG_INFO("Saved frame to: " + path);
        } else {
            LOG_ERROR("Failed to save frame: " + path);
        }
    } catch (const std::exception& e) {
        LOG_ERROR("Exception saving frame: " + std::string(e.what()));
    }
}

EventAnalysisMode ParseMode(const std::string& mode) {
    if (mode == "video") {
        return EventAnalysisMode::kVideo;
    }
    return EventAnalysisMode::kImage;
}

const char* ModeToString(EventAnalysisMode mode) {
    switch (mode) {
        case EventAnalysisMode::kImage: return "image";
        case EventAnalysisMode::kVideo: return "video";
    }
    return "image";
}

} // namespace

EventAnalyzer::EventAnalyzer(const EventAnalysisConfig& config)
    : mode_(ParseMode(config.mode))
    , videoDurationSec_(config.videoDurationSec)
    , webhookUrl_(config.webhookUrl)
    , webhookEnabled_(config.webhookEnabled)
    , saveImg_(config.saveImg)
    , withBox_(config.withBox)
    , videoBuffer_(nullptr)
    , eventCount_(0)
    , lastEventId_(0) {

    LOG_INFO("EventAnalyzer initialized: mode=" + std::string(ModeToString(mode_)) +
             ", videoDuration=" + std::to_string(videoDurationSec_) + "s" +
             ", webhook=" + (webhookEnabled_ ? webhookUrl_ : "disabled") +
             ", saveImg=" + (saveImg_ ? "true" : "false") +
             ", withBox=" + (withBox_ ? "true" : "false"));
}

EventAnalyzer::~EventAnalyzer() {
}

bool EventAnalyzer::ValidateBoxes(const std::vector<BoundingBox>& boxes) {
    if (boxes.empty()) {
        LOG_WARN("Event analysis received empty boxes, skipping");
        return false;
    }
    return true;
}

bool EventAnalyzer::ValidateInput(const cv::Mat& frame,
                                    const std::vector<BoundingBox>& boxes) {
    if (frame.empty()) {
        LOG_ERROR("Event analysis received empty frame");
        return false;
    }
    return ValidateBoxes(boxes);
}

bool EventAnalyzer::ValidateInput(const std::vector<cv::Mat>& frames,
                                    const std::vector<BoundingBox>& boxes) {
    if (frames.empty()) {
        LOG_ERROR("Event analysis received empty frames");
        return false;
    }
    return ValidateBoxes(boxes);
}

void EventAnalyzer::AnalyzeImage(const cv::Mat& frame,
                                   const std::vector<BoundingBox>& boxes) {
    if (!ValidateInput(frame, boxes)) {
        return;
    }

    std::string eventId = generateEventId();
    eventCount_++;

    cv::Mat annotatedFrame = frame.clone();
    if (withBox_) {
        DrawBoundingBoxes(annotatedFrame, boxes);
    }

    if (saveImg_) {
        SaveFrame(annotatedFrame, eventId + ".jpg");
    }

    int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    if (webhookEnabled_) {
        SendWebhook(webhookUrl_, annotatedFrame, timestamp);
    }

    LOG_INFO("Event analysis (image): eventId=" + eventId +
             ", boxes=" + std::to_string(boxes.size()) +
             ", totalEvents=" + std::to_string(eventCount_));

    if (callback_) {
        callback_(annotatedFrame, boxes, eventCount_, timestamp);
    }
}

void EventAnalyzer::AnalyzeVideo(const std::vector<cv::Mat>& frames,
                                   const std::vector<BoundingBox>& boxes) {
    if (!ValidateInput(frames, boxes)) {
        return;
    }

    std::string eventId = generateEventId();
    eventCount_++;

    if (saveImg_) {
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat frameToSave = frames[i].clone();
            if (withBox_) {
                DrawBoundingBoxes(frameToSave, boxes);
            }
            SaveFrame(frameToSave, eventId + "_frame_" + std::to_string(i) + ".jpg");
        }
    }

    int64_t timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    cv::Mat annotatedFrame = frames[0].clone();
    if (withBox_) {
        DrawBoundingBoxes(annotatedFrame, boxes);
    }

    if (webhookEnabled_) {
        SendWebhook(webhookUrl_, annotatedFrame, timestamp);
    }

    LOG_INFO("Event analysis (video): eventId=" + eventId +
             ", frames=" + std::to_string(frames.size()) +
             ", boxes=" + std::to_string(boxes.size()) +
             ", totalEvents=" + std::to_string(eventCount_));

    if (callback_) {
        for (size_t i = 0; i < frames.size(); ++i) {
            cv::Mat annotated = frames[i].clone();
            if (withBox_ && i == 0) {
                DrawBoundingBoxes(annotated, boxes);
            }
            callback_(annotated, boxes, static_cast<int>(i),
                      std::chrono::duration_cast<std::chrono::milliseconds>(
                          std::chrono::system_clock::now().time_since_epoch()).count());
        }
    }
}

void EventAnalyzer::setEventCallback(EventCallback callback) {
    callback_ = callback;
}

void EventAnalyzer::setVideoBuffer(std::deque<cv::Mat>* buffer) {
    videoBuffer_ = buffer;
}

std::string EventAnalyzer::generateEventId() {
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();

    std::ostringstream oss;
    oss << "event_" << std::setw(12) << std::setfill('0') << ms;

    return oss.str();
}

VideoBuffer::VideoBuffer(int maxSize)
    : maxSize_(maxSize) {
    LOG_INFO("VideoBuffer initialized: maxSize=" + std::to_string(maxSize_));
}

void VideoBuffer::addFrame(const cv::Mat& frame, int frameId, int64_t timestamp) {
    if (frame.empty()) {
        LOG_WARN("VideoBuffer received empty frame");
        return;
    }

    frames_.push_back(frame.clone());
    frameIds_.push_back(frameId);
    timestamps_.push_back(timestamp);

    while (frames_.size() > static_cast<size_t>(maxSize_)) {
        frames_.pop_front();
        frameIds_.pop_front();
        timestamps_.pop_front();
    }

    LOG_DEBUG("VideoBuffer: added frame " + std::to_string(frameId) +
              ", bufferSize=" + std::to_string(frames_.size()));
}

std::vector<cv::Mat> VideoBuffer::getFrames(int count) {
    int actualCount = std::min(count, static_cast<int>(frames_.size()));

    std::vector<cv::Mat> result;
    result.reserve(actualCount);

    auto it = frames_.begin() + (frames_.size() - actualCount);
    for (; it != frames_.end(); ++it) {
        result.push_back(*it);
    }

    LOG_DEBUG("VideoBuffer: retrieved " + std::to_string(result.size()) + " frames");

    return result;
}

std::vector<cv::Mat> VideoBuffer::getFramesByDuration(int durationSec, double fps) {
    int frameCount = static_cast<int>(durationSec * fps);
    return getFrames(frameCount);
}

void VideoBuffer::Clear() {
    frames_.clear();
    frameIds_.clear();
    timestamps_.clear();
    LOG_INFO("VideoBuffer cleared");
}

}