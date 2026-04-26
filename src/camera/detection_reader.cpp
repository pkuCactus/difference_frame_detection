#include "camera/detection_reader.h"
#include "common/logger.h"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <chrono>
#include <thread>
#include <sstream>

namespace diff_det {

using json = nlohmann::json;

CameraDetectionReader::CameraDetectionReader()
    : lastFetchTime_(0)
    , lastMatchedFrameId_(-1)
    , lastMatchedTimestamp_(-1) {
}

CameraDetectionReader::~CameraDetectionReader() {
}

bool CameraDetectionReader::init(const CameraDetectionConfig& config) {
    config_ = config;
    resultQueue_.clear();
    
    LOG_INFO("CameraDetectionReader initialized: protocol=" + config.protocol +
             ", endpoint=" + config.endpoint +
             ", poll_interval=" + std::to_string(config.pollIntervalMs) + "ms");
    
    return true;
}

CameraDetectionResult CameraDetectionReader::getDetectionResult() {
    auto now = std::chrono::system_clock::now();
    int64_t currentTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    
    if (currentTime - lastFetchTime_ >= config_.pollIntervalMs) {
        CameraDetectionResult result;
        
        if (config_.protocol == "REST") {
            result = fetchByRest();
        } else if (config_.protocol == "ONVIF") {
            result = fetchByOnvif();
        }
        
        lastFetchTime_ = currentTime;
        
        if (result.objNum > 0) {
            resultQueue_.push_back(result);
            
            if (resultQueue_.size() > 100) {
                resultQueue_.pop_front();
            }
        }
        
        return result;
    }
    
    if (!resultQueue_.empty()) {
        return resultQueue_.front();
    }
    
    CameraDetectionResult emptyResult;
    return emptyResult;
}

CameraDetectionResult CameraDetectionReader::fetchByRest() {
    CameraDetectionResult result;
    
    std::string url = config_.endpoint;
    
    LOG_DEBUG("Fetching detection result from REST: " + url);
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR("Failed to initialize CURL for detection fetch");
        return result;
    }
    
    std::string responseString;
    
    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, 
                     +[](void* contents, size_t size, size_t nmemb, std::string* s) {
                         s->append((char*)contents, size * nmemb);
                         return size * nmemb;
                     });
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseString);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config_.timeoutMs);
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        LOG_WARN("REST detection fetch failed: " + std::string(curl_easy_strerror(res)));
        curl_easy_cleanup(curl);
        return result;
    }
    
    curl_easy_cleanup(curl);
    
    try {
        json j = json::parse(responseString);
        
        if (j.contains("obj_num")) {
            result.objNum = j["obj_num"].get<int>();
        }
        
        if (j.contains("frame_id")) {
            result.frameId = j["frame_id"].get<int>();
        }
        
        if (j.contains("time_stamp")) {
            result.timeStamp = j["time_stamp"].get<int64_t>();
        }
        
        if (j.contains("objs") && j["objs"].is_array()) {
            for (const auto& obj : j["objs"]) {
                BoundingBox bbox;
                
                if (obj.contains("bbox") && obj["bbox"].is_array() && 
                    obj["bbox"].size() >= 4) {
                    bbox.x1 = obj["bbox"][0].get<float>();
                    bbox.y1 = obj["bbox"][1].get<float>();
                    bbox.x2 = obj["bbox"][2].get<float>();
                    bbox.y2 = obj["bbox"][3].get<float>();
                }
                
                if (obj.contains("conf")) {
                    bbox.conf = obj["conf"].get<float>();
                }
                
                if (obj.contains("label")) {
                    bbox.label = obj["label"].get<int>();
                }
                
                if (bbox.label == 0) {
                    result.objs.push_back(bbox);
                }
            }
        }
        
        if (result.objNum > 0) {
            LOG_INFO("Received detection result: objNum=" + std::to_string(result.objNum) +
                     ", frameId=" + std::to_string(result.frameId) +
                     ", timestamp=" + std::to_string(result.timeStamp));
        }
        
    } catch (const json::parse_error& e) {
        LOG_ERROR("JSON parse failed for detection result: " + std::string(e.what()));
        return result;
    } catch (const std::exception& e) {
        LOG_ERROR("Exception parsing detection response: " + std::string(e.what()));
        return result;
    }
    
    return result;
}

CameraDetectionResult CameraDetectionReader::fetchByOnvif() {
    CameraDetectionResult result;
    
    LOG_DEBUG("Fetching detection result via ONVIF (stub)");
    
    return result;
}

bool CameraDetectionReader::matchFrame(int frameId, int64_t timestamp, 
                                         const CameraDetectionResult& result) {
    if (result.frameId == frameId || result.timeStamp == timestamp) {
        lastMatchedFrameId_ = frameId;
        lastMatchedTimestamp_ = timestamp;
        
        if (!resultQueue_.empty()) {
            resultQueue_.pop_front();
        }
        
        LOG_DEBUG("Frame matched: frameId=" + std::to_string(frameId));
        return true;
    }
    
    int64_t frameDiff = std::abs(result.frameId - frameId);
    int64_t timeDiff = std::abs(result.timeStamp - timestamp);
    
    int64_t maxFrameDiff = 5;
    int64_t maxTimeDiff = 200;
    
    if (frameDiff <= maxFrameDiff || timeDiff <= maxTimeDiff) {
        lastMatchedFrameId_ = frameId;
        lastMatchedTimestamp_ = timestamp;
        
        if (!resultQueue_.empty()) {
            resultQueue_.pop_front();
        }
        
        LOG_DEBUG("Frame matched with tolerance: frameDiff=" + std::to_string(frameDiff) +
                  ", timeDiff=" + std::to_string(timeDiff) + "ms");
        return true;
    }
    
    LOG_WARN("Frame match failed: expected frameId=" + std::to_string(frameId) +
             ", got frameId=" + std::to_string(result.frameId) +
             ", expected timestamp=" + std::to_string(timestamp) +
             ", got timestamp=" + std::to_string(result.timeStamp));
    
    return false;
}

}