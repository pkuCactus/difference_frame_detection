#include "camera/capability_checker.h"
#include "common/logger.h"
#include <nlohmann/json.hpp>
#include <curl/curl.h>
#include <chrono>
#include <thread>

namespace diff_det {

using json = nlohmann::json;

CameraCapabilityChecker::CameraCapabilityChecker() : supported_(false) {
}

CameraCapabilityChecker::~CameraCapabilityChecker() {
}

CapabilityResult CameraCapabilityChecker::check(const CameraDetectionConfig& config) {
    CapabilityResult result;
    
    if (!config.enabled) {
        result.supported = false;
        result.reason = "Camera detection disabled in config";
        supported_ = false;
        LOG_INFO("camera_id=" + config.cameraId + 
                 ", detection_supported=false, reason=" + result.reason);
        return result;
    }
    
    if (config.protocol == "REST") {
        result = checkByRest(config);
    } else if (config.protocol == "ONVIF") {
        result = checkByOnvif(config);
    } else {
        result.supported = false;
        result.reason = "Unknown protocol: " + config.protocol;
    }
    
    supported_ = result.supported;
    
    LOG_INFO("camera_id=" + config.cameraId + 
             ", detection_supported=" + std::string(result.supported ? "true" : "false") +
             ", reason=" + result.reason);
    
    return result;
}

CapabilityResult CameraCapabilityChecker::checkByRest(const CameraDetectionConfig& config) {
    CapabilityResult result;
    
    std::string url = "http://" + config.cameraHost + ":" + 
                      std::to_string(config.cameraPort) + config.capabilityUrl;
    
    LOG_INFO("Checking capability via REST: " + url);
    
    CURL* curl = curl_easy_init();
    if (!curl) {
        result.supported = false;
        result.reason = "Failed to initialize CURL";
        LOG_ERROR(result.reason);
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
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, config.timeoutMs);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, config.timeoutMs / 2);
    
    CURLcode res = curl_easy_perform(curl);
    
    if (res != CURLE_OK) {
        result.supported = false;
        result.reason = "CURL error: " + std::string(curl_easy_strerror(res));
        LOG_WARN("REST capability check failed: " + result.reason);
        curl_easy_cleanup(curl);
        return result;
    }
    
    long httpCode = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
    curl_easy_cleanup(curl);
    
    if (httpCode != 200) {
        result.supported = false;
        result.reason = "HTTP response code: " + std::to_string(httpCode);
        LOG_WARN("REST capability check failed: " + result.reason);
        return result;
    }
    
    try {
        json j = json::parse(responseString);
        
        if (j.contains("supported")) {
            result.supported = j["supported"].get<bool>();
            result.reason = result.supported ? 
                "Camera supports detection via REST" : 
                "Camera reports no detection support";
        } else {
            result.supported = false;
            result.reason = "Response missing 'supported' field";
        }
        
    } catch (const json::parse_error& e) {
        result.supported = false;
        result.reason = "JSON parse failed: " + std::string(e.what());
        LOG_ERROR(result.reason);
        return result;
    } catch (const std::exception& e) {
        result.supported = false;
        result.reason = "Exception parsing response: " + std::string(e.what());
        LOG_ERROR(result.reason);
        return result;
    }
    
    return result;
}

CapabilityResult CameraCapabilityChecker::checkByOnvif(const CameraDetectionConfig& config) {
    CapabilityResult result;
    
    LOG_INFO("Checking capability via ONVIF (stub implementation)");
    
    result.supported = false;
    result.reason = "ONVIF capability check not implemented, using stub";
    
    return result;
}

bool CameraCapabilityChecker::isSupportDetection() {
    return supported_;
}

}