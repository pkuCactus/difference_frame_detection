#pragma once

#include "common/config.h"
#include <string>

namespace diff_det {

struct CapabilityResult {
    bool supported;
    std::string reason;
    
    CapabilityResult() : supported(false), reason("") {}
};

class ICameraCapabilityChecker {
public:
    virtual ~ICameraCapabilityChecker() = default;
    
    virtual CapabilityResult check(const CameraDetectionConfig& config) = 0;
    virtual bool isSupportDetection() = 0;
};

class CameraCapabilityChecker : public ICameraCapabilityChecker {
public:
    CameraCapabilityChecker();
    ~CameraCapabilityChecker();
    
    CapabilityResult check(const CameraDetectionConfig& config) override;
    bool isSupportDetection() override;
    
private:
    CapabilityResult checkByRest(const CameraDetectionConfig& config);
    CapabilityResult checkByOnvif(const CameraDetectionConfig& config);
    
    bool supported_;
};

}