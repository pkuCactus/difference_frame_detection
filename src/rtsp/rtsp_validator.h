#pragma once

#include "rtsp/rtsp_client.h"
#include <string>

namespace diff_det {

struct RtspValidationResult {
    bool success = false;
    int framesReceived = 0;
    double fps = 0.0;
    int width = 0;
    int height = 0;
    std::string errorMessage;
};

class RtspValidator {
public:
    static RtspValidationResult Validate(IRtspClient* client, const std::string& url, int durationSec);
    static RtspValidationResult ValidateWithVisualization(IRtspClient* client, const std::string& url, int durationSec);
};

} // namespace diff_det
