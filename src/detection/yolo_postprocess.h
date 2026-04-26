#pragma once

#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

namespace diff_det {

constexpr int MODEL_INPUT_WIDTH = 832;
constexpr int MODEL_INPUT_HEIGHT = 448;

class YoloPostprocess {
public:
    YoloPostprocess(const std::string& modelType, float confThreshold, float nmsThreshold = 0.45f);
    
    std::vector<BoundingBox> process(const std::vector<float>& outputs,
                                      int numOutputs,
                                      int outputWidth,
                                      int outputHeight,
                                      int originalWidth,
                                      int originalHeight,
                                      float scaleX,
                                      float scaleY,
                                      int offsetX,
                                      int offsetY);
    
    void setConfThreshold(float threshold) { confThreshold_ = threshold; }
    void setNmsThreshold(float threshold) { nmsThreshold_ = threshold; }
    
private:
    std::vector<BoundingBox> processYolov5(const std::vector<float>& outputs,
                                            int numOutputs,
                                            int originalWidth,
                                            int originalHeight,
                                            float scaleX,
                                            float scaleY,
                                            int offsetX,
                                            int offsetY);
    
    std::vector<BoundingBox> processYolov8(const std::vector<float>& outputs,
                                            int numOutputs,
                                            int originalWidth,
                                            int originalHeight,
                                            float scaleX,
                                            float scaleY,
                                            int offsetX,
                                            int offsetY);
    
    std::vector<BoundingBox> processYolov3(const std::vector<float>& outputs,
                                            int numOutputs,
                                            int originalWidth,
                                            int originalHeight,
                                            float scaleX,
                                            float scaleY,
                                            int offsetX,
                                            int offsetY);
    
    std::vector<BoundingBox> nms(std::vector<BoundingBox>& boxes);
    BoundingBox restoreBox(const BoundingBox& box,
                           float scaleX, float scaleY,
                           int offsetX, int offsetY);
    
    std::string modelType_;
    float confThreshold_;
    float nmsThreshold_;
    
    static constexpr int NUM_CLASSES = 80;
    static constexpr int PERSON_CLASS = 0;
};

}