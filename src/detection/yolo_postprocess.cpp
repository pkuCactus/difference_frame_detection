#include "detection/yolo_postprocess.h"
#include "common/logger.h"
#include <algorithm>

namespace diff_det {

YoloPostprocess::YoloPostprocess(const std::string& modelType, float confThreshold, float nmsThreshold)
    : modelType_(modelType)
    , confThreshold_(confThreshold)
    , nmsThreshold_(nmsThreshold) {
}

std::vector<BoundingBox> YoloPostprocess::Process(const std::vector<float>& outputs,
                                                    int numOutputs,
                                                    int outputWidth,
                                                    int outputHeight,
                                                    int originalWidth,
                                                    int originalHeight,
                                                    float scaleX,
                                                    float scaleY,
                                                    int offsetX,
                                                    int offsetY) {
    if (modelType_ == "yolov5" || modelType_ == "yolov3") {
        return ProcessYolov5(outputs, numOutputs, originalWidth, originalHeight,
                             scaleX, scaleY, offsetX, offsetY);
    } else if (modelType_ == "yolov8") {
        return ProcessYolov8(outputs, numOutputs, originalWidth, originalHeight,
                             scaleX, scaleY, offsetX, offsetY);
    }
    
    LOG_ERROR("Unknown model type: " + modelType_);
    return {};
}

std::vector<BoundingBox> YoloPostprocess::ProcessYolov5(const std::vector<float>& outputs,
                                                         int numOutputs,
                                                         int originalWidth,
                                                         int originalHeight,
                                                         float scaleX,
                                                         float scaleY,
                                                         int offsetX,
                                                         int offsetY) {
    std::vector<BoundingBox> boxes;
    
    int numAnchors = numOutputs / (5 + NUM_CLASSES);
    
    for (int i = 0; i < numAnchors; ++i) {
        int baseIdx = i * (5 + NUM_CLASSES);
        
        float objConf = outputs[baseIdx + 4];
        
        int classIdx = 0;
        float maxClassConf = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float classConf = outputs[baseIdx + 5 + c];
            if (classConf > maxClassConf) {
                maxClassConf = classConf;
                classIdx = c;
            }
        }
        
        float finalConf = objConf * maxClassConf;
        
        if (finalConf < confThreshold_ || classIdx != PERSON_CLASS) {
            continue;
        }
        
        float cx = outputs[baseIdx + 0];
        float cy = outputs[baseIdx + 1];
        float w = outputs[baseIdx + 2];
        float h = outputs[baseIdx + 3];
        
        BoundingBox box;
        box.x1 = cx - w / 2.0f;
        box.y1 = cy - h / 2.0f;
        box.x2 = cx + w / 2.0f;
        box.y2 = cy + h / 2.0f;
        box.conf = finalConf;
        box.label = classIdx;
        
        box = RestoreBox(box, scaleX, scaleY, offsetX, offsetY);
        boxes.push_back(box);
    }
    
    return nms(boxes);
}

std::vector<BoundingBox> YoloPostprocess::ProcessYolov8(const std::vector<float>& outputs,
                                                         int numOutputs,
                                                         int originalWidth,
                                                         int originalHeight,
                                                         float scaleX,
                                                         float scaleY,
                                                         int offsetX,
                                                         int offsetY) {
    std::vector<BoundingBox> boxes;
    
    int numAnchors = numOutputs / (4 + NUM_CLASSES);
    
    for (int i = 0; i < numAnchors; ++i) {
        int baseIdx = i * (4 + NUM_CLASSES);
        
        int classIdx = 0;
        float maxClassConf = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float classConf = outputs[baseIdx + 4 + c];
            if (classConf > maxClassConf) {
                maxClassConf = classConf;
                classIdx = c;
            }
        }
        
        if (maxClassConf < confThreshold_ || classIdx != PERSON_CLASS) {
            continue;
        }
        
        float cx = outputs[baseIdx + 0];
        float cy = outputs[baseIdx + 1];
        float w = outputs[baseIdx + 2];
        float h = outputs[baseIdx + 3];
        
        BoundingBox box;
        box.x1 = cx - w / 2.0f;
        box.y1 = cy - h / 2.0f;
        box.x2 = cx + w / 2.0f;
        box.y2 = cy + h / 2.0f;
        box.conf = maxClassConf;
        box.label = classIdx;
        
        box = RestoreBox(box, scaleX, scaleY, offsetX, offsetY);
        boxes.push_back(box);
    }
    
    return nms(boxes);
}

std::vector<BoundingBox> YoloPostprocess::ProcessYolov3(const std::vector<float>& outputs,
                                                         int numOutputs,
                                                         int originalWidth,
                                                         int originalHeight,
                                                         float scaleX,
                                                         float scaleY,
                                                         int offsetX,
                                                         int offsetY) {
    return ProcessYolov5(outputs, numOutputs, originalWidth, originalHeight,
                         scaleX, scaleY, offsetX, offsetY);
}

std::vector<BoundingBox> YoloPostprocess::nms(std::vector<BoundingBox>& boxes) {
    if (boxes.empty()) {
        return boxes;
    }
    
    std::sort(boxes.begin(), boxes.end(), [](const BoundingBox& a, const BoundingBox& b) {
        return a.conf > b.conf;
    });
    
    std::vector<BoundingBox> result;
    std::vector<bool> suppressed(boxes.size(), false);
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) {
            continue;
        }
        
        result.push_back(boxes[i]);
        
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) {
                continue;
            }
            
            float x1 = std::max(boxes[i].x1, boxes[j].x1);
            float y1 = std::max(boxes[i].y1, boxes[j].y1);
            float x2 = std::min(boxes[i].x2, boxes[j].x2);
            float y2 = std::min(boxes[i].y2, boxes[j].y2);
            
            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float areaI = boxes[i].Area();
            float areaJ = boxes[j].Area();
            float unionArea = areaI + areaJ - intersection;
            
            float iou = intersection / unionArea;
            
            if (iou > nmsThreshold_) {
                suppressed[j] = true;
            }
        }
    }
    
    return result;
}

BoundingBox YoloPostprocess::RestoreBox(const BoundingBox& box,
                                          float scaleX, float scaleY,
                                          int offsetX, int offsetY) {
    BoundingBox restored;
    restored.x1 = (box.x1 - offsetX) / scaleX;
    restored.y1 = (box.y1 - offsetY) / scaleY;
    restored.x2 = (box.x2 - offsetX) / scaleX;
    restored.y2 = (box.y2 - offsetY) / scaleY;
    restored.conf = box.conf;
    restored.label = box.label;
    
    restored.x1 = std::max(0.0f, restored.x1);
    restored.y1 = std::max(0.0f, restored.y1);
    
    return restored;
}

}