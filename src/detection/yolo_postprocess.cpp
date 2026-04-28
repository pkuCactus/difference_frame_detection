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

    std::cout << "[POST] ProcessYolov5 raw boxes before NMS: " << boxes.size() << std::endl;
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

    std::cout << "[POST] ProcessYolov8 raw boxes before NMS: " << boxes.size() << std::endl;
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

std::vector<BoundingBox> YoloPostprocess::ProcessRknnYolov5(
    const std::vector<std::vector<float>>& outputs,
    int originalWidth,
    int originalHeight,
    float scaleX,
    float scaleY,
    int offsetX,
    int offsetY) {

    std::vector<BoundingBox> boxes;

    const std::vector<std::vector<std::pair<float, float>>> anchors = {
        {{10, 13}, {16, 30}, {33, 23}},
        {{30, 61}, {62, 45}, {59, 119}},
        {{116, 90}, {156, 198}, {373, 326}}
    };
    const std::vector<int32_t> strides = {8, 16, 32};

    for (size_t i = 0; i < outputs.size(); ++i) {
        const auto& output = outputs[i];
        if (output.empty()) {
            continue;
        }

        int32_t gridSize = static_cast<int32_t>(std::sqrt(output.size() / 255));
        std::cout << "[POST] Layer " << i << ": output.size=" << output.size()
                  << ", gridSize=" << gridSize << std::endl;
        if (gridSize * gridSize * 255 != static_cast<int32_t>(output.size())) {
            std::cerr << "[POST ERROR] Unexpected output size for layer " << i
                      << ": expected " << gridSize * gridSize * 255
                      << ", got " << output.size() << std::endl;
            continue;
        }

        int32_t stride = 0;
        if (gridSize == 80 || gridSize == 52) stride = 8;
        else if (gridSize == 40 || gridSize == 26) stride = 16;
        else if (gridSize == 20 || gridSize == 13) stride = 32;
        else {
            LOG_WARN("Unknown grid size " + std::to_string(gridSize) + " for layer " + std::to_string(i));
            continue;
        }

        int32_t anchorIdx = 0;
        if (stride == 8) anchorIdx = 0;
        else if (stride == 16) anchorIdx = 1;
        else anchorIdx = 2;

        int32_t gridPixels = gridSize * gridSize;

        for (int32_t gy = 0; gy < gridSize; ++gy) {
            for (int32_t gx = 0; gx < gridSize; ++gx) {
                for (int32_t a = 0; a < 3; ++a) {
                    int32_t pixelIdx = gy * gridSize + gx;
                    int32_t baseC = a * 85;
                    float x = output[(baseC + 0) * gridPixels + pixelIdx];
                    float y = output[(baseC + 1) * gridPixels + pixelIdx];
                    float w = output[(baseC + 2) * gridPixels + pixelIdx];
                    float h = output[(baseC + 3) * gridPixels + pixelIdx];
                    float objConf = output[(baseC + 4) * gridPixels + pixelIdx];

                    x = 1.0f / (1.0f + std::exp(-x));
                    y = 1.0f / (1.0f + std::exp(-y));
                    w = 1.0f / (1.0f + std::exp(-w));
                    h = 1.0f / (1.0f + std::exp(-h));
                    objConf = 1.0f / (1.0f + std::exp(-objConf));

                    int classIdx = 0;
                    float maxClassConf = 0.0f;
                    for (int c = 0; c < NUM_CLASSES; ++c) {
                        float classConf = 1.0f / (1.0f + std::exp(-output[(baseC + 5 + c) * gridPixels + pixelIdx]));
                        if (classConf > maxClassConf) {
                            maxClassConf = classConf;
                            classIdx = c;
                        }
                    }

                    float finalConf = objConf * maxClassConf;

                    if (finalConf < confThreshold_ || classIdx != PERSON_CLASS) {
                        continue;
                    }

                    float anchorW = anchors[anchorIdx][a].first;
                    float anchorH = anchors[anchorIdx][a].second;
                    float strideF = static_cast<float>(stride);

                    float cx = (x * 2.0f - 0.5f + static_cast<float>(gx)) * strideF;
                    float cy = (y * 2.0f - 0.5f + static_cast<float>(gy)) * strideF;
                    float bw = std::pow(w * 2.0f, 2) * anchorW;
                    float bh = std::pow(h * 2.0f, 2) * anchorH;

                    BoundingBox box;
                    box.x1 = cx - bw / 2.0f;
                    box.y1 = cy - bh / 2.0f;
                    box.x2 = cx + bw / 2.0f;
                    box.y2 = cy + bh / 2.0f;
                    box.conf = finalConf;
                    box.label = classIdx;

                    box = RestoreBox(box, scaleX, scaleY, offsetX, offsetY);
                    boxes.push_back(box);
                }
            }
        }
    }

    std::cout << "[POST] ProcessRknnYolov5 raw boxes before NMS: " << boxes.size() << std::endl;
    return nms(boxes);
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