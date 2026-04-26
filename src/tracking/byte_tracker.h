#pragma once

#include "tracker.h"
#include "common/config.h"
#include "common/types.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <map>
#include <deque>

namespace diff_det {

class KalmanFilter {
public:
    KalmanFilter();
    
    void init(const BoundingBox& bbox);
    void predict();
    void update(const BoundingBox& bbox);
    
    cv::Mat getState() const;
    BoundingBox getBox() const;
    
private:
    cv::Mat state_;
    cv::Mat covariance_;
    cv::Mat transitionMatrix_;
    cv::Mat measurementMatrix_;
    cv::Mat processNoiseCov_;
    cv::Mat measurementNoiseCov_;
};

class TrackObject {
public:
    TrackObject(const BoundingBox& bbox, int trackId, int frameId, int confirmFrames = 3);
    
    void predict();
    void update(const BoundingBox& bbox, int frameId);
    
    int getTrackId() const { return trackId_; }
    TrackState getState() const { return state_; }
    int getAge() const { return age_; }
    int getHitStreak() const { return hitStreak_; }
    int getFrameId() const { return frameId_; }
    BoundingBox getBox() const { return kalman_.getBox(); }
    float getScore() const { return score_; }
    
    void markLost();
    void markRemoved();
    void markTracked();
    
    void incrementAge() { age_++; }
    
private:
    int trackId_;
    TrackState state_;
    KalmanFilter kalman_;
    int age_;
    int hitStreak_;
    int frameId_;
    float score_;
    int confirmFrames_;
};

class ByteTracker : public ITracker {
public:
    ByteTracker(const TrackerConfig& config);
    ~ByteTracker();
    
    std::vector<Track> update(const cv::Mat& frame, 
                              const std::vector<BoundingBox>& boxes) override;
    std::vector<Track> predict() override;
    void reset() override;
    
private:
    std::vector<std::pair<int, int>> linearAssignment(const std::vector<std::vector<float>>& costMatrix);
    std::vector<std::vector<float>> computeIoUMatrix(const std::vector<BoundingBox>& boxesA,
                                                      const std::vector<BoundingBox>& boxesB);
    
    void associateDetectionsToTracks(const std::vector<BoundingBox>& detections,
                                      std::vector<TrackObject*>& tracks,
                                      std::vector<int>& matchedDetections,
                                      std::vector<int>& matchedTracks,
                                      std::vector<int>& unmatchedDetections,
                                      std::vector<int>& unmatchedTracks,
                                      float threshold);
    
    void secondAssociateDetectionsToTracks(const std::vector<BoundingBox>& detections,
                                            std::vector<TrackObject*>& tracks,
                                            std::vector<int>& matchedDetections,
                                            std::vector<int>& matchedTracks,
                                            std::vector<int>& unmatchedDetections,
                                            std::vector<int>& unmatchedTracks,
                                            float threshold);
    
    int generateNewTrackId();
    
    TrackerConfig config_;
    std::vector<std::unique_ptr<TrackObject>> tracks_;
    int nextTrackId_;
    int frameCount_;
    
    std::vector<Track> outputTracks_;
};

}