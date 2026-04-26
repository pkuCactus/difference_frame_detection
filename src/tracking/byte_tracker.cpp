#include "tracking/byte_tracker.h"
#include "common/logger.h"
#include <algorithm>

namespace diff_det {

KalmanFilter::KalmanFilter() {
    const int stateSize = 8;
    const int measSize = 4;
    
    state_ = cv::Mat(stateSize, 1, CV_32F, cv::Scalar(0));
    covariance_ = cv::Mat(stateSize, stateSize, CV_32F, cv::Scalar(0));
    
    transitionMatrix_ = cv::Mat::eye(stateSize, stateSize, CV_32F);
    transitionMatrix_.at<float>(0, 4) = 1;
    transitionMatrix_.at<float>(1, 5) = 1;
    transitionMatrix_.at<float>(2, 6) = 1;
    transitionMatrix_.at<float>(3, 7) = 1;
    
    measurementMatrix_ = cv::Mat::zeros(measSize, stateSize, CV_32F);
    measurementMatrix_.at<float>(0, 0) = 1;
    measurementMatrix_.at<float>(1, 1) = 1;
    measurementMatrix_.at<float>(2, 2) = 1;
    measurementMatrix_.at<float>(3, 3) = 1;
    
    processNoiseCov_ = cv::Mat::eye(stateSize, stateSize, CV_32F) * 1e-2;
    measurementNoiseCov_ = cv::Mat::eye(measSize, measSize, CV_32F) * 1e-1;
    
    for (int i = 4; i < stateSize; ++i) {
        processNoiseCov_.at<float>(i, i) *= 1e-4;
    }
}

void KalmanFilter::Init(const BoundingBox& bbox) {
    float cx = (bbox.x1 + bbox.x2) / 2.0f;
    float cy = (bbox.y1 + bbox.y2) / 2.0f;
    float w = bbox.x2 - bbox.x1;
    float h = bbox.y2 - bbox.y1;
    
    state_.at<float>(0) = cx;
    state_.at<float>(1) = cy;
    state_.at<float>(2) = w;
    state_.at<float>(3) = h;
    state_.at<float>(4) = 0;
    state_.at<float>(5) = 0;
    state_.at<float>(6) = 0;
    state_.at<float>(7) = 0;
    
    covariance_ = cv::Mat::eye(state_.rows, covariance_.rows, CV_32F) * 10;
}

void KalmanFilter::Predict() {
    state_ = transitionMatrix_ * state_;
    
    cv::Mat Q = transitionMatrix_ * covariance_ * transitionMatrix_.t() + processNoiseCov_;
    covariance_ = Q;
    
    float w = state_.at<float>(2);
    float h = state_.at<float>(3);
    state_.at<float>(2) = std::max(1.0f, w);
    state_.at<float>(3) = std::max(1.0f, h);
}

void KalmanFilter::Update(const BoundingBox& bbox) {
    float cx = (bbox.x1 + bbox.x2) / 2.0f;
    float cy = (bbox.y1 + bbox.y2) / 2.0f;
    float w = bbox.x2 - bbox.x1;
    float h = bbox.y2 - bbox.y1;
    
    cv::Mat measurement(4, 1, CV_32F);
    measurement.at<float>(0) = cx;
    measurement.at<float>(1) = cy;
    measurement.at<float>(2) = w;
    measurement.at<float>(3) = h;
    
    cv::Mat H = measurementMatrix_;
    cv::Mat K = covariance_ * H.t() * (H * covariance_ * H.t() + measurementNoiseCov_).inv();
    
    state_ = state_ + K * (measurement - H * state_);
    covariance_ = (cv::Mat::eye(covariance_.rows, covariance_.rows, CV_32F) - K * H) * covariance_;
}

cv::Mat KalmanFilter::getState() const {
    return state_;
}

BoundingBox KalmanFilter::getBox() const {
    float cx = state_.at<float>(0);
    float cy = state_.at<float>(1);
    float w = state_.at<float>(2);
    float h = state_.at<float>(3);
    
    BoundingBox bbox;
    bbox.x1 = cx - w / 2.0f;
    bbox.y1 = cy - h / 2.0f;
    bbox.x2 = cx + w / 2.0f;
    bbox.y2 = cy + h / 2.0f;
    
    return bbox;
}

TrackObject::TrackObject(const BoundingBox& bbox, int trackId, int frameId, int confirmFrames)
    : trackId_(trackId)
    , state_(TrackState::Tentative)
    , age_(0)
    , hitStreak_(1)
    , frameId_(frameId)
    , score_(bbox.conf)
    , confirmFrames_(confirmFrames) {
    kalman_.Init(bbox);
    
    if (hitStreak_ >= confirmFrames_) {
        state_ = TrackState::Tracked;
    }
}

void TrackObject::Predict() {
    kalman_.Predict();
    age_++;
}

void TrackObject::Update(const BoundingBox& bbox, int frameId) {
    kalman_.Update(bbox);
    frameId_ = frameId;
    score_ = bbox.conf;
    age_ = 0;
    hitStreak_++;
    
    if (state_ == TrackState::Tentative && hitStreak_ >= confirmFrames_) {
        state_ = TrackState::Tracked;
    } else if (state_ == TrackState::Lost) {
        state_ = TrackState::Tracked;
    }
}

void TrackObject::markLost() {
    state_ = TrackState::Lost;
    hitStreak_ = 0;
}

void TrackObject::markRemoved() {
    state_ = TrackState::Removed;
}

void TrackObject::markTracked() {
    state_ = TrackState::Tracked;
}

ByteTracker::ByteTracker(const TrackerConfig& config)
    : config_(config)
    , nextTrackId_(0)
    , frameCount_(0) {
    LOG_INFO("ByteTracker initialized: confirm_frames=" + std::to_string(config.confirmFrames) +
             ", max_lost_frames=" + std::to_string(config.maxLostFrames) +
             ", high_threshold=" + std::to_string(config.highThreshold) +
             ", low_threshold=" + std::to_string(config.lowThreshold));
}

ByteTracker::~ByteTracker() {
}

std::vector<Track> ByteTracker::Update(const cv::Mat& frame,
                                        const std::vector<BoundingBox>& boxes) {
    frameCount_++;
    
    for (auto& track : tracks_) {
        track->Predict();
    }
    
    std::vector<BoundingBox> highDetBoxes;
    std::vector<BoundingBox> lowDetBoxes;
    std::vector<int> highIndices;
    std::vector<int> lowIndices;
    
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (boxes[i].conf >= config_.highThreshold) {
            highDetBoxes.push_back(boxes[i]);
            highIndices.push_back(i);
        } else if (boxes[i].conf >= config_.lowThreshold) {
            lowDetBoxes.push_back(boxes[i]);
            lowIndices.push_back(i);
        }
    }
    
    std::vector<TrackObject*> trackedTracks;
    std::vector<TrackObject*> lostTracks;
    std::vector<int> trackedIndices;
    std::vector<int> lostIndices;
    
    for (size_t i = 0; i < tracks_.size(); ++i) {
        if (tracks_[i]->getState() == TrackState::Tracked) {
            trackedTracks.push_back(tracks_[i].get());
            trackedIndices.push_back(i);
        } else if (tracks_[i]->getState() == TrackState::Lost) {
            if (tracks_[i]->getAge() < config_.maxLostFrames) {
                lostTracks.push_back(tracks_[i].get());
                lostIndices.push_back(i);
            }
        }
    }
    
    std::vector<int> matchedHighDet, matchedTrackedTrack;
    std::vector<int> unmatchedHighDet, unmatchedTrackedTrack;
    
    associateDetectionsToTracks(highDetBoxes, trackedTracks,
                                 matchedHighDet, matchedTrackedTrack,
                                 unmatchedHighDet, unmatchedTrackedTrack,
                                 config_.matchThreshold);
    
    for (size_t i = 0; i < matchedHighDet.size(); ++i) {
        int detIdx = matchedHighDet[i];
        int trackIdx = matchedTrackedTrack[i];
        trackedTracks[trackIdx]->Update(highDetBoxes[detIdx], frameCount_);
    }
    
    std::vector<TrackObject*> unmatchedTrackObjects;
    std::vector<int> unmatchedTrackIndices;
    
    for (size_t i = 0; i < unmatchedTrackedTrack.size(); ++i) {
        int idx = unmatchedTrackedTrack[i];
        unmatchedTrackObjects.push_back(trackedTracks[idx]);
        unmatchedTrackIndices.push_back(trackedIndices[idx]);
    }
    
    for (auto& lostTrack : lostTracks) {
        unmatchedTrackObjects.push_back(lostTrack);
    }
    
    std::vector<int> matchedLowDet, matchedLostTrack;
    std::vector<int> unmatchedLowDet, unmatchedLostTrack;
    
    associateDetectionsToTracks(lowDetBoxes, unmatchedTrackObjects,
                                 matchedLowDet, matchedLostTrack,
                                 unmatchedLowDet, unmatchedLostTrack,
                                 config_.matchThreshold);
    
    for (size_t i = 0; i < matchedLowDet.size(); ++i) {
        int detIdx = matchedLowDet[i];
        int trackIdx = matchedLostTrack[i];
        unmatchedTrackObjects[trackIdx]->Update(lowDetBoxes[detIdx], frameCount_);
        unmatchedTrackObjects[trackIdx]->markLost();
    }
    
    for (int idx : unmatchedTrackedTrack) {
        if (idx < static_cast<int>(trackedTracks.size())) {
            trackedTracks[idx]->markLost();
        }
    }
    
    std::vector<BoundingBox> remainingDetections;
    for (int idx : unmatchedHighDet) {
        remainingDetections.push_back(highDetBoxes[idx]);
    }
    for (int idx : unmatchedLowDet) {
        remainingDetections.push_back(lowDetBoxes[idx]);
    }
    
    for (auto& det : remainingDetections) {
        auto newTrack = std::make_unique<TrackObject>(det, generateNewTrackId(), frameCount_, config_.confirmFrames);
        tracks_.push_back(std::move(newTrack));
    }
    
    std::vector<Track> result;
    
    for (auto& track : tracks_) {
        if (track->getState() == TrackState::Removed) {
            continue;
        }
        
        if (track->getState() == TrackState::Lost && track->getAge() >= config_.maxLostFrames) {
            track->markRemoved();
            continue;
        }
        
        if (track->getState() == TrackState::Tracked) {
            Track t;
            t.trackId = track->getTrackId();
            t.state = track->getState();
            BoundingBox bbox = track->getBox();
            t.x = bbox.x1;
            t.y = bbox.y1;
            t.w = bbox.x2 - bbox.x1;
            t.h = bbox.y2 - bbox.y1;
            t.score = track->getScore();
            result.push_back(t);
        } else if (track->getState() == TrackState::Tentative && 
                   track->getHitStreak() >= config_.confirmFrames) {
            track->markTracked();
            Track t;
            t.trackId = track->getTrackId();
            t.state = TrackState::Tracked;
            BoundingBox bbox = track->getBox();
            t.x = bbox.x1;
            t.y = bbox.y1;
            t.w = bbox.x2 - bbox.x1;
            t.h = bbox.y2 - bbox.y1;
            t.score = track->getScore();
            result.push_back(t);
        }
    }
    
    tracks_.erase(
        std::remove_if(tracks_.begin(), tracks_.end(),
                       [](const std::unique_ptr<TrackObject>& t) {
                           return t->getState() == TrackState::Removed;
                       }),
        tracks_.end());
    
    outputTracks_ = result;
    
    LOG_INFO("ByteTracker Update: " + std::to_string(boxes.size()) + " detections -> " +
             std::to_string(result.size()) + " confirmed tracks");
    
    return result;
}

std::vector<Track> ByteTracker::Predict() {
    std::vector<Track> predictions;
    
    for (auto& track : tracks_) {
        track->Predict();
        
        if (track->getState() != TrackState::Removed) {
            Track t;
            t.trackId = track->getTrackId();
            t.state = track->getState();
            BoundingBox bbox = track->getBox();
            t.x = bbox.x1;
            t.y = bbox.y1;
            t.w = bbox.x2 - bbox.x1;
            t.h = bbox.y2 - bbox.y1;
            t.score = track->getScore();
            predictions.push_back(t);
        }
    }
    
    return predictions;
}

void ByteTracker::reset() {
    tracks_.clear();
    nextTrackId_ = 0;
    frameCount_ = 0;
    outputTracks_.clear();
    LOG_INFO("ByteTracker reset");
}

std::vector<std::pair<int, int>> ByteTracker::linearAssignment(
    const std::vector<std::vector<float>>& costMatrix) {
    
    std::vector<std::pair<int, int>> result;
    if (costMatrix.empty() || costMatrix[0].empty()) {
        return result;
    }
    
    size_t numRows = costMatrix.size();
    size_t numCols = costMatrix[0].size();
    
    std::vector<bool> rowAssigned(numRows, false);
    std::vector<bool> colAssigned(numCols, false);
    
    std::vector<std::tuple<float, int, int>> costs;
    for (size_t i = 0; i < numRows; ++i) {
        for (size_t j = 0; j < numCols; ++j) {
            costs.push_back(std::make_tuple(costMatrix[i][j], i, j));
        }
    }
    
    std::sort(costs.begin(), costs.end());
    
    for (auto& [cost, row, col] : costs) {
        if (!rowAssigned[row] && !colAssigned[col]) {
            result.push_back(std::make_pair(row, col));
            rowAssigned[row] = true;
            colAssigned[col] = true;
        }
    }
    
    return result;
}

std::vector<std::vector<float>> ByteTracker::computeIoUMatrix(
    const std::vector<BoundingBox>& boxesA,
    const std::vector<BoundingBox>& boxesB) {
    
    std::vector<std::vector<float>> iouMatrix;
    
    for (size_t i = 0; i < boxesA.size(); ++i) {
        std::vector<float> row;
        for (size_t j = 0; j < boxesB.size(); ++j) {
            float x1 = std::max(boxesA[i].x1, boxesB[j].x1);
            float y1 = std::max(boxesA[i].y1, boxesB[j].y1);
            float x2 = std::min(boxesA[i].x2, boxesB[j].x2);
            float y2 = std::min(boxesA[i].y2, boxesB[j].y2);
            
            float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            float areaA = boxesA[i].Area();
            float areaB = boxesB[j].Area();
            float unionArea = areaA + areaB - intersection;
            
            float iou = unionArea > 0 ? intersection / unionArea : 0.0f;
            row.push_back(1.0f - iou);
        }
        iouMatrix.push_back(row);
    }
    
    return iouMatrix;
}

void ByteTracker::associateDetectionsToTracks(
    const std::vector<BoundingBox>& detections,
    std::vector<TrackObject*>& tracks,
    std::vector<int>& matchedDetections,
    std::vector<int>& matchedTracks,
    std::vector<int>& unmatchedDetections,
    std::vector<int>& unmatchedTracks,
    float threshold) {
    
    if (detections.empty() || tracks.empty()) {
        for (size_t i = 0; i < detections.size(); ++i) {
            unmatchedDetections.push_back(i);
        }
        for (size_t i = 0; i < tracks.size(); ++i) {
            unmatchedTracks.push_back(i);
        }
        return;
    }
    
    std::vector<BoundingBox> trackBoxes;
    for (auto& track : tracks) {
        trackBoxes.push_back(track->getBox());
    }
    
    auto costMatrix = computeIoUMatrix(detections, trackBoxes);
    auto matches = linearAssignment(costMatrix);
    
    std::vector<bool> detMatched(detections.size(), false);
    std::vector<bool> trackMatched(tracks.size(), false);
    
    for (auto& [detIdx, trackIdx] : matches) {
        float iou = 1.0f - costMatrix[detIdx][trackIdx];
        
        if (iou >= threshold) {
            matchedDetections.push_back(detIdx);
            matchedTracks.push_back(trackIdx);
            detMatched[detIdx] = true;
            trackMatched[trackIdx] = true;
        }
    }
    
    for (size_t i = 0; i < detections.size(); ++i) {
        if (!detMatched[i]) {
            unmatchedDetections.push_back(i);
        }
    }
    
    for (size_t i = 0; i < tracks.size(); ++i) {
        if (!trackMatched[i]) {
            unmatchedTracks.push_back(i);
        }
    }
}

void ByteTracker::secondAssociateDetectionsToTracks(
    const std::vector<BoundingBox>& detections,
    std::vector<TrackObject*>& tracks,
    std::vector<int>& matchedDetections,
    std::vector<int>& matchedTracks,
    std::vector<int>& unmatchedDetections,
    std::vector<int>& unmatchedTracks,
    float threshold) {
    associateDetectionsToTracks(detections, tracks,
                                 matchedDetections, matchedTracks,
                                 unmatchedDetections, unmatchedTracks,
                                 threshold);
}

int ByteTracker::generateNewTrackId() {
    return nextTrackId_++;
}

}