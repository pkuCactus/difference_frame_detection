#include "analysis/similarity.h"
#include "common/logger.h"
#include <cmath>
#include <algorithm>

namespace diff_det {

float ISimilarityCalculator::Calculate(const cv::Mat& frame1, const cv::Mat& frame2) {
    if (frame1.empty() || frame2.empty()) {
        LOG_WARN(Name() + ": empty frame(s) received");
        return 0.0f;
    }
    return DoCalculate(frame1, frame2);
}

SsimCalculator::SsimCalculator() {
}

float SsimCalculator::DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) {
    if (frame1.size() != frame2.size()) {
        LOG_WARN("SSIM: frame sizes mismatch");
        return 0.0f;
    }
    
    cv::Mat gray1, gray2;
    
    if (frame1.channels() == 3) {
        cv::cvtColor(frame1, gray1, cv::COLOR_BGR2GRAY);
    } else {
        gray1 = frame1.clone();
    }
    
    if (frame2.channels() == 3) {
        cv::cvtColor(frame2, gray2, cv::COLOR_BGR2GRAY);
    } else {
        gray2 = frame2.clone();
    }
    
    gray1.convertTo(gray1, CV_32F);
    gray2.convertTo(gray2, CV_32F);
    
    float ssim = calculateSsimChannel(gray1, gray2);
    
    LOG_DEBUG("SSIM calculated: " + std::to_string(ssim));
    
    return ssim;
}

float SsimCalculator::calculateSsimChannel(const cv::Mat& img1, const cv::Mat& img2) {
    const float C1 = 6.5025f;
    const float C2 = 58.5225f;
    
    cv::Mat mu1, mu2;
    cv::GaussianBlur(img1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2, mu2, cv::Size(11, 11), 1.5);
    
    cv::Mat mu1_sq = mu1.mul(mu1);
    cv::Mat mu2_sq = mu2.mul(mu2);
    cv::Mat mu1_mu2 = mu1.mul(mu2);
    
    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    
    cv::Mat img1_sq = img1.mul(img1);
    cv::Mat img2_sq = img2.mul(img2);
    cv::Mat img1_img2 = img1.mul(img2);
    
    cv::GaussianBlur(img1_sq, sigma1_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2_sq, sigma2_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1_img2, sigma12, cv::Size(11, 11), 1.5);
    
    sigma1_sq = sigma1_sq - mu1_sq;
    sigma2_sq = sigma2_sq - mu2_sq;
    sigma12 = sigma12 - mu1_mu2;
    
    cv::Mat numerator1 = 2.0f * mu1_mu2 + C1;
    cv::Mat numerator2 = 2.0f * sigma12 + C2;
    cv::Mat denominator1 = mu1_sq + mu2_sq + C1;
    cv::Mat denominator2 = sigma1_sq + sigma2_sq + C2;
    
    cv::Mat ssim_map;
    cv::divide(numerator1.mul(numerator2), 
               denominator1.mul(denominator2), 
               ssim_map);
    
    cv::Scalar mean = cv::mean(ssim_map);
    
    return mean[0];
}

cv::Mat SsimCalculator::createGaussianKernel(int size, float sigma) {
    cv::Mat kernel(size, size, CV_32F);
    int half = size / 2;
    
    for (int y = -half; y <= half; ++y) {
        for (int x = -half; x <= half; ++x) {
            float value = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel.at<float>(y + half, x + half) = value;
        }
    }
    
    kernel /= cv::sum(kernel)[0];
    
    return kernel;
}

PixelDiffCalculator::PixelDiffCalculator() {
}

float PixelDiffCalculator::DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) {
    if (frame1.size() != frame2.size()) {
        LOG_WARN("PixelDiff: frame sizes mismatch");
        return 0.0f;
    }
    
    cv::Mat diff;
    cv::absdiff(frame1, frame2, diff);
    
    cv::Scalar mean = cv::mean(diff);
    
    float avgDiff = 0.0f;
    for (int i = 0; i < diff.channels(); ++i) {
        avgDiff += mean[i];
    }
    avgDiff /= diff.channels();
    
    float similarity = 1.0f - (avgDiff / 255.0f);
    
    LOG_DEBUG("PixelDiff calculated: avgDiff=" + std::to_string(avgDiff) +
              ", similarity=" + std::to_string(similarity));
    
    return similarity;
}

float PixelDiffCalculator::calculatePixelDiffChannel(const cv::Mat& img1, const cv::Mat& img2) {
    cv::Mat diff;
    cv::absdiff(img1, img2, diff);
    
    cv::Scalar mean = cv::mean(diff);
    return 1.0f - mean[0] / 255.0f;
}

HashCalculator::HashCalculator() {
}

float HashCalculator::DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) {
    auto hash1 = computePHash(frame1);
    auto hash2 = computePHash(frame2);
    
    if (hash1.empty() || hash2.empty()) {
        return 0.0f;
    }
    
    int distance = hammingDistance(hash1, hash2);
    
    float maxDistance = static_cast<float>(hash1.size() * 8);
    float similarity = 1.0f - static_cast<float>(distance) / maxDistance;
    
    LOG_DEBUG("PHash calculated: hammingDistance=" + std::to_string(distance) +
              ", similarity=" + std::to_string(similarity));
    
    return similarity;
}

std::vector<uint8_t> HashCalculator::computePHash(const cv::Mat& img) {
    cv::Mat gray;
    if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
        gray = img.clone();
    }
    
    cv::Mat resized;
    cv::resize(gray, resized, cv::Size(32, 32));
    
    cv::Mat blurred;
    cv::GaussianBlur(resized, blurred, cv::Size(5, 5), 0);
    
    cv::Mat dctResult;
    cv::dct(cv::Mat_<float>(blurred), dctResult);
    
    cv::Mat lowFreq = dctResult(cv::Rect(0, 0, 8, 8));
    
    cv::Scalar mean = cv::mean(lowFreq);
    float avgVal = mean[0];
    
    std::vector<uint8_t> hash;
    hash.reserve(8);
    
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = 0;
        for (int j = 0; j < 8; ++j) {
            float val = lowFreq.at<float>(i, j);
            if (val > avgVal) {
                byte |= (1 << j);
            }
        }
        hash.push_back(byte);
    }
    
    return hash;
}

int HashCalculator::hammingDistance(const std::vector<uint8_t>& hash1, 
                                     const std::vector<uint8_t>& hash2) {
    if (hash1.size() != hash2.size()) {
        return static_cast<int>(hash1.size() * 8);
    }
    
    int distance = 0;
    for (size_t i = 0; i < hash1.size(); ++i) {
        uint8_t xorResult = hash1[i] ^ hash2[i];
        for (int j = 0; j < 8; ++j) {
            if (xorResult & (1 << j)) {
                distance++;
            }
        }
    }
    
    return distance;
}

}