#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace diff_det {

class ISimilarityCalculator {
public:
    virtual ~ISimilarityCalculator() = default;
    
    virtual float Calculate(const cv::Mat& frame1, const cv::Mat& frame2) = 0;
    virtual std::string name() = 0;
};

class SsimCalculator : public ISimilarityCalculator {
public:
    SsimCalculator();
    
    float Calculate(const cv::Mat& frame1, const cv::Mat& frame2) override;
    std::string name() override { return "ssim"; }
    
private:
    float calculateSsimChannel(const cv::Mat& img1, const cv::Mat& img2);
    cv::Mat createGaussianKernel(int Size, float sigma);
};

class PixelDiffCalculator : public ISimilarityCalculator {
public:
    PixelDiffCalculator();
    
    float Calculate(const cv::Mat& frame1, const cv::Mat& frame2) override;
    std::string name() override { return "pixel_diff"; }
    
private:
    float calculatePixelDiffChannel(const cv::Mat& img1, const cv::Mat& img2);
};

class HashCalculator : public ISimilarityCalculator {
public:
    HashCalculator();
    
    float Calculate(const cv::Mat& frame1, const cv::Mat& frame2) override;
    std::string name() override { return "phash"; }
    
private:
    std::vector<uint8_t> computePHash(const cv::Mat& img);
    int hammingDistance(const std::vector<uint8_t>& hash1, const std::vector<uint8_t>& hash2);
};

}