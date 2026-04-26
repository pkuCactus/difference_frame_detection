#pragma once

#include <opencv2/opencv.hpp>
#include <string>

namespace diff_det {

class ISimilarityCalculator {
public:
    virtual ~ISimilarityCalculator() = default;

    float Calculate(const cv::Mat& frame1, const cv::Mat& frame2);
    virtual std::string Name() = 0;

private:
    virtual float DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) = 0;
};

class SsimCalculator : public ISimilarityCalculator {
public:
    SsimCalculator();

    std::string Name() override { return "ssim"; }

private:
    float DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) override;
    float calculateSsimChannel(const cv::Mat& img1, const cv::Mat& img2);
};

class PixelDiffCalculator : public ISimilarityCalculator {
public:
    PixelDiffCalculator();

    std::string Name() override { return "pixel_diff"; }

private:
    float DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) override;
    float calculatePixelDiffChannel(const cv::Mat& img1, const cv::Mat& img2);
};

class HashCalculator : public ISimilarityCalculator {
public:
    HashCalculator();

    std::string Name() override { return "phash"; }

private:
    float DoCalculate(const cv::Mat& frame1, const cv::Mat& frame2) override;
    std::vector<uint8_t> computePHash(const cv::Mat& img);
    int hammingDistance(const std::vector<uint8_t>& hash1, const std::vector<uint8_t>& hash2);
};

}