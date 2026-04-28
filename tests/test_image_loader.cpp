#include <gtest/gtest.h>
#include "common/image_loader.h"
#include <filesystem>
#include <fstream>

using namespace diff_det;

TEST(ImageLoaderTest, DiagnoseNonExistentFile) {
    std::string result = DiagnoseImageLoadFailure("/nonexistent/path/test.jpg");
    EXPECT_NE(result.find("文件不存在"), std::string::npos);
    EXPECT_NE(result.find("当前工作目录"), std::string::npos);
}

TEST(ImageLoaderTest, DiagnoseEmptyFile) {
    std::string tmpPath = "/tmp/test_empty_img.jpg";
    {
        std::ofstream ofs(tmpPath);
    }
    std::string result = DiagnoseImageLoadFailure(tmpPath);
    std::filesystem::remove(tmpPath);
    EXPECT_NE(result.find("文件为空"), std::string::npos);
}

TEST(ImageLoaderTest, DiagnoseNormalFile) {
    std::string tmpPath = "/tmp/test_dummy_img.jpg";
    {
        std::ofstream ofs(tmpPath);
        ofs << "dummy data";
    }
    std::string result = DiagnoseImageLoadFailure(tmpPath);
    std::filesystem::remove(tmpPath);
    EXPECT_NE(result.find("文件大小"), std::string::npos);
    EXPECT_NE(result.find("cv::imread 返回空"), std::string::npos);
    EXPECT_NE(result.find("OpenCV 编译时未启用"), std::string::npos);
}
