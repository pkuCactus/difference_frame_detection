#include "common/image_loader.h"
#include <filesystem>
#include <sstream>

namespace diff_det {

std::string DiagnoseImageLoadFailure(const std::string& path) {
    namespace fs = std::filesystem;
    std::ostringstream oss;

    if (!fs::exists(path)) {
        oss << "文件不存在: " << path << "\n";
        oss << "当前工作目录: " << fs::current_path().string();
        return oss.str();
    }

    if (!fs::is_regular_file(path)) {
        oss << "路径不是常规文件: " << path;
        return oss.str();
    }

    auto size = fs::file_size(path);
    oss << "文件大小: " << size << " bytes\n";
    if (size == 0) {
        oss << "文件为空";
        return oss.str();
    }

    oss << "cv::imread 返回空，可能原因：\n";
    oss << "  - OpenCV 编译时未启用 JPEG/PNG 支持（RK3566 交叉编译常见）\n";
    oss << "  - 图像文件格式损坏\n";
    oss << "  - 文件权限不足";
    return oss.str();
}

} // namespace diff_det
