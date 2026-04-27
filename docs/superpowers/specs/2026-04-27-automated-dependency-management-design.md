# 自动化第三方依赖管理设计文档

## 背景

当前项目的 CMake 配置依赖系统安装的库（OpenCV、yaml-cpp、GoogleTest），且 yaml-cpp 路径硬编码了开发者本地 conda 环境（`/home/jianzhong/miniconda3/...`）。这导致：
- 新开发者 clone 后无法直接编译
- 不同环境路径不一致
- RK3566 交叉编译需要手动下载 NDK 和 RKNN SDK

## 目标

1. clone 后自动下载 opencv、yaml-cpp 到 `third-party/`，编译时一起编译
2. 编译 test 时才下载 GoogleTest
3. 编译 rknn 时才下载 Android NDK 和 RKNN SDK
4. 消除所有硬编码的本地路径
5. OpenCV 只启用项目实际使用的模块（imgproc、video、imgcodecs），缩短编译时间

## 方案选型

采用 **git submodule + add_subdirectory / ExternalProject 混合方案**：
- yaml-cpp、GoogleTest 结构简单，直接 `add_subdirectory` 嵌入编译
- OpenCV 模块多、target 多，使用 `ExternalProject_Add`（仅指定 `SOURCE_DIR` 指向 submodule 目录，不配置 `GIT_REPOSITORY`，避免与 submodule 双重管理冲突）在独立目录编译，避免污染主项目
- NDK、RKNN SDK 纯工具链/库，无需编译，初始化后配置路径即可

## 目录结构

```
third-party/
├── opencv/          # OpenCV 源码仓库 (github.com/opencv/opencv.git, 4.x 分支)
├── yaml-cpp/        # yaml-cpp 源码仓库 (github.com/jbeder/yaml-cpp.git)
├── googletest/      # GoogleTest 源码仓库 (github.com/google/googletest.git)
└── rknn-sdk/        # RKNN SDK (github.com/rockchip-linux/rknpu2.git, 按需初始化)

cmake/
├── FetchDeps.cmake      # submodule 初始化与依赖管理
├── BuildOpenCV.cmake    # OpenCV ExternalProject 配置
└── DownloadNDK.cmake    # NDK 自动下载与解压

# NDK 下载到构建目录外（体积大，不放入源码树）
# 默认路径: $ENV{HOME}/.local/android-ndk-r25c
```

## git submodule 配置

```ini
[submodule "third-party/opencv"]
    path = third-party/opencv
    url = https://github.com/opencv/opencv.git
    branch = 4.x
[submodule "third-party/yaml-cpp"]
    path = third-party/yaml-cpp
    url = https://github.com/jbeder/yaml-cpp.git
[submodule "third-party/googletest"]
    path = third-party/googletest
    url = https://github.com/google/googletest.git
[submodule "third-party/rknn-sdk"]
    path = third-party/rknn-sdk
    url = https://github.com/rockchip-linux/rknpu2.git
```

> **版本锁定说明**：git submodule 本身会记录一个具体的 commit hash。开发者初始化 submodule 后，需进入各 submodule 目录 `git checkout <稳定tag>`，然后在主仓库提交锁定。`.gitmodules` 中的 `branch` 字段仅作为参考，实际构建使用的是 `.git` 中记录的 commit。OpenCV 建议锁定到 `4.9.0` 或更高稳定 tag。

## CMake 依赖管理架构

### FetchDeps.cmake

封装 submodule 初始化逻辑，提供 `EnsureSubmodule` 函数：

```cmake
function(EnsureSubmodule submodulePath)
    if(NOT EXISTS "${CMAKE_SOURCE_DIR}/${submodulePath}/CMakeLists.txt")
        message(STATUS "Initializing submodule: ${submodulePath}")
        execute_process(
            COMMAND git submodule update --init -- "${submodulePath}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE result
        )
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Failed to initialize submodule: ${submodulePath}")
        endif()
    endif()
endfunction()
```

### 主 CMakeLists.txt 流程

1. 设置 `CMAKE_MODULE_PATH` 包含 `cmake/` 目录
2. `include(FetchDeps)`
3. 初始化核心依赖：`EnsureSubmodule(third-party/opencv)`、`EnsureSubmodule(third-party/yaml-cpp)`
4. `add_subdirectory(third-party/yaml-cpp)`
5. `include(BuildOpenCV)`
6. `if(BUILD_TESTS)` → `EnsureSubmodule(third-party/googletest)` → `add_subdirectory(third-party/googletest)`
7. `if(RK3566_PLATFORM)` → 下载 NDK、初始化 RKNN SDK submodule，配置工具链和库路径

### 移除的硬编码路径

- `/home/jianzhong/miniconda3/lib/cmake/yaml-cpp`
- `/home/jianzhong/miniconda3/include`
- `/home/jianzhong/miniconda3/lib/libyaml-cpp.so`

替换为 `target_link_libraries(... yaml-cpp)` 和正确的 include 路径。

## OpenCV ExternalProject 配置

`cmake/BuildOpenCV.cmake` 使用 `ExternalProject_Add`：

- **源码目录**：`${CMAKE_SOURCE_DIR}/third-party/opencv`
- **构建目录**：`${CMAKE_BINARY_DIR}/third-party/opencv-build`
- **安装目录**：`${CMAKE_BINARY_DIR}/third-party/opencv-install`
- **启用模块**：`core,imgproc,video,imgcodecs`（通过 `-DBUILD_LIST`）
- **关闭项**：examples、perf_tests、tests、python bindings、GTK、QT、Eigen、OpenCL、CUDA、FFmpeg、V4L、PNG、JPEG、TIFF、WebP、OpenEXR、OpenJPEG
- **输出**：静态库 `.a` 文件
- **BUILD_BYPRODUCTS**：显式声明安装标记文件 `${OPENCV_INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake`，满足 Ninja 生成器要求。该文件在所有模块和第三方依赖编译安装完成后生成，避免枚举所有中间 `.a` 文件

编译时间预计从 30 分钟降至 3~5 分钟。

## RK3566 交叉编译配置

当 `RK3566_PLATFORM=ON`：

1. **NDK 下载**：通过 `cmake/DownloadNDK.cmake` 使用 `file(DOWNLOAD ...)` 从 Google 官方镜像下载 Android NDK 的 Linux x86_64 tar.gz（如 `android-ndk-r25c-linux-x86_64.zip`），使用 `EXPECTED_HASH` 校验完整性，解压到 `${CMAKE_BINARY_DIR}/android-ndk` 或 `$ENV{HOME}/.local/android-ndk-r25c`
2. 初始化 `third-party/rknn-sdk` submodule
3. 设置 Android ABI（默认 `arm64-v8a`，RK3566 为 Cortex-A55 64 位核心）
4. 设置 `ANDROID_NATIVE_API_LEVEL=21`
5. `CMAKE_TOOLCHAIN_FILE` 指向 NDK 内置的 `android.toolchain.cmake`
6. RKNN SDK 头文件路径：`${RKNN_SDK_PATH}/runtime/Android/rknn_api/include`
7. RKNN 库路径：`${RKNN_SDK_PATH}/runtime/Android/rknn_api/arm64-v8a/librknnrt.so`

> **工具链文件设置时机**：`CMAKE_TOOLCHAIN_FILE` 需在 `project()` 之前生效。推荐做法：在 CMakeLists.txt 的 `project()` 调用之前，检测 `RK3566_PLATFORM`（通过命令行 `-DRK3566_PLATFORM=ON` 传入），若未找到 NDK 工具链文件，先执行 `include(DownloadNDK)` 下载并解压 NDK，然后设置 `CMAKE_TOOLCHAIN_FILE` 变量指向解压后的 `android.toolchain.cmake`。

> **RKNN SDK 版本说明**：RKNN SDK2 (rknpu2) 的 runtime 库文件名为 `librknnrt.so`。不同版本的 SDK 目录结构可能略有差异，建议锁定到 rknpu2 的某个稳定 commit。

## 测试策略

本任务为构建系统改造，测试重点：

1. **CMake 配置通过**：`cmake -B build` 成功完成，正确初始化 submodule
2. **主程序编译通过**：`cmake --build build` 生成 `difference_detection` 可执行文件
3. **测试编译通过**：`cmake -B build -DBUILD_TESTS=ON` 成功编译 `difference_detection_tests`
4. **RK3566 配置通过**：`cmake -B build -DRK3566_PLATFORM=ON` 正确下载 NDK、识别 RKNN SDK 路径
5. **消除硬编码路径验证**：在 `grep -r "jianzhong/miniconda3" CMakeLists.txt tests/ cmake/` 无匹配的前提下，编译成功即通过

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| OpenCV 首次编译仍耗时数分钟 | 仅启用必要模块，已最大程度裁剪；后续编译复用 build 缓存 |
| NDK 体积大（1~4GB），下载慢 | NDK 通过 `file(DOWNLOAD)` 按需下载，默认编译不触发；下载后缓存在构建目录或用户本地目录，可复用 |
| submodule 仓库网络不可达 | `execute_process` 失败时给出明确错误信息；支持预先手动 clone |
| RKNN SDK 仓库结构变化 | 使用相对路径配置，若结构变化需适配；文档中记录验证的版本 |
| 上游依赖版本不兼容 | 所有 submodule 固定到具体 commit/tag，不跟踪分支最新；OpenCV 锁定在 4.x 的某个稳定 tag |
