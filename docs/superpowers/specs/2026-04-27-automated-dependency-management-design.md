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
- OpenCV 模块多、target 多，使用 `ExternalProject_Add` 在独立目录编译，避免污染主项目
- NDK、RKNN SDK 纯工具链/库，无需编译，初始化后配置路径即可

## 目录结构

```
third-party/
├── opencv/          # OpenCV 源码仓库 (github.com/opencv/opencv.git, 4.x 分支)
├── yaml-cpp/        # yaml-cpp 源码仓库 (github.com/jbeder/yaml-cpp.git)
├── googletest/      # GoogleTest 源码仓库 (github.com/google/googletest.git)
├── android-ndk/     # Android NDK (按需初始化)
└── rknn-sdk/        # RKNN SDK (按需初始化)

cmake/
├── FetchDeps.cmake      # submodule 初始化与依赖管理
└── BuildOpenCV.cmake    # OpenCV ExternalProject 配置
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
[submodule "third-party/android-ndk"]
    path = third-party/android-ndk
    url = https://github.com/android/ndk.git
[submodule "third-party/rknn-sdk"]
    path = third-party/rknn-sdk
    url = https://github.com/rockchip-linux/rknpu2.git
```

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
7. `if(RK3566_PLATFORM)` → 初始化 NDK、RKNN SDK，配置工具链和库路径

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
- **关闭项**：examples、perf_tests、tests、python bindings、GTK、QT、Eigen、OpenCL、CUDA、FFmpeg、V4L
- **输出**：静态库 `.a` 文件
- **BUILD_BYPRODUCTS**：显式声明所有生成的静态库，满足 Ninja 生成器要求：
  - `libopencv_core.a`
  - `libopencv_imgproc.a`
  - `libopencv_video.a`
  - `libopencv_imgcodecs.a`

编译时间预计从 30 分钟降至 3~5 分钟。

## RK3566 交叉编译配置

当 `RK3566_PLATFORM=ON`：

1. 初始化 `third-party/android-ndk` 和 `third-party/rknn-sdk`
2. 设置 Android ABI（默认 `arm64-v8a`，RK3566 为 Cortex-A55 64 位核心）
3. 设置 `ANDROID_NATIVE_API_LEVEL=21`
4. `CMAKE_TOOLCHAIN_FILE` 指向 NDK 内置的 `android.toolchain.cmake`
5. RKNN SDK 头文件路径：`${RKNN_SDK_PATH}/runtime/Android/rknn_api/include`
6. RKNN 库路径：`${RKNN_SDK_PATH}/runtime/Android/rknn_api/arm64-v8a/librknn_api.so`

> 工具链文件需在 `project()` 之前生效，推荐通过 `cmake -DCMAKE_TOOLCHAIN_FILE=...` 传入，或在 CMakeLists.txt 顶部提前检测并设置。

## 测试策略

本任务为构建系统改造，测试重点：

1. **CMake 配置通过**：`cmake -B build` 成功完成，正确初始化 submodule
2. **主程序编译通过**：`cmake --build build` 生成 `difference_detection` 可执行文件
3. **测试编译通过**：`cmake -B build -DBUILD_TESTS=ON` 成功编译 `difference_detection_tests`
4. **RK3566 配置通过**：`cmake -B build -DRK3566_PLATFORM=ON` 正确识别 NDK 和 RKNN SDK 路径
5. **消除硬编码路径验证**：在新干净环境中（无 conda 路径）也能编译通过

## 风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| OpenCV 首次编译仍耗时数分钟 | 仅启用必要模块，已最大程度裁剪；后续编译复用 build 缓存 |
| NDK 体积大（1~4GB），clone 慢 | `EnsureSubmodule` 执行 `git submodule update --init third-party/xxx`，仅初始化指定 submodule，不会下载全部；默认编译不触发 NDK 下载 |
| submodule 仓库网络不可达 | `execute_process` 失败时给出明确错误信息；支持预先手动 clone |
| RKNN SDK 仓库结构变化 | 使用相对路径配置，若结构变化需适配；文档中记录验证的版本 |
| 上游依赖版本不兼容 | 所有 submodule 固定到具体 commit/tag，不跟踪分支最新；OpenCV 锁定在 4.x 的某个稳定 tag |
