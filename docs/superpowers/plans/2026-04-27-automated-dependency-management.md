# 自动化第三方依赖管理实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 改造 CMake 构建系统，通过 git submodule 自动管理 OpenCV/yaml-cpp/GoogleTest/RKNN-SDK，通过 file(DOWNLOAD) 管理 Android NDK，消除所有硬编码本地路径。

**Architecture:** 新增 `cmake/` 目录存放构建辅助模块（`FetchDeps.cmake`、`BuildOpenCV.cmake`、`DownloadNDK.cmake`），根 `CMakeLists.txt` 按需初始化 submodule 并引入依赖，`tests/CMakeLists.txt` 复用主项目的依赖配置。

**Tech Stack:** CMake 3.14+, git submodule, ExternalProject, add_subdirectory

---

## 文件结构

| 文件 | 操作 | 职责 |
|------|------|------|
| `.gitmodules` | 创建 | 注册 opencv、yaml-cpp、googletest、rknn-sdk 四个 submodule |
| `cmake/FetchDeps.cmake` | 创建 | `EnsureSubmodule` 函数：按需初始化指定 git submodule |
| `cmake/BuildOpenCV.cmake` | 创建 | `ExternalProject_Add` 配置：精简模块编译 OpenCV 为静态库 |
| `cmake/DownloadNDK.cmake` | 创建 | `file(DOWNLOAD)` + `execute_process(unzip)` 自动获取 Android NDK |
| `CMakeLists.txt` | 修改 | 引入依赖管理模块，移除硬编码路径，添加 RK3566 前置处理 |
| `tests/CMakeLists.txt` | 修改 | 复用主项目依赖，条件引入 googletest，移除硬编码路径 |

---

### Task 1: 创建 .gitmodules 并注册 submodule

**Files:**
- 创建: `.gitmodules`

- [ ] **Step 1: 创建 .gitmodules**

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

- [ ] **Step 2: 注册 submodule 到 git（仅 opencv 和 yaml-cpp，核心依赖）**

```bash
git submodule add -b 4.x https://github.com/opencv/opencv.git third-party/opencv
git submodule add https://github.com/jbeder/yaml-cpp.git third-party/yaml-cpp
```

- [ ] **Step 3: 锁定 opencv 到稳定 tag**

```bash
cd third-party/opencv && git checkout 4.9.0 && cd ../..
git add third-party/opencv
```

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third-party/yaml-cpp
git commit -m "chore: add opencv, yaml-cpp, googletest, rknn-sdk submodules

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 2: 创建 cmake/FetchDeps.cmake

**Files:**
- 创建: `cmake/FetchDeps.cmake`

- [ ] **Step 1: 创建 cmake 目录和 FetchDeps.cmake**

```cmake
function(EnsureSubmodule submodulePath)
    set(fullPath "${CMAKE_SOURCE_DIR}/${submodulePath}")
    if(NOT EXISTS "${fullPath}/CMakeLists.txt")
        message(STATUS "Initializing submodule: ${submodulePath}")
        execute_process(
            COMMAND git submodule update --init -- "${submodulePath}"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            RESULT_VARIABLE result
            OUTPUT_VARIABLE output
            ERROR_VARIABLE error
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(NOT result EQUAL 0)
            message(FATAL_ERROR "Failed to initialize submodule: ${submodulePath}\nstdout: ${output}\nstderr: ${error}")
        endif()
        message(STATUS "Submodule initialized: ${submodulePath}")
    else()
        message(STATUS "Submodule already present: ${submodulePath}")
    endif()
endfunction()
```

- [ ] **Step 2: Commit**

```bash
git add cmake/FetchDeps.cmake
git commit -m "build(cmake): add FetchDeps.cmake for submodule auto-init

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 3: 修改根 CMakeLists.txt - 核心依赖与路径清理

**Files:**
- 修改: `CMakeLists.txt`

- [ ] **Step 1: 记录当前验证基线**

```bash
cmake -B build_test 2>&1 | tail -5
```
Expected: 当前配置可能成功（依赖系统库），但包含硬编码 conda 路径。

- [ ] **Step 2: 修改 `CMakeLists.txt` 顶部，添加 `CMAKE_MODULE_PATH` 和 `FetchDeps`**

在 `project()` 之前新增：
```cmake
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(FetchDeps)
```

- [ ] **Step 3: 在 `project()` 之后，初始化核心 submodule 并引入 yaml-cpp**

在 `option(RK3566_PLATFORM ...)` 之后、`find_package(OpenCV REQUIRED)` 之前，替换为：
```cmake
# 初始化核心依赖
EnsureSubmodule(third-party/opencv)
EnsureSubmodule(third-party/yaml-cpp)

# 引入 yaml-cpp
add_subdirectory(third-party/yaml-cpp)
```

- [ ] **Step 4: 移除 OpenCV 和 yaml-cpp 的 find_package 及硬编码路径**

删除原第 18-22 行：
```cmake
find_package(OpenCV REQUIRED)
find_package(CURL REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};/home/jianzhong/miniconda3/lib/cmake/yaml-cpp")
find_package(yaml-cpp REQUIRED)
```

保留 `find_package(CURL REQUIRED)`（CURL 仍使用系统库）。

- [ ] **Step 5: 更新 target_include_directories 和 target_link_libraries**

将原 target_include_directories 中：
- `${OpenCV_INCLUDE_DIRS}` 保留（后续由 BuildOpenCV.cmake 设置）
- `/home/jianzhong/miniconda3/include` 删除
- 添加 `${YAML_CPP_INCLUDE_DIR}`（yaml-cpp 的 include 目录）

将原 target_link_libraries 中：
- `/home/jianzhong/miniconda3/lib/libyaml-cpp.so` 替换为 `yaml-cpp`

- [ ] **Step 6: Commit**

```bash
git add CMakeLists.txt
git commit -m "build(cmake): integrate yaml-cpp via submodule, remove hardcoded conda paths

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 4: 创建 cmake/BuildOpenCV.cmake

**Files:**
- 创建: `cmake/BuildOpenCV.cmake`

- [ ] **Step 1: 创建 BuildOpenCV.cmake**

```cmake
include(ExternalProject)

set(OPENCV_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/opencv")
set(OPENCV_BUILD_DIR "${CMAKE_BINARY_DIR}/third-party/opencv-build")
set(OPENCV_INSTALL_DIR "${CMAKE_BINARY_DIR}/third-party/opencv-install")

set(OPENCV_BUILD_OPTIONS
    -DBUILD_LIST=core,imgproc,video,imgcodecs
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_EXAMPLES=OFF
    -DBUILD_PERF_TESTS=OFF
    -DBUILD_TESTS=OFF
    -DBUILD_opencv_apps=OFF
    -DBUILD_opencv_python2=OFF
    -DBUILD_opencv_python3=OFF
    -DBUILD_opencv_java=OFF
    -DWITH_GTK=OFF
    -DWITH_QT=OFF
    -DWITH_EIGEN=OFF
    -DWITH_OPENCL=OFF
    -DWITH_CUDA=OFF
    -DWITH_FFMPEG=OFF
    -DWITH_V4L=OFF
    -DWITH_PNG=OFF
    -DWITH_JPEG=OFF
    -DWITH_TIFF=OFF
    -DWITH_WEBP=OFF
    -DWITH_OPENEXR=OFF
    -DWITH_OPENJPEG=OFF
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
)

ExternalProject_Add(opencv_external
    SOURCE_DIR ${OPENCV_SOURCE_DIR}
    BINARY_DIR ${OPENCV_BUILD_DIR}
    CMAKE_ARGS ${OPENCV_BUILD_OPTIONS}
    BUILD_BYPRODUCTS ${OPENCV_INSTALL_DIR}/lib/cmake/opencv4/OpenCVConfig.cmake
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

set(OpenCV_DIR ${OPENCV_INSTALL_DIR}/lib/cmake/opencv4 CACHE PATH "" FORCE)
set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)

set(OPENCV_CORE_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_core.a)
set(OPENCV_IMGPROC_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_imgproc.a)
set(OPENCV_VIDEO_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_video.a)
set(OPENCV_IMGCODECS_LIB ${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.a)

set(OpenCV_LIBS
    ${OPENCV_IMGCODECS_LIB}
    ${OPENCV_VIDEO_LIB}
    ${OPENCV_IMGPROC_LIB}
    ${OPENCV_CORE_LIB}
)
```

- [ ] **Step 2: Commit**

```bash
git add cmake/BuildOpenCV.cmake
git commit -m "build(cmake): add BuildOpenCV.cmake with minimal modules

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 5: 修改根 CMakeLists.txt - 集成 OpenCV

**Files:**
- 修改: `CMakeLists.txt`

- [ ] **Step 1: 在 yaml-cpp 集成之后引入 OpenCV**

在 `add_subdirectory(third-party/yaml-cpp)` 之后新增：
```cmake
include(BuildOpenCV)
```

- [ ] **Step 2: 为主可执行文件添加 OpenCV 依赖**

确保 `difference_detection` 目标依赖 `opencv_external`：
```cmake
add_dependencies(difference_detection opencv_external)
```

- [ ] **Step 3: Commit**

```bash
git add CMakeLists.txt
git commit -m "build(cmake): integrate OpenCV via ExternalProject

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 6: 验证主程序编译

**Files:**
- 测试命令

- [ ] **Step 1: 清理并重新配置**

```bash
rm -rf build
cmake -B build 2>&1 | tail -20
```
Expected: cmake configure 成功，无 conda 路径引用，自动初始化 opencv 和 yaml-cpp submodule。

- [ ] **Step 2: 编译主程序**

```bash
cmake --build build -j$(nproc) 2>&1 | tail -20
```
Expected: 编译成功，生成 `build/difference_detection`。

- [ ] **Step 3: 验证无硬编码路径**

```bash
grep -ri "jianzhong\|miniconda" CMakeLists.txt tests/ cmake/ src/ 2>/dev/null || echo "No hardcoded paths found"
```
Expected: "No hardcoded paths found" 或空输出。

---

### Task 7: 修改 tests/CMakeLists.txt

**Files:**
- 修改: `tests/CMakeLists.txt`

- [ ] **Step 1: 移除硬编码路径和系统 find_package**

删除：
```cmake
find_package(GTest REQUIRED)
```
改为条件引入：
```cmake
if(NOT TARGET GTest::gtest)
    message(FATAL_ERROR "GoogleTest target not available. Build with -DBUILD_TESTS=ON")
endif()
```

删除 include 中的 `/home/jianzhong/miniconda3/include`。
删除 link 中的 `/home/jianzhong/miniconda3/lib/libyaml-cpp.so`，替换为 `yaml-cpp`。

- [ ] **Step 2: 确保测试目标依赖 opencv_external**

在 `add_executable(difference_detection_tests ...)` 之后添加：
```cmake
add_dependencies(difference_detection_tests opencv_external)
```

- [ ] **Step 3: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "build(tests): remove hardcoded paths, use submodule deps

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 8: 修改根 CMakeLists.txt - 条件集成 GoogleTest

**Files:**
- 修改: `CMakeLists.txt`

- [ ] **Step 1: 在 BUILD_TESTS 条件块中初始化并引入 googletest**

将原来的：
```cmake
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
```

替换为：
```cmake
if(BUILD_TESTS)
    enable_testing()
    EnsureSubmodule(third-party/googletest)
    add_subdirectory(third-party/googletest)
    add_subdirectory(tests)
endif()
```

- [ ] **Step 2: Commit**

```bash
git add CMakeLists.txt
git commit -m "build(tests): conditionally init googletest submodule

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 9: 验证测试编译

**Files:**
- 测试命令

- [ ] **Step 1: 重新配置并编译测试**

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -20
```
Expected: configure 成功，自动初始化 googletest submodule。

- [ ] **Step 2: 编译测试**

```bash
cmake --build build -j$(nproc) --target difference_detection_tests 2>&1 | tail -20
```
Expected: 编译成功，生成 `build/tests/difference_detection_tests`。

- [ ] **Step 3: 运行测试**

```bash
./build/tests/difference_detection_tests 2>&1 | tail -10
```
Expected: 测试运行，输出测试通过/失败结果。

---

### Task 10: 创建 cmake/DownloadNDK.cmake

**Files:**
- 创建: `cmake/DownloadNDK.cmake`

- [ ] **Step 1: 创建 DownloadNDK.cmake**

```cmake
set(ANDROID_NDK_VERSION "r25c")
set(ANDROID_NDK_ZIP "android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip")
set(ANDROID_NDK_URL "https://dl.google.com/android/repository/${ANDROID_NDK_ZIP}")
set(ANDROID_NDK_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/downloads")
set(ANDROID_NDK_EXTRACT_DIR "${CMAKE_BINARY_DIR}/android-ndk")
set(ANDROID_NDK_ZIP_PATH "${ANDROID_NDK_DOWNLOAD_DIR}/${ANDROID_NDK_ZIP}")
set(ANDROID_NDK_TOOLCHAIN "${ANDROID_NDK_EXTRACT_DIR}/android-ndk-${ANDROID_NDK_VERSION}/build/cmake/android.toolchain.cmake")

# SHA-256 hash for r25c
set(ANDROID_NDK_SHA256 "769ee35e1cf9dbf9c7a05c8a4c1e5e3d1c1c8a1e1e1e1e1e1e1e1e1e1e1e1e1e1e")

if(NOT EXISTS ${ANDROID_NDK_TOOLCHAIN})
    message(STATUS "Downloading Android NDK ${ANDROID_NDK_VERSION}...")
    file(MAKE_DIRECTORY ${ANDROID_NDK_DOWNLOAD_DIR})
    file(DOWNLOAD ${ANDROID_NDK_URL} ${ANDROID_NDK_ZIP_PATH}
        EXPECTED_HASH SHA256=${ANDROID_NDK_SHA256}
        SHOW_PROGRESS
        STATUS download_status
    )
    list(GET download_status 0 status_code)
    if(NOT status_code EQUAL 0)
        list(GET download_status 1 error_msg)
        message(FATAL_ERROR "Failed to download NDK: ${error_msg}")
    endif()

    message(STATUS "Extracting Android NDK...")
    file(MAKE_DIRECTORY ${ANDROID_NDK_EXTRACT_DIR})
    execute_process(
        COMMAND unzip -q ${ANDROID_NDK_ZIP_PATH} -d ${ANDROID_NDK_EXTRACT_DIR}
        RESULT_VARIABLE unzip_result
    )
    if(NOT unzip_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract NDK")
    endif()
    message(STATUS "Android NDK ready: ${ANDROID_NDK_TOOLCHAIN}")
else()
    message(STATUS "Android NDK already present: ${ANDROID_NDK_TOOLCHAIN}")
endif()
```

> **Note:** 上述 SHA-256 是占位符，实现时需替换为 NDK r25c 的真实 hash。真实 hash 可通过 `curl -sL https://dl.google.com/android/repository/${ANDROID_NDK_ZIP} | sha256sum` 获取。

- [ ] **Step 2: Commit**

```bash
git add cmake/DownloadNDK.cmake
git commit -m "build(cmake): add DownloadNDK.cmake for automatic NDK fetch

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 11: 修改根 CMakeLists.txt - RK3566 交叉编译支持

**Files:**
- 修改: `CMakeLists.txt`

- [ ] **Step 1: 在 `project()` 之前添加 RK3566 NDK 前置处理**

在 `list(APPEND CMAKE_MODULE_PATH ...)` 之后、`project()` 之前插入：
```cmake
if(RK3566_PLATFORM)
    include(DownloadNDK)
    set(CMAKE_TOOLCHAIN_FILE ${ANDROID_NDK_TOOLCHAIN} CACHE STRING "" FORCE)
    set(ANDROID_ABI arm64-v8a CACHE STRING "" FORCE)
    set(ANDROID_NATIVE_API_LEVEL 21 CACHE STRING "" FORCE)
    message(STATUS "RK3566 cross-compile: ANDROID_ABI=${ANDROID_ABI}, API_LEVEL=${ANDROID_NATIVE_API_LEVEL}")
endif()
```

- [ ] **Step 2: 在 `project()` 之后添加 RKNN SDK 配置**

在 `if(RK3566_PLATFORM)` 原有块中（第60-68行附近），在 `find_library(RKNN_LIBRARY ...)` 之前添加 RKNN SDK 初始化：
```cmake
if(RK3566_PLATFORM)
    add_definitions(-DRK3566_PLATFORM)
    message(STATUS "Building for RK3566 platform with RKNN SDK")

    EnsureSubmodule(third-party/rknn-sdk)
    set(RKNN_SDK_PATH "${CMAKE_SOURCE_DIR}/third-party/rknn-sdk")
    set(RKNN_INCLUDE_DIR "${RKNN_SDK_PATH}/runtime/Android/rknn_api/include")
    set(RKNN_LIBRARY "${RKNN_SDK_PATH}/runtime/Android/rknn_api/arm64-v8a/librknnrt.so")

    if(NOT EXISTS ${RKNN_LIBRARY})
        message(WARNING "RKNN library not found at ${RKNN_LIBRARY}, building without real RKNN support")
    else()
        message(STATUS "Found RKNN library: ${RKNN_LIBRARY}")
    endif()
else()
    message(STATUS "Building with stub mode (non-RK3566 platform)")
endif()
```

- [ ] **Step 3: 为主可执行文件添加 RKNN include 和 link**

在 `target_include_directories(difference_detection PRIVATE ...)` 中条件添加：
```cmake
if(RK3566_PLATFORM AND EXISTS ${RKNN_INCLUDE_DIR})
    target_include_directories(difference_detection PRIVATE ${RKNN_INCLUDE_DIR})
endif()
```

在 `target_link_libraries(difference_detection PRIVATE ...)` 中条件添加：
```cmake
if(RK3566_PLATFORM AND EXISTS ${RKNN_LIBRARY})
    target_link_libraries(difference_detection PRIVATE ${RKNN_LIBRARY})
endif()
```

- [ ] **Step 4: Commit**

```bash
git add CMakeLists.txt
git commit -m "build(cmake): add RK3566 cross-compile with auto NDK download and RKNN SDK

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

### Task 12: 最终全面验证

**Files:**
- 测试命令

- [ ] **Step 1: 验证默认编译（无测试，无 RK3566）**

```bash
rm -rf build
cmake -B build 2>&1 | tail -10
cmake --build build -j$(nproc) 2>&1 | tail -10
```
Expected: configure 和 build 均成功。

- [ ] **Step 2: 验证测试编译**

```bash
rm -rf build
cmake -B build -DBUILD_TESTS=ON 2>&1 | tail -10
cmake --build build -j$(nproc) --target difference_detection_tests 2>&1 | tail -10
```
Expected: 测试目标编译成功。

- [ ] **Step 3: 验证 RK3566 配置（仅 cmake configure，不实际编译）**

```bash
rm -rf build
cmake -B build -DRK3566_PLATFORM=ON 2>&1 | tail -15
```
Expected: configure 成功，输出显示 NDK 下载/识别和 RKNN SDK 路径配置。若 NDK 下载耗时较长属正常。

- [ ] **Step 4: 硬编码路径最终检查**

```bash
grep -ri "jianzhong\|miniconda" . --include="*.cmake" --include="CMakeLists.txt" --include="*.cpp" --include="*.h" --include="*.hpp" 2>/dev/null || echo "PASS: No hardcoded paths found"
```
Expected: "PASS: No hardcoded paths found"。

---

## 计划 Review

保存计划后，使用 `plan-document-reviewer` subagent 审查：
- 路径是否正确
- 命令是否可执行
- 是否有遗漏的依赖或边界情况
