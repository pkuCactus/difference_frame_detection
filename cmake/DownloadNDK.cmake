set(ANDROID_NDK_VERSION "r21e")
set(ANDROID_NDK_ZIP_NAME "android-ndk-${ANDROID_NDK_VERSION}-linux-x86_64.zip")
set(ANDROID_NDK_URL "https://dl.google.com/android/repository/${ANDROID_NDK_ZIP_NAME}")
set(ANDROID_NDK_DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/third-party")
set(ANDROID_NDK_ZIP_PATH "${ANDROID_NDK_DOWNLOAD_DIR}/${ANDROID_NDK_ZIP_NAME}")
set(ANDROID_NDK_DIR "${ANDROID_NDK_DOWNLOAD_DIR}/android-ndk-${ANDROID_NDK_VERSION}")

if(NOT EXISTS "${ANDROID_NDK_DIR}")
    if(NOT EXISTS "${ANDROID_NDK_ZIP_PATH}")
        message(STATUS "Downloading Android NDK ${ANDROID_NDK_VERSION}...")
        file(DOWNLOAD "${ANDROID_NDK_URL}" "${ANDROID_NDK_ZIP_PATH}"
            SHOW_PROGRESS
            STATUS download_status
        )
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            list(GET download_status 1 status_message)
            message(FATAL_ERROR "Failed to download Android NDK: ${status_message}")
        endif()
    endif()

    message(STATUS "Extracting Android NDK...")
    execute_process(
        COMMAND unzip -q "${ANDROID_NDK_ZIP_PATH}" -d "${ANDROID_NDK_DOWNLOAD_DIR}"
        RESULT_VARIABLE unzip_result
    )
    if(NOT unzip_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract Android NDK")
    endif()

endif()

# unzip 不保留可执行权限，需要确保 NDK 中编译器工具链具有执行权限
file(GLOB NDK_PREBUILT_BIN_DIRS "${ANDROID_NDK_DIR}/toolchains/*/prebuilt/*/bin")
foreach(binDir ${NDK_PREBUILT_BIN_DIRS})
    if(EXISTS "${binDir}")
        message(STATUS "Fixing executable permissions: ${binDir}")
        execute_process(
            COMMAND chmod -R +x "${binDir}"
            RESULT_VARIABLE chmod_result
        )
    endif()
endforeach()

# 验证文件系统是否支持可执行权限
set(NDK_CLANG_PATH "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/clang++")
if(EXISTS "${NDK_CLANG_PATH}")
    execute_process(
        COMMAND test -x "${NDK_CLANG_PATH}"
        RESULT_VARIABLE test_exec_result
    )
    if(NOT test_exec_result EQUAL 0)
        message(FATAL_ERROR
            "NDK 工具链没有可执行权限，当前文件系统可能不支持 Unix 执行权限（例如 NTFS/exFAT）。\n"
            "请设置 ANDROID_NDK_DOWNLOAD_DIR 到一个支持 Unix 权限的目录，例如:\n"
            "  cmake -DANDROID_NDK_DOWNLOAD_DIR=/tmp/difference_detection-ndk ...\n"
            "或者手动指定已安装的 NDK:\n"
            "  cmake -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake ..."
        )
    endif()
endif()

set(CMAKE_TOOLCHAIN_FILE "${ANDROID_NDK_DIR}/build/cmake/android.toolchain.cmake" CACHE FILEPATH "" FORCE)

if(NOT ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "" FORCE)
endif()
if(NOT ANDROID_NATIVE_API_LEVEL)
    set(ANDROID_NATIVE_API_LEVEL 21 CACHE STRING "" FORCE)
endif()

message(STATUS "Android NDK: ${ANDROID_NDK_DIR}")
message(STATUS "Android ABI: ${ANDROID_ABI}")
message(STATUS "Android API Level: ${ANDROID_NATIVE_API_LEVEL}")
