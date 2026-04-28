include(ExternalProject)

set(FFMPEG_VERSION "n6.1.1")
set(FFMPEG_SOURCE_DIR "/tmp/difference_detection-ffmpeg-src")
set(FFMPEG_BUILD_DIR "/tmp/difference_detection-ffmpeg-build")
set(FFMPEG_INSTALL_DIR "${CMAKE_BINARY_DIR}/third-party/ffmpeg-install")

if(CMAKE_TOOLCHAIN_FILE AND ANDROID_ABI)
    set(FFMPEG_CC "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang")
    set(FFMPEG_CXX "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android${ANDROID_NATIVE_API_LEVEL}-clang++")
    set(FFMPEG_AR "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ar")
    set(FFMPEG_STRIP "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-strip")
    set(FFMPEG_NM "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-nm")
    set(FFMPEG_RANLIB "${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/bin/llvm-ranlib")

    set(FFMPEG_CONFIGURE_OPTIONS
        --target-os=android
        --arch=aarch64
        --cpu=armv8-a
        --enable-cross-compile
        --cc=${FFMPEG_CC}
        --cxx=${FFMPEG_CXX}
        --ar=${FFMPEG_AR}
        --strip=${FFMPEG_STRIP}
        --nm=${FFMPEG_NM}
        --ranlib=${FFMPEG_RANLIB}
        --sysroot=${ANDROID_NDK_DIR}/toolchains/llvm/prebuilt/linux-x86_64/sysroot
        --prefix=${FFMPEG_INSTALL_DIR}
        --enable-shared
        --disable-static
        --disable-programs
        --disable-doc
        --disable-debug
        --extra-cflags="-fPIC -Os"
        --extra-ldflags="-Wl,--no-undefined"
    )

    ExternalProject_Add(ffmpeg_external
        URL https://github.com/FFmpeg/FFmpeg/archive/refs/tags/${FFMPEG_VERSION}.tar.gz
        SOURCE_DIR ${FFMPEG_SOURCE_DIR}
        BINARY_DIR ${FFMPEG_BUILD_DIR}
        CONFIGURE_COMMAND bash ${FFMPEG_SOURCE_DIR}/configure ${FFMPEG_CONFIGURE_OPTIONS}
        BUILD_COMMAND $(MAKE) -j4
        INSTALL_COMMAND $(MAKE) install
        BUILD_BYPRODUCTS
        ${FFMPEG_INSTALL_DIR}/lib/libavformat.so
        ${FFMPEG_INSTALL_DIR}/lib/libswresample.so
        ${FFMPEG_INSTALL_DIR}/lib/libavcodec.so
        ${FFMPEG_INSTALL_DIR}/lib/libswscale.so
        ${FFMPEG_INSTALL_DIR}/lib/libavutil.so
    )

    set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INSTALL_DIR}/include)
    set(FFMPEG_LIBRARIES
        ${FFMPEG_INSTALL_DIR}/lib/libavformat.so
        ${FFMPEG_INSTALL_DIR}/lib/libswresample.so
        ${FFMPEG_INSTALL_DIR}/lib/libavcodec.so
        ${FFMPEG_INSTALL_DIR}/lib/libswscale.so
        ${FFMPEG_INSTALL_DIR}/lib/libavutil.so
    )
else()
    # 本地编译：使用系统 FFmpeg
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(FFMPEG REQUIRED libavcodec libavformat libavutil libswscale libswresample)
    set(FFMPEG_INCLUDE_DIRS ${FFMPEG_INCLUDE_DIRS})
    set(FFMPEG_LIBRARIES ${FFMPEG_LIBRARIES})
    # 虚拟 target 保持兼容性
    add_custom_target(ffmpeg_external)
endif()
