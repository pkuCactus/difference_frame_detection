include(ExternalProject)

set(OPENCV_SOURCE_DIR "${CMAKE_SOURCE_DIR}/third-party/opencv")
set(OPENCV_BUILD_DIR "${CMAKE_BINARY_DIR}/third-party/opencv-build")
set(OPENCV_INSTALL_DIR "${CMAKE_BINARY_DIR}/third-party/opencv-install")

set(OPENCV_BUILD_OPTIONS
    -DBUILD_LIST=core,imgproc,video,videoio,imgcodecs,highgui
    -DWITH_FFMPEG=ON
    -DHAVE_FFMPEG=ON
    -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON
    -DFFMPEG_INCLUDE_DIRS=${FFMPEG_INCLUDE_DIRS}
    -DFFMPEG_libavcodec_VERSION=60.31.102
    -DFFMPEG_libavformat_VERSION=60.16.100
    -DFFMPEG_libavutil_VERSION=58.29.100
    -DFFMPEG_libswscale_VERSION=7.5.100
    -DVIDEOIO_PLUGIN_LIST=ffmpeg
    -DBUILD_opencv_videoio=ON
    -DWITH_ANDROID_MEDIANDK=ON
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
    -DWITH_V4L=OFF
    -DWITH_1394=OFF
    -DWITH_PNG=OFF
    -DWITH_JPEG=OFF
    -DWITH_TIFF=OFF
    -DWITH_WEBP=OFF
    -DWITH_OPENEXR=OFF
    -DWITH_OPENJPEG=OFF
    -DWITH_JASPER=OFF
    -DWITH_ITT=OFF
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${OPENCV_INSTALL_DIR}
)

if(CMAKE_TOOLCHAIN_FILE)
    list(APPEND OPENCV_BUILD_OPTIONS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}
        -DBUILD_ANDROID_PROJECTS=OFF
        -DBUILD_ANDROID_EXAMPLES=OFF
        -DBUILD_JAVA=OFF
    )
endif()

ExternalProject_Add(opencv_external
    DEPENDS ffmpeg_external
    SOURCE_DIR ${OPENCV_SOURCE_DIR}
    BINARY_DIR ${OPENCV_BUILD_DIR}
    CMAKE_ARGS ${OPENCV_BUILD_OPTIONS}
    CMAKE_CACHE_ARGS
        -DFFMPEG_LIBRARIES:STRING=${FFMPEG_LIBRARIES}
    BUILD_BYPRODUCTS
        ${OPENCV_INSTALL_DIR}/sdk/native/libs/${ANDROID_ABI}/libopencv_videoio_ffmpeg.so
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_core.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_imgproc.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_video.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_videoio.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_imgcodecs.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_highgui.a
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

# Android 交叉编译时 OpenCV 安装结构不同
if(CMAKE_TOOLCHAIN_FILE AND ANDROID_ABI)
    # Android SDK 结构：sdk/native/jni/include 和 sdk/native/staticlibs/arm64-v8a
    set(OpenCV_DIR ${OPENCV_INSTALL_DIR}/sdk/native/jni CACHE PATH "" FORCE)
    set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/sdk/native/jni/include)

    set(OpenCV_LIBS
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_highgui.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_imgcodecs.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_videoio.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_video.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_imgproc.a
        ${OPENCV_INSTALL_DIR}/sdk/native/staticlibs/${ANDROID_ABI}/libopencv_core.a
        ${OPENCV_INSTALL_DIR}/sdk/native/3rdparty/libs/${ANDROID_ABI}/libcpufeatures.a
        ${OPENCV_INSTALL_DIR}/sdk/native/3rdparty/libs/${ANDROID_ABI}/libtegra_hal.a
    )
else()
    # 标准 Linux 安装结构
    set(OpenCV_DIR ${OPENCV_INSTALL_DIR}/lib/cmake/opencv4 CACHE PATH "" FORCE)
    set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)

    set(OpenCV_LIBS
        ${OPENCV_INSTALL_DIR}/lib/libopencv_highgui.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_videoio.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_video.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_imgproc.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_core.a
    )
endif()
