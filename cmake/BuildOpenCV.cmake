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

if(RK3566_PLATFORM)
    list(APPEND OPENCV_BUILD_OPTIONS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}
    )
endif()

ExternalProject_Add(opencv_external
    SOURCE_DIR ${OPENCV_SOURCE_DIR}
    BINARY_DIR ${OPENCV_BUILD_DIR}
    CMAKE_ARGS ${OPENCV_BUILD_OPTIONS}
    BUILD_BYPRODUCTS
        ${OPENCV_INSTALL_DIR}/lib/libopencv_core.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_imgproc.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_video.a
        ${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.a
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

set(OpenCV_DIR ${OPENCV_INSTALL_DIR}/lib/cmake/opencv4 CACHE PATH "" FORCE)
set(OpenCV_INCLUDE_DIRS ${OPENCV_INSTALL_DIR}/include/opencv4)

set(OpenCV_LIBS
    ${OPENCV_INSTALL_DIR}/lib/libopencv_imgcodecs.a
    ${OPENCV_INSTALL_DIR}/lib/libopencv_video.a
    ${OPENCV_INSTALL_DIR}/lib/libopencv_imgproc.a
    ${OPENCV_INSTALL_DIR}/lib/libopencv_core.a
)
