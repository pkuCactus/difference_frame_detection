include(ExternalProject)

set(CURL_VERSION "8.7.1")
set(CURL_TARBALL "curl-${CURL_VERSION}.tar.gz")
set(CURL_URL "https://curl.se/download/${CURL_TARBALL}")
if(NOT CURL_DOWNLOAD_DIR)
    set(CURL_DOWNLOAD_DIR "${CMAKE_SOURCE_DIR}/third-party")
endif()
set(CURL_TARBALL_PATH "${CURL_DOWNLOAD_DIR}/${CURL_TARBALL}")
set(CURL_SOURCE_DIR "${CURL_DOWNLOAD_DIR}/curl-${CURL_VERSION}")
set(CURL_BUILD_DIR "${CMAKE_BINARY_DIR}/third-party/curl-build")
set(CURL_INSTALL_DIR "${CMAKE_BINARY_DIR}/third-party/curl-install")

if(NOT EXISTS "${CURL_SOURCE_DIR}")
    if(NOT EXISTS "${CURL_TARBALL_PATH}")
        message(STATUS "Downloading curl ${CURL_VERSION}...")
        file(DOWNLOAD "${CURL_URL}" "${CURL_TARBALL_PATH}"
            SHOW_PROGRESS
            STATUS download_status
        )
        list(GET download_status 0 status_code)
        if(NOT status_code EQUAL 0)
            list(GET download_status 1 status_message)
            message(FATAL_ERROR "Failed to download curl: ${status_message}")
        endif()
    endif()

    message(STATUS "Extracting curl...")
    execute_process(
        COMMAND tar -xzf "${CURL_TARBALL_PATH}" -C "${CURL_DOWNLOAD_DIR}"
        RESULT_VARIABLE tar_result
    )
    if(NOT tar_result EQUAL 0)
        message(FATAL_ERROR "Failed to extract curl")
    endif()
endif()

set(CURL_BUILD_OPTIONS
    -DBUILD_CURL_EXE=OFF
    -DBUILD_SHARED_LIBS=OFF
    -DBUILD_TESTING=OFF
    -DCURL_DISABLE_TESTS=ON
    -DHTTP_ONLY=ON
    -DCURL_USE_OPENSSL=OFF
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_INSTALL_PREFIX=${CURL_INSTALL_DIR}
)

if(RK3566_PLATFORM)
    list(APPEND CURL_BUILD_OPTIONS
        -DCMAKE_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}
        -DANDROID_ABI=${ANDROID_ABI}
        -DANDROID_NATIVE_API_LEVEL=${ANDROID_NATIVE_API_LEVEL}
    )
endif()

ExternalProject_Add(curl_external
    SOURCE_DIR ${CURL_SOURCE_DIR}
    BINARY_DIR ${CURL_BUILD_DIR}
    CMAKE_ARGS ${CURL_BUILD_OPTIONS}
    BUILD_BYPRODUCTS
        ${CURL_INSTALL_DIR}/lib/libcurl.a
    INSTALL_COMMAND ${CMAKE_COMMAND} --build . --target install
)

set(CURL_INCLUDE_DIRS ${CURL_INSTALL_DIR}/include)
set(CURL_LIBRARIES ${CURL_INSTALL_DIR}/lib/libcurl.a)
