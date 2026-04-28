set(JSON_LOCAL_DIR "${CMAKE_SOURCE_DIR}/third-party/json")

if(EXISTS "${JSON_LOCAL_DIR}/CMakeLists.txt")
    message(STATUS "Using local nlohmann/json")
    add_subdirectory("${JSON_LOCAL_DIR}")
else()
    include(FetchContent)

    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
    )

    message(STATUS "Downloading nlohmann/json...")
    FetchContent_MakeAvailable(json)
endif()
