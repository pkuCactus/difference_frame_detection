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
