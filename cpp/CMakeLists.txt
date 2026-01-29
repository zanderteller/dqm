cmake_minimum_required(VERSION 3.17)
project(dqm_python LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Source files
set(SOURCES
    utils.cpp
    array3d.cpp
    basis.cpp
    trajectories.cpp
    operators.cpp
    dqm_python.cpp
)

# Create shared library
add_library(dqm_python SHARED ${SOURCES})

# Remove the "lib" prefix that CMake adds by default on Unix
set_target_properties(dqm_python PROPERTIES PREFIX "")

# Include directories (Eigen is header-only, bundled in include/)
target_include_directories(dqm_python PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find OpenMP - try multiple approaches for macOS compatibility
find_package(OpenMP)

if(NOT OpenMP_CXX_FOUND AND APPLE)
    # Try to find Homebrew's libomp on macOS
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE HOMEBREW_LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
        RESULT_VARIABLE BREW_RESULT
    )
    if(BREW_RESULT EQUAL 0 AND HOMEBREW_LIBOMP_PREFIX)
        message(STATUS "Found Homebrew libomp at: ${HOMEBREW_LIBOMP_PREFIX}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp" CACHE STRING "" FORCE)
        set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "" FORCE)
        set(OpenMP_omp_LIBRARY "${HOMEBREW_LIBOMP_PREFIX}/lib/libomp.dylib" CACHE STRING "" FORCE)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -I${HOMEBREW_LIBOMP_PREFIX}/include")
        find_package(OpenMP)
    endif()
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(dqm_python PRIVATE OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP enabled: ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP not found. Building without OpenMP support (performance will be reduced).")
endif()

# Platform-specific settings
if(APPLE)
    set_target_properties(dqm_python PROPERTIES
        SUFFIX ".dylib"
        INSTALL_RPATH "@loader_path"
    )
elseif(UNIX)
    set_target_properties(dqm_python PROPERTIES
        SUFFIX ".so"
    )
endif()

# Release build optimizations (when not in debug mode)
if(NOT CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(dqm_python PRIVATE -O2 -DNDEBUG)
    # Note: -flto can cause issues on some platforms, omit for broader compatibility
endif()

# Install the library to the dqm/bin directory within the wheel
# scikit-build-core will use this install target
install(TARGETS dqm_python
    LIBRARY DESTINATION dqm/bin
    RUNTIME DESTINATION dqm/bin
)
