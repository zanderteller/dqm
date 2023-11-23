include_directories(include) # relative include path (for Eigen library)

# Check if DEBUG is defined
option(DEBUG "Enable debugging" OFF)
if(DEBUG)
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS_DEBUG "-g3")
else()
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS_RELEASE "-O2 -flto -DNDEBUG")
endif()

# Detect operating system
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")

    # Mac OS
    # Check if CONDA_PREFIX is defined
    if(NOT DEFINED ENV{CONDA_PREFIX})
        message(FATAL_ERROR "CONDA_PREFIX is not defined. Please activate the conda environment.")
    endif()
    set(CONDA_PREFIX $ENV{CONDA_PREFIX})
    set(OPENMPFLAGS "-I/usr/local/include -I${CONDA_PREFIX}/include -Xpreprocessor -fopenmp")
    set(OPENMPLIB "-L/usr/local/lib -Wl,-rpath,${CONDA_PREFIX}/lib -L${CONDA_PREFIX}/lib -lomp")
else()
    # Assuming Linux
    set(DLL "so")
    set(OPENMPFLAGS "-Xpreprocessor -fopenmp")
    set(OPENMPLIB "-fopenmp")
endif()

# Source files
set(SOURCE_FILES utils.cpp array3d.cpp basis.cpp trajectories.cpp operators.cpp dqm_python.cpp)

# Add shared library target
add_library(dqm_python SHARED ${SOURCE_FILES})
set_target_properties(dqm_python
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/dqm/bin
)
target_compile_options(dqm_python PRIVATE ${CMAKE_CXX_FLAGS} ${OPENMPFLAGS})
target_link_libraries(dqm_python ${OPENMPLIB})
target_compile_definitions(dqm_python
                           PRIVATE VERSION_INFO=${DQM_VERSION_INFO})
