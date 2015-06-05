CMAKE_MINIMUM_REQUIRED(VERSION 2.6.2 FATAL_ERROR)

if (CMAKE_MAJOR_VERSION VERSION_GREATER 2)
if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
    cmake_policy(SET CMP0026 OLD)
    cmake_policy(SET CMP0042 OLD)
endif(COMMAND cmake_policy)
endif()

project (EUCLID)

# bring in custom modules
add_subdirectory (CMake)

#message(STATUS "Configuring Euclid ${HOOMD_VERSION_LONG}")

#################################
## setup python library and executable
# find the python libraries to link to
include(HOOMDPythonSetup)
# Find the boost libraries and set them up
include (HOOMDBoostSetup)
# setup numpy
include (HOOMDNumpySetup)

# setup the library compile mode
if(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS OFF)
else(ENABLE_STATIC)
    set(BUILD_SHARED_LIBS ON)
endif(ENABLE_STATIC)

################################
## Process subdirectories
add_subdirectory (src)
