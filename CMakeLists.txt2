cmake_minimum_required(VERSION 3.10)
project(rockchip_npu)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories
include_directories(include)
include_directories("/opt/homebrew/opt/opencv/include/opencv4/opencv2")
include_directories("include/asio")

# Find OpenCV (required for main.cpp)
find_package(OpenCV REQUIRED)
if (OpenCV_FOUND)
    include_directories(${OpenCV_INCLUDE_DIRS})
    message(STATUS "OpenCV found at: " ${OpenCV_INCLUDE_DIRS}) #added debug message
else()
    message(FATAL_ERROR "OpenCV not found. Please install OpenCV and set OpenCV_DIR.")
endif()

# Source files
file(GLOB_RECURSE SOURCES
    "src/*.cpp"
    "*.hpp" # added .hpp files for completeness, in case they contain source code
    "include/*.hpp"
    "include/*.h"
    "*.h"
    "*.c"
    "*.cpp"
)


# Create executable
add_executable(rockchip_npu ${SOURCES})

# Link libraries
target_link_libraries(rockchip_npu ${OpenCV_LIBS})