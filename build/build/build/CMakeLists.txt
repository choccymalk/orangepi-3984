cmake_minimum_required(VERSION 3.6)

project(rockchip_npu)
set(CMAKE_CROSSCOMPILING TRUE)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++23 -Ofast")

# OpenCV includes and libraries
include_directories(/usr/include/opencv4)
link_directories(/usr/lib/aarch64-linux-gnu)

# rknn api
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB "${CMAKE_SOURCE_DIR}/3rdparty/static/librknnrt.so")
include_directories(include)

set(CMAKE_INSTALL_RPATH "lib")

add_executable(rockchip_npu
    src/main.cpp
)

target_link_libraries(rockchip_npu
    ${RKNN_RT_LIB}
    /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0
    /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0
    /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.6.0
    -lpthread
    -ldl
)

# install target and libraries
install(TARGETS rockchip_npu DESTINATION ./)
