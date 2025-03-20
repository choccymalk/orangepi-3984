cmake_minimum_required(VERSION 3.6)
project(rockchip_npu)

# Set cross-compilation settings
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(CMAKE_C_COMPILER aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)

# Set the sysroot
set(CMAKE_SYSROOT /usr/aarch64-linux-gnu)
set(CMAKE_LIBRARY_PATH ${CMAKE_SYSROOT}/lib ${CMAKE_SYSROOT}/usr/lib)
set(CMAKE_FIND_ROOT_PATH ${CMAKE_SYSROOT})

# Configure cross-compilation search
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -march=armv8.2-a -Ofast ${TORCH_CXX_FLAGS}")

# Set Torch paths
set(TORCH_INSTALL_PREFIX "${CMAKE_SOURCE_DIR}/3rdparty/libtorch")
set(CMAKE_PREFIX_PATH ${CMAKE_PREFIX_PATH} "${TORCH_INSTALL_PREFIX}")
set(Torch_DIR "${TORCH_INSTALL_PREFIX}/share/cmake/Torch")

# These variables tell OpenCV to avoid using LAPACK/BLAS when possible
set(WITH_LAPACK OFF CACHE BOOL "Build without LAPACK support")
set(WITH_OPENCL OFF CACHE BOOL "Build without OpenCL support")
set(BUILD_opencv_dnn OFF CACHE BOOL "Don't build DNN module")

# Find OpenCV
set(OpenCV_DIR /usr/lib/aarch64-linux-gnu/cmake/opencv4)
find_package(OpenCV 4 REQUIRED)

# Explicitly set Torch library paths
set(TORCH_LIBRARIES
    "${TORCH_INSTALL_PREFIX}/lib/libtorch.so"
    "${TORCH_INSTALL_PREFIX}/lib/libtorch_cpu.so"
    "${TORCH_INSTALL_PREFIX}/lib/libc10.so"
)
set(TORCH_INCLUDE_DIRS "${TORCH_INSTALL_PREFIX}/include")

# Add the libtorch library directory to the runtime library path
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH};${TORCH_INSTALL_PREFIX}/lib")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS}/torch/csrc/api/include)
include_directories(include)

# Set library search paths
link_directories(/usr/lib/aarch64-linux-gnu)
link_directories("${TORCH_INSTALL_PREFIX}/lib")

# Use a stubbed liblapack.so
file(WRITE ${CMAKE_BINARY_DIR}/lapack_stub.c "
// Empty implementations for required LAPACK/BLAS functions
void dgemm_() {}
void sgemm_() {}
")
add_library(lapack_stub STATIC ${CMAKE_BINARY_DIR}/lapack_stub.c)
set_target_properties(lapack_stub PROPERTIES OUTPUT_NAME "lapack")

# rknn api
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB "${CMAKE_SOURCE_DIR}/3rdparty/static/librknnrt.so")

add_executable(rockchip_npu src/main-torch.cpp)

# Disable the linking of the real liblapack.so
target_link_options(rockchip_npu PRIVATE 
    "-Wl,--allow-shlib-undefined"
    "-Wl,--copy-dt-needed-entries"
)

target_link_libraries(rockchip_npu
    ${RKNN_RT_LIB}
    ${OpenCV_LIBS}
    ${TORCH_LIBRARIES}
    -L/usr/lib/aarch64-linux-gnu
    -L${CMAKE_BINARY_DIR}
    -ltbb
    -lz
    -Wl,-Bstatic
    ${CMAKE_BINARY_DIR}/liblapack.a
    -Wl,-Bdynamic
    -lavcodec
    -lavformat
    -lavutil
    -lswscale
    -lgstreamer-1.0
    -lgstapp-1.0
    -lgstbase-1.0
    -lgstriff-1.0
    -lgstvideo-1.0
    -lgstaudio-1.0
    -lgstpbutils-1.0
    -lgobject-2.0
    -lglib-2.0
    -ldc1394
    -lgphoto2
    -lgphoto2_port
    -lpthread
    -ldl
    -lm
)

# Set RPATH
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")
set(CMAKE_BUILD_WITH_INSTALL_RPATH TRUE)

# Install targets
install(TARGETS rockchip_npu DESTINATION bin)
install(FILES ${RKNN_RT_LIB} DESTINATION lib)
install(TARGETS lapack_stub DESTINATION lib)

message(STATUS "Configuration complete. Run the build.sh script to build and prepare deployment package.")
