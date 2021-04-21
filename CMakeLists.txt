cmake_minimum_required(VERSION 3.4.1)
set(CMAKE_CXX_STANDARD 17)

find_package(OpenCV REQUIRED)
MESSAGE(STATUS "OpenCV version: " ${OpenCV_VERSION})
option(BUILD_MNN"infer mnn" OFF)

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib/)
#link_directories(/usr/local/lib/)
include_directories(
#        3rdparty/darwin/include
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
)
#link_directories(3rdparty/darwin/lib)

set(CMAKE_CXX_STANDARD 14)

add_executable(test_images samples/test_images.cpp src/dnn_detector/code_detect.cpp)
target_link_libraries(
        test_images
        ${OpenCV_LIBS}
#        mnn
)