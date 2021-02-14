cmake_minimum_required(VERSION 3.17)
project(ORB_SLAM2_Learning)

set(CMAKE_CXX_STANDARD 11)

find_package(OpenCV 3.4.12 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include/)

add_library(${PROJECT_NAME} SHARED src/ORBextractor.cc)

add_executable(paramid src/ORBextractor.cc)
target_link_libraries(paramid ${OpenCV_LIBS})
