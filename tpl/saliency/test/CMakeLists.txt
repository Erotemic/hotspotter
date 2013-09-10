cmake_minimum_required(VERSION 2.8)
#load library path
# usr/local/lib ...
# otool -L 
# @executable_path <- probably want todo this
#DYLIB_PATH ENVAR Maybe set? 
project(HessAff)
set(CMAKE_BUILD_TYPE "Release")
find_package( OpenCV REQUIRED )
# This should be used but it is not. Weird.
include_directories(${OpenCV_INCLUDE_DIR})
# Apparently this seems to be used
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "Defining saliency executable")
add_executable(saliency main.cpp saliency.cpp)
message(STATUS "Specifing libraries to link to")
target_link_libraries(saliency ${OpenCV_LIBS})
