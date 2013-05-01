project(cxx_ext)
cmake_minimum_required(VERSION 2.8)

# This is not a robust CMake File. It only works on my system

#set(BOOST_INCLUDE "C:/boost_1_53_0")
#set(BOOST_STAGE_LIB "C:/boost_1_53_0/stage/lib")
#link_directories(${BOOST_STAGE_LIB})
#include_directories (${BOOST_INCLUDE}) 


# get boost
#set(Boost_USE_STATIC_LIBS   OFF)
#set(Boost_USE_MULTITHREADED OFF)

find_package(PythonLibs REQUIRED)

find_package(Boost COMPONENTS
                system
                thread
                python
             REQUIRED)


set(NUMPY_INCLUDE "C:/Python27/Lib/site-packages/numpy/core/include")

include_directories (${NUMPY_INCLUDE}) 
include_directories (${PYTHON_INCLUDE_DIRS}) 
include_directories(${Boost_INCLUDE_DIRS})

add_library(hs_boostpy SHARED hello_world.cxx)

target_link_libraries(hs_boostpy
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES}
    )

