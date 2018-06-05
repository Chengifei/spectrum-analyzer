cmake_minimum_required(VERSION 3.10)
include(FindPkgConfig)
project(spectrum_analyzer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

pkg_check_modules(PKG REQUIRED python3 eigen3 portaudio-2.0 libswresample)
set(NUMPY_INCLUDE_DIR /usr/local/lib/python3.6/site-packages/numpy/core/include/)

include_directories(SYSTEM ${PKG_INCLUDE_DIRS} ${NUMPY_INCLUDE_DIR})
link_directories(${PKG_LIBRARY_DIRS})
link_libraries(${PKG_LIBRARIES})

add_library(spectrum_analyzer SHARED
        file.cpp
        file.hpp
        piano.hpp
        python_common.hpp
        spectrum_analyzer.cpp
        wrapper.cpp
        wrapper.hpp)

set_target_properties(spectrum_analyzer PROPERTIES SUFFIX .so PREFIX "")
