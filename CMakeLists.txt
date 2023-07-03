cmake_minimum_required(VERSION 3.10)
include(FindPkgConfig)
project(spectrum_analyzer)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)

pkg_check_modules(PKG REQUIRED eigen3 portaudio-2.0 libswresample)
find_package(Python3 COMPONENTS Interpreter Development NumPy)

include_directories(SYSTEM ${PKG_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${Python3_NumPy_INCLUDE_DIRS})
link_directories(${PKG_LIBRARY_DIRS})
link_libraries(${PKG_LIBRARIES})
add_compile_definitions(NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

add_library(spectrum_analyzer SHARED
        file.cpp
        file.hpp
        piano.hpp
        python_common.hpp
        spectrum_analyzer.cpp
        wrapper.cpp
        wrapper.hpp)

set_target_properties(spectrum_analyzer PROPERTIES SUFFIX .so PREFIX "")
