cmake_minimum_required(VERSION 3.7)
project(ML)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES BP.cpp BP.hpp MFNN.cpp MFNN.hpp)
add_executable(MLtest ${SOURCE_FILES})
