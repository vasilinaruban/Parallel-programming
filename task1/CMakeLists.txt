cmake_minimum_required(VERSION 3.26)
project(paral_task1)

set(CMAKE_CXX_STANDARD 17)

option(float "use_float" OFF)

if(float)
    add_compile_definitions(FLOAT_ARRAY)
    add_executable(paral_task1_float main.cpp)
else()
    add_executable(paral_task1_double main.cpp)
endif()