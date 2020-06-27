cmake_minimum_required(VERSION 3.16)
project(GaClxSolution)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_FLAGS "-fopenmp -Wall -Wextra")
if(${CMAKE_SYSTEM_NAME} STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS "-pthread ${CMAKE_CXX_FLAGS}")
endif()

add_executable(GaClxSolution main.cpp)
