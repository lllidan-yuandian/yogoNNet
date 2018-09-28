cmake_minimum_required(VERSION 3.2)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_C_FLAGS_RELEASE "-O3")

#add_definitions("-Wall -g")
if(OPENMP_FOUND)
    message("OPENMP FOUND")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

PROJECT(yogoNNet)

set (CMAKE_CXX_STANDARD 11)
include_directories(
        ./source/header
        ./../../eigen-git-mirror
)

set(SOURCE_FILES
        ./source/src/layers.cpp
        ./source/src/net.cpp
        )

add_library(yogoNNet ${SOURCE_FILES})
target_link_libraries(yogoNNet openblas)

add_executable(yogoNNetTest ./test/main_test.cpp)
target_link_libraries(yogoNNetTest yogoNNet)
