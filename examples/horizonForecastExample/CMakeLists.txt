# CMakeLists.txt for horizonForecastExample
cmake_minimum_required(VERSION 3.19)

project(horizonForecastExample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(horizonForecastExample main.cpp)

target_include_directories(horizonForecastExample PRIVATE ../../include)
