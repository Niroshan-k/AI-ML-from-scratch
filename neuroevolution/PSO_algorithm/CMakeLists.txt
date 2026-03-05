cmake_minimum_required(VERSION 3.15)
project(PSO_Visualizer)

set(CMAKE_CXX_STANDARD 17)

# Tell CMake where to look for SDL3
list(APPEND CMAKE_PREFIX_PATH "C:/dev/SDL3")

find_package(SDL3 REQUIRED)

add_executable(pso_anim main.cpp)

# Link SDL3 to your executable
target_link_libraries(pso_anim PRIVATE SDL3::SDL3)