cmake_minimum_required(VERSION 3.12)
project(FluidSimulation)

# Set C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add your source files
add_executable(fluid main.cpp)

# Link against OpenCV libraries
target_link_libraries(fluid ${OpenCV_LIBS})
option(WITH_CUDA "Include CUDA support" ON)