cmake_minimum_required(VERSION 3.15)
project(mujoco_roboticarm LANGUAGES CXX)

# Python 仿真直接运行脚本；C++ IK 示例按需编译。
option(BUILD_CPP_IK "Build the standalone Pinocchio IK example" OFF)
if(BUILD_CPP_IK)
  add_subdirectory(src/control)
endif()
