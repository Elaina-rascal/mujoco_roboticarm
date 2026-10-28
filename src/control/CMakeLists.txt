cmake_minimum_required(VERSION 3.15)
project(arm_control LANGUAGES CXX)

find_package(pinocchio REQUIRED)
add_executable(invkinematic
  src/invkinematic.cpp
  src/ik_controller.cpp
  src/pinocchio_wrapper.cpp
)
target_compile_features(invkinematic PRIVATE cxx_std_17)
target_include_directories(invkinematic PRIVATE include)
target_link_libraries(invkinematic PRIVATE pinocchio::pinocchio)
target_compile_definitions(invkinematic PRIVATE
  ARM_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../models/universal_robots_ur5e/ur5e.xml"
)
