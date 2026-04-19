cmake_minimum_required(VERSION 3.15)
project(workspace_top_level)

# 自动把 src 下的所有子文件夹当作 CMake 项目加入
add_subdirectory(src/cmake_test)
add_subdirectory(src/control)