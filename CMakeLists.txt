cmake_minimum_required(VERSION 3.10)
project(HNSW_Project CXX)

# 设置 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 开启 O2 优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 确保 files 目录存在
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/files")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/files")
endif()

# 定义可执行文件目标
add_executable(main main.cpp)