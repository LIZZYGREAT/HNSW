cmake_minimum_required(VERSION 3.10)
project(FlatScan_Baseline CXX)

# 1. 设置 C++11 标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 2. 开启 -O2 编译优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

# 3. 自动检测并创建 files 目录（防止运行生成结果时因为找不到文件夹而报错）
if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/files")
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/files")
endif()

# 4. 定义生成的目标可执行文件
add_executable(main_baseline main.cpp)