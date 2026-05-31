cmake_minimum_required(VERSION 3.10)
project(HNSW_Parallel)

# 使用 C++17 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 核心优化参数：开启 O3 优化
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall")

# 寻找并链接 OpenMP
find_package(OpenMP REQUIRED)

# 编译生成可执行文件
add_executable(hnsw_parallel main.cpp)

# 链接 OpenMP 库
if(OpenMP_CXX_FOUND)
    target_link_libraries(hnsw_parallel PUBLIC OpenMP::OpenMP_CXX)
endif()