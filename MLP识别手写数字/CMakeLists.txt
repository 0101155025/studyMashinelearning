cmake_minimum_required(VERSION 3.10)
project("MachineLearning" CXX)

# 设置 C++20 标准
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ========== 核心修复：强制添加 OpenMP 选项 ==========
# 给所有模式添加 -fopenmp（编译+链接一体，MinGW下这个选项会自动链接libgomp）
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

# 保留你的优化配置（Debug/Release）
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -ffast-math")

# 添加可执行文件
add_executable(machinelearning
        main.cpp
        MNISTReader.cpp
        network.cpp
)

# 手动指定链接 libgomp（兜底，避免极少数环境找不到）
target_link_libraries(machinelearning PRIVATE gomp)