cmake_minimum_required(VERSION 3.26)
project(RT-Yolo)

set(CMAKE_CXX_STANDARD 20)

# 旧协议指令屏蔽
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

#========== cuda ==========
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
find_package(CUDA REQUIRED)

include_directories(/usr/local/cuda/include)
link_directories(/usr/local/cuda/lib64)

include_directories(/home/valmorx/DeepLearningSource/cudnn-8.9.7/include)
link_directories(/home/valmorx/DeepLearningSource/cudnn-8.9.7/lib)

#========== OpenCV ==========
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

aux_source_directory(${PROJECT_BINARY_DIR}/../src SRC)
include_directories(${PROJECT_BINARY_DIR}/../include)
link_directories(${PROJECT_BINARY_DIR}/../lib/)

#========= TENSORRT =========

include_directories(/home/valmorx/DeepLearningSource/TensorRT-10.9.0.34/include)
link_directories(/home/valmorx/DeepLearningSource/TensorRT-10.9.0.34/lib)

#========== EXTRA ==========

set(RT_YOLO_ROOT /home/valmorx/DeepLearningSource/TensorRT-YOLO)
include_directories(/home/valmorx/DeepLearningSource/TensorRT-YOLO)
link_directories(/home/valmorx/DeepLearningSource/TensorRT-YOLO/lib)
find_package(Python3 COMPONENTS Interpreter Development)
include_directories(${Python3_INCLUDE_DIRS})


message(${PROJECT_BINARY_DIR})
message(${PROJECT_SOURCE_DIR})

SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/out)
aux_source_directory(/home/valmorx/DeepLearningSource/TensorRT-YOLO/deploy YOLO_SRC)

add_executable(RT-Yolo
        RT-Yolo.cpp
        ${SRC}
        ${YOLO_SRC}
)

target_link_libraries(RT-Yolo
        ${OpenCV_LIBS}
        ${Python3_LIBRARIES}
        /home/valmorx/DeepLearningSource/TensorRT-YOLO/lib/libdeploy.so


)
