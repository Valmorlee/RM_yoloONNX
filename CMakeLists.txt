cmake_minimum_required(VERSION 3.26)
project(RM_yoloONNX)

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

add_library(Lib_out SHARED IMPORTED)
set_target_properties( Lib_out PROPERTIES IMPORTED_LOCATION ${PROJECT_SOURCE_DIR}/lib/libdeploy.so )
add_library(Local_Lib_Static STATIC ${SRC})
add_library(Local_Lib_Shared SHARED ${SRC})

set_target_properties(Local_Lib_Shared PROPERTIES OUTPUT_NAME "Local_Lib")
set_target_properties(Local_Lib_Static PROPERTIES OUTPUT_NAME "Local_Lib")

set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/../lib)
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/out)

cuda_add_executable(RM_yoloONNX
        main.cpp
        ${SRC}
)

target_link_libraries(RM_yoloONNX ${OpenCV_LIBS})
