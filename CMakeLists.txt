cmake_minimum_required(VERSION 3.18)
project(CUDA_Engine_ViT LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_COMPILER_WORKS TRUE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --default-stream per-thread -w")

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# Main executable
file(GLOB_RECURSE CORE_SRC "core/*.cu" "core/*.cpp")
file(GLOB_RECURSE LAYERS_SRC "layers/*.cu" "layers/*.cpp")
file(GLOB_RECURSE TRAINING_SRC "training/*.cu" "training/*.cpp")
file(GLOB_RECURSE DATA_SRC "data/*.cu" "data/*.cpp")
file(GLOB_RECURSE INFERENCE_SRC "inference/*.cu" "inference/*.cpp")

list(REMOVE_ITEM TRAINING_SRC "${CMAKE_CURRENT_SOURCE_DIR}/training/train_loop.cu")

add_executable(cuda_engine main.cu ${CORE_SRC} ${LAYERS_SRC} ${TRAINING_SRC} ${DATA_SRC} ${INFERENCE_SRC})
set_target_properties(cuda_engine PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Tools executable
add_executable(preprocess tools/preprocess.cpp data/tokenizer.cpp)
