#include "../core/graph_runner.h"
#include <stdexcept>

CUDAGraphRunner::CUDAGraphRunner() : is_captured(false), graph(nullptr), instance(nullptr) {}

CUDAGraphRunner::~CUDAGraphRunner() {
    if (instance) cudaGraphExecDestroy(instance);
    if (graph) cudaGraphDestroy(graph);
}



void CUDAGraphRunner::execute(cudaStream_t stream) {
    if (!is_captured) throw std::runtime_error("Graph not captured!");
    
    cudaError_t err = cudaGraphLaunch(instance, stream);
    if (err != cudaSuccess) throw std::runtime_error(std::string("Failed to launch CUDA Graph: ") + cudaGetErrorString(err));
}


