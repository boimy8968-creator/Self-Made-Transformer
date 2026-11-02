#pragma once
#include <cuda_runtime.h>


class CUDAGraphRunner {
public:
    cudaGraph_t graph;
    cudaGraphExec_t instance;
    bool is_captured;

    CUDAGraphRunner();
    ~CUDAGraphRunner();

    // Captures the execution of the provided function into a cudaGraph
    template <typename F> void capture(F work, cudaStream_t stream = cudaStreamPerThread) { if (is_captured) { cudaGraphExecDestroy(instance); cudaGraphDestroy(graph); is_captured = false; } cudaError_t err = cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal); if (err != cudaSuccess) throw std::runtime_error("Failed to begin CUDA Graph capture."); work(); err = cudaStreamEndCapture(stream, &graph); if (err != cudaSuccess) throw std::runtime_error(std::string("Failed to end CUDA Graph capture: ") + cudaGetErrorString(err)); err = cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0); if (err != cudaSuccess) throw std::runtime_error("Failed to instantiate CUDA Graph."); is_captured = true; }
    
    // Launches the captured graph
    void execute(cudaStream_t stream = cudaStreamPerThread);
};

