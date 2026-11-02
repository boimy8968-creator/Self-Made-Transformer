#pragma once
#include "../core/tensor.h"
#include <vector>
#include <memory>

class AdamW {
public:
    AdamW(std::vector<std::shared_ptr<Tensor>> parameters, 
          float lr = 1e-3, 
          float beta1 = 0.9, 
          float beta2 = 0.999, 
          float eps = 1e-8, 
          float weight_decay = 0.01);
    ~AdamW();
    
    void step();
    void zero_grad();

public:
    std::vector<std::shared_ptr<Tensor>> params;
    float lr, beta1, beta2, eps, weight_decay;
    int t;
    
    // First and second moment buffers kept solely in VRAM
    std::vector<float*> m_buffers;
    std::vector<float*> v_buffers;
};

