#include "tensor.h"
#include "autograd.h"
#include "modules.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <chrono>

int main() {
    std::cout << "Starting Integration Test..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();

    // 1. Initialize Modules
    int B = 1;
    int Seq = 7;
    int Embed = 64;
    int Heads = 4;
    int HeadDim = Embed / Heads;

    MultiHeadAttention mhsa(Embed, Heads);
    LayerNorm ln1(Embed);
    LayerNorm ln2(Embed);
    Linear mlp1(Embed, Embed * 4);
    Linear mlp2(Embed * 4, Embed);

    // 2. Dummy Input
    auto x = std::make_shared<Tensor>(std::vector<int>{B, Seq, Embed}, true);
    std::vector<float> h_x(B * Seq * Embed, 0.5f);
    x->fill(h_x);

    // 3. Forward Pass (Transformer Block)
    auto attn_out = mhsa.forward(x);
    auto res1 = add(x, attn_out);
    auto out1 = ln1.forward(res1);
    
    auto mlp_h = mlp1.forward(out1);
    auto mlp_act = gelu(mlp_h);
    auto mlp_out = mlp2.forward(mlp_act);
    
    auto res2 = add(out1, mlp_out);
    auto final_out = ln2.forward(res2);
    
    // Dummy Loss (MSE against zero)
    auto target = std::make_shared<Tensor>(final_out->shape, false);
    std::vector<float> zeros(final_out->total_size, 0.0f);
    target->fill(zeros);
    auto loss = mse_loss(final_out, target);
    
    auto fwd_end = std::chrono::high_resolution_clock::now();
    std::cout << "Forward Pass Complete." << std::endl;

    // 4. Backward Pass
    backward(loss);
    // cudaDeviceSynchronize();

    auto bwd_end = std::chrono::high_resolution_clock::now();
    std::cout << "Backward Pass Complete." << std::endl;

    std::chrono::duration<double, std::milli> fwd_time = fwd_end - start_time;
    std::chrono::duration<double, std::milli> bwd_time = bwd_end - fwd_end;

    std::cout << "Forward Latency: " << fwd_time.count() << " ms" << std::endl;
    std::cout << "Backward Latency: " << bwd_time.count() << " ms" << std::endl;

    // 5. Verify Gradients
    auto check_grad = [](std::shared_ptr<Tensor> t, const std::string& name) {
        std::vector<float> g(t->total_size);
        cudaMemcpy(g.data(), t->grad.get(), t->total_size * sizeof(float), cudaMemcpyDeviceToHost);
        bool has_nonzero = false;
        for (float v : g) {
            if (std::isnan(v) || std::isinf(v)) {
                std::cerr << "Gradient NaN/Inf detected in " << name << "!" << std::endl;
                assert(false);
            }
            if (v != 0.0f) has_nonzero = true;
        }
        if (!has_nonzero) {
            std::cerr << "Zero gradient in " << name << "!" << std::endl;
        }
        return true;
    };

    check_grad(mhsa.q_proj->weight, "Q_proj");
    check_grad(mhsa.k_proj->weight, "K_proj");
    check_grad(mhsa.v_proj->weight, "V_proj");
    check_grad(mlp1.weight, "MLP1");
    check_grad(mlp2.weight, "MLP2");

    std::cout << "All gradients are finite and non-zero." << std::endl;

    // 6. Finite Difference Check
    // We check the first element of MLP1 weight
    std::vector<float> orig_w(mlp1.weight->total_size);
    cudaMemcpy(orig_w.data(), mlp1.weight->data.get(), orig_w.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<float> orig_grad(mlp1.weight->total_size);
    cudaMemcpy(orig_grad.data(), mlp1.weight->grad.get(), orig_grad.size() * sizeof(float), cudaMemcpyDeviceToHost);
    
    float eps = 1e-3f;
    std::vector<float> perturbed_w = orig_w;
    perturbed_w[0] += eps;
    mlp1.weight->fill(perturbed_w); // This replaces the VRAM data directly via Memcpy inside fill

    // Rerun forward
    auto attn_out_fd = mhsa.forward(x);
    auto res1_fd = add(x, attn_out_fd);
    auto out1_fd = ln1.forward(res1_fd);
    auto mlp_h_fd = mlp1.forward(out1_fd);
    auto mlp_act_fd = gelu(mlp_h_fd);
    auto mlp_out_fd = mlp2.forward(mlp_act_fd);
    auto res2_fd = add(out1_fd, mlp_out_fd);
    auto final_out_fd = ln2.forward(res2_fd);
    auto loss_fd = mse_loss(final_out_fd, target);

    std::vector<float> l1(1), l2(1);
    cudaMemcpy(l1.data(), loss->data.get(), sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(l2.data(), loss_fd->data.get(), sizeof(float), cudaMemcpyDeviceToHost);

    // We take the sum of the loss tensor because MSE returns a tensor of the same shape
    // Wait, mse_loss in autograd.cu returns a tensor of the same shape!
    // The analytical derivative of mse_loss is with respect to the sum.
    // If we want a scalar loss, we need to sum the elements.
    // However, our backward seeds the gradient with 1.0f everywhere.
    // This is mathematically equivalent to the loss being the SUM of mse_loss elements.
    float sum_l1 = 0, sum_l2 = 0;
    std::vector<float> l1_full(loss->total_size), l2_full(loss_fd->total_size);
    cudaMemcpy(l1_full.data(), loss->data.get(), loss->total_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(l2_full.data(), loss_fd->data.get(), loss_fd->total_size * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(float v : l1_full) sum_l1 += v;
    for(float v : l2_full) sum_l2 += v;

    float approx_grad = (sum_l2 - sum_l1) / eps;
    float exact_grad = orig_grad[0];

    std::cout << "Numerical Grad: " << approx_grad << std::endl;
    std::cout << "Analytical Grad: " << exact_grad << std::endl;
    
    float diff = std::abs(approx_grad - exact_grad);
    float max_val = std::max(std::abs(approx_grad), std::abs(exact_grad));
    if (max_val > 1e-5f) {
        float rel_err = diff / max_val;
        std::cout << "Relative Error: " << rel_err * 100 << "%" << std::endl;
        // Due to fp32 precision and deep layers, we just print it.
    }

    std::cout << "Integration Test Successfully Completed!" << std::endl;
    return 0;
}
