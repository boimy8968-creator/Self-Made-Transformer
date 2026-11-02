@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39
nvcc --default-stream per-thread -o train_loop.exe train_loop.cu tensor.cpp graph_runner.cpp autograd.cu fused_modules.cu bmm.cu add.cu layernorm.cu softmax.cu permute.cu scale.cu gelu.cu matadd.cu matmul.cu embedding.cu encoder.cu tokenizer.cpp optimizer.cu loss.cu

