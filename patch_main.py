import re

with open('main.cpp', 'r') as f:
    content = f.read()

# Update V = 3000 to V = 30000
content = re.sub(r'int V\s*=\s*\d+;', 'int V = 30000;', content)

# Remove dummy dataset creation and update loader to train_tokens.bin
content = re.sub(r'std::ofstream dummy_data[\s\S]*?dummy_data\.close\(\);\s*', '', content)
content = re.sub(r'BinaryDataLoader loader\("[^"]+",\s*B,\s*S\);', 'BinaryDataLoader loader("data/train_tokens.bin", B, S);', content)

# Fix CUDA graph trap
old_graph_capture = r'''runner\.capture\(\[&\]\(\)\s*\{
\s*auto input_ids = loader\.get_next_batch\(B, S\);
\s*// Cast input float tokens to integer targets and shift
\s*int threads = 256;
\s*int blocks = \(B \* S \+ threads - 1\) / threads;
\s*cast_and_shift_targets_kernel<<<blocks, threads>>>\(input_ids->data\.get\(\), d_targets, B, S\);
\s*// Forward Pass
\s*auto encoded = model\.forward\(input_ids\);'''

new_graph_capture = r'''auto d_input_tensor = std::make_shared<Tensor>(std::vector<int>{B, S}, false);
        
        runner.capture([&]() {
            int threads = 256;
            int blocks = (B * S + threads - 1) / threads;
            cast_and_shift_targets_kernel<<<blocks, threads>>>(d_input_tensor->data.get(), d_targets, B, S);
            
            auto encoded = model.forward(d_input_tensor);'''

content = re.sub(old_graph_capture, new_graph_capture, content)

# Update execution loop
old_loop = r'''for \(int i = 1; i <= steps; \+\+i\) \{
\s*runner\.execute\(\);'''

new_loop = r'''for (int i = 1; i <= steps; ++i) {
            auto batch = loader.get_next_batch(B, S);
            cudaMemcpyAsync(d_input_tensor->data.get(), batch->data.get(), B * S * sizeof(float), cudaMemcpyDeviceToDevice);
            runner.execute();'''

content = re.sub(old_loop, new_loop, content)

# Also update step count from 2000 to 10000
content = re.sub(r'int steps = \d+;', 'int steps = 10000;', content)

with open('main.cpp', 'w') as f:
    f.write(content)
