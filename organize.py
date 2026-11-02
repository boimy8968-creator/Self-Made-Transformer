import os
import shutil
import re

folders = {
    'core': ['tensor.h', 'tensor.cpp', 'autograd.h', 'autograd.cu', 'graph_runner.h', 'graph_runner.cpp'],
    'layers': ['modules.h', 'fused_modules.h', 'fused_modules.cu', 'layernorm.h', 'layernorm.cu', 'gelu.h', 'gelu.cu', 'softmax.h', 'softmax.cu', 'bmm.h', 'bmm.cu', 'embedding.h', 'embedding.cu', 'encoder.h', 'encoder.cu', 'scale.h', 'scale.cu', 'add.h', 'add.cu', 'permute.h', 'permute.cu'],
    'training': ['loss.h', 'loss.cu', 'optimizer.h', 'optimizer.cu', 'checkpoint.h', 'checkpoint.cpp', 'clip_grad.h', 'clip_grad.cu', 'train_loop.cu'],
    'data': ['tokenizer.h', 'tokenizer.cpp', 'data_loader.h', 'data_loader.cpp'],
    'inference': ['inference.h', 'inference.cu', 'kv_cache.h', 'kv_cache.cu', 'sampling.h', 'sampling.cu'],
    'tools': ['preprocess.cpp']
}

# Create folders and move files
file_to_folder = {}
for folder, files in folders.items():
    if not os.path.exists(folder):
        os.makedirs(folder)
    for f in files:
        if os.path.exists(f):
            shutil.move(f, os.path.join(folder, f))
            file_to_folder[f] = folder

# Update #include in all files
def update_includes(file_path, is_root=False):
    with open(file_path, 'r') as f:
        content = f.read()
        
    def replace_include(match):
        inc_file = match.group(1)
        if inc_file in file_to_folder:
            if is_root:
                return f'#include "{file_to_folder[inc_file]}/{inc_file}"'
            else:
                return f'#include "../{file_to_folder[inc_file]}/{inc_file}"'
        return match.group(0)
    
    new_content = re.sub(r'#include\s+"([^"]+)"', replace_include, content)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)

for folder, files in folders.items():
    for f in files:
        path = os.path.join(folder, f)
        if os.path.exists(path):
            update_includes(path, is_root=False)

if os.path.exists('main.cpp'):
    update_includes('main.cpp', is_root=True)

print("Phase 1 done.")
