@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cl.exe /EHsc /std:c++17 tools\preprocess.cpp data\tokenizer.cpp /Fe:preprocess.exe
preprocess.exe data\TinyStories_train.txt data\train_tokens.bin
