@echo off
call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat" -vcvars_ver=14.39

set "CMAKE_EXE=%~dp0cmake\cmake-3.29.3-windows-x86_64\bin\cmake.exe"

if exist build rmdir /s /q build

echo Generating build files...
"%CMAKE_EXE%" -G "NMake Makefiles" -B build

echo Compiling CUDA Engine in Release mode...
"%CMAKE_EXE%" --build build --config Release

echo ----------------------------------------------------
echo Build complete. Executable is located at:
echo .\build\cuda_engine.exe
echo ----------------------------------------------------
