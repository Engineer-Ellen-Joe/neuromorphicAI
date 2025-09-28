@echo off

REM MSYS2 PATH 덮어쓰기
set PATH=C:\msys64\ucrt64\bin;C:\msys64\usr\bin;%PATH%

if not exist build mkdir build
cd build
cmake -S ../ -B . -G "MSYS Makefiles" -DCMAKE_MAKE_PROGRAM=C:/msys64/ucrt64
C:/msys64/usr/bin/make.exe && C:/msys64/usr/bin/make.exe Shaders
start LveEngine.exe
cd..