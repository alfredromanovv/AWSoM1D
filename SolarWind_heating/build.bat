@echo off
setlocal
cd /d "%~dp0"
if not exist build mkdir build


g++ -std=c++17 -O3 -march=native -fopenmp src/main.cpp -o build/sw_heating_1d.exe

if errorlevel 1 (
    echo [ERROR] Build failed.
    exit /b 1
)

echo Build OK: build\sw_heating_1d.exe
endlocal