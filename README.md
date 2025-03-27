# Intercept compilation instructions
nvcc -arch=sm_61 -v main.cu



nvcc -arch=sm_61 --ptx main.cu -o main.ptx

nvcc -arch=sm_61 -cubin main.ptx -o main.cubin

nvcc -arch=sm_61 --cubin main.ptx -o main.exe

chmod +x ./mainexe.