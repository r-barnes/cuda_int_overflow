// C++ code pretending to be CUDA code to test compilation

#include <cstdint>

void kernel(int* data, uint32_t N){
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for(int i = tid; i <= N; i += blockDim.x * gridDim.x) {
        tid * 4;
    }
}

int main(){
    int data[10];
    kernel(data, 10);
}