#include <iostream>
#include <random>
#include <type_traits>


// CUDA API error handler
#define CUDA_CHECK(call) \
    if ((call) != cudaSuccess) { \
        const auto err = cudaGetLastError(); \
        std::cerr << "CUDA error in " << __FILE__ << " at line " << __LINE__ << ": " \
                  << cudaGetErrorString(err) << std::endl; \
        exit(1); \
    }


// Kernel function to multiply two arrays
__global__ void multiply(uint32_t *a, uint32_t *b, uint32_t *c, uint32_t N) {
  const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < N) {
      c[idx] = a[idx] * b[idx];
  }
}

int main() {
  // Size of the arrays
  uint32_t N = 1000;

  std::mt19937 rng;
  // rng.seed(std::random_device()());
  std::uniform_int_distribution<std::mt19937::result_type> dist(0, std::numeric_limits<uint32_t>::max());

  // Host arrays
  std::vector<uint32_t> a(N);
  std::vector<uint32_t> b(N);
  std::vector<uint32_t> c(N);

  // Fill a and b with random numbers between 0 and INT_MAX
  for (int i = 0; i < N; i++) {
      a[i] = dist(rng);
      b[i] = dist(rng);
  }

  // Device arrays
  uint32_t *d_a, *d_b, *d_c;
  CUDA_CHECK(cudaMalloc(&d_a, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_b, N * sizeof(uint32_t)));
  CUDA_CHECK(cudaMalloc(&d_c, N * sizeof(uint32_t)));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(d_a, a.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, b.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice));

  // Launch the kernel
  const int threadsPerBlock = 256;
  const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  multiply<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
  // Check that the kernel launched successfully
  CUDA_CHECK(cudaPeekAtLastError());

  // Copy data from device to host
  CUDA_CHECK(cudaMemcpy(c.data(), d_c, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));

  // Print the result
  for (int i = 0; i < N; i++) {
      std::cout << c[i] << " ";
  }
  std::cout << std::endl;

  // Free device memory
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));

  return 0;
}