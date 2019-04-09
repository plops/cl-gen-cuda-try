#include <stdio.h>
// https://www.youtube.com/watch?v=Ed_h2km0liI CUDACast #2 - Your First CUDA C
// Program
// https://github.com/NVIDIA-developer-blog/cudacasts/blob/master/ep2-first-cuda-c-program/kernel.cu
__global__ void vector_add(int *a, int *b, int *c, int n) {
  {
    int i = threadIdx.x;
    if ((i < n)) {
      c[i] = (a[i] + b[i]);
    }
  }
}
enum { N = 1024 };

int main() {
  {
    int *a = malloc((N * sizeof(int)));
    int *b = malloc((N * sizeof(int)));
    int *c = malloc((N * sizeof(int)));
    int *d_a;
    int *d_b;
    int *d_c;
    cudaMalloc(&d_a, (N * sizeof(int)));
    cudaMalloc(&d_b, (N * sizeof(int)));
    cudaMalloc(&d_c, (N * sizeof(int)));
    for (unsigned int i = 0; (i < N); i += 1) {
      a[i] = i;
      b[i] = i;
      c[i] = 0;
    }
    cudaMemcpy(d_a, a, (N * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, (N * sizeof(int)), cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, c, (N * sizeof(int)), cudaMemcpyHostToDevice);
    vector_add<<<1, N>>>(d_a, d_b, d_c, N);
    cudaMemcpy(c, d_c, (N * sizeof(int)), cudaMemcpyDeviceToHost);
    free(a);
    cudaFree(d_a);
    free(b);
    cudaFree(d_b);
    free(c);
    cudaFree(d_c);
    return 0;
  }
}