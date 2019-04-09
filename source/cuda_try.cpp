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
enum(N 1024){};

int main() {
  {
    int *a = malloc((N * not processable : (sizeof int)));
    int *b = malloc((N * not processable : (sizeof int)));
    int *c = malloc((N * not processable : (sizeof int)));
    int *d_a;
    int *d_b;
    int *d_c;
  }
}