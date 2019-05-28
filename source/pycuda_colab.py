import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.compiler
import numpy as np
mod=pycuda.compiler.SourceModule("""
__global__ void cu_mul(float *result, float *a, float *b) {
  {
    const int i = threadIdx.x;
    result[i] = (a[i] * b[i]);
  }
}
""")
cu_mul=mod.get_function("cu_mul")
n=400
a=np.random.randn(n).astype(np.float32)
b=np.random.randn(n).astype(np.float32)
result=np.zeros_like(a)
cu_mul(drv.Out(result), drv.In(a), drv.In(b), block=(n,1,1,))
print(result)