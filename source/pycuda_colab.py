! pip install pycuda
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.compiler
import numpy as np
mod=pycuda.compiler.SourceModule("""__global__ void cu_dot(int *result, int *a, int *b) {
  {
    const int i = threadIdx.x;
    result = (result + a[i] + b[i]);
  }
}
""")
cu_dot=mod.get_function("cu_dot")
a=np.random.randint(1, 20, 5)
b=np.random.randint(1, 20, 5)
result=0
cu_dot(drv.Out(result), drv.In(a), drv.In(b), block=(5,1,1,))
print(result)