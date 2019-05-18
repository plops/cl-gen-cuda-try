! pip install pycuda
import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.compiler
import numpy as np
mod=pycuda.compiler.SourceModule("""__global void dot(int *result, int *a, int *b) {
  {
    const int i = threadIdx.x;
    result = (result + a[i] + b[i]);
  }
}""")
multiply_them=mod.get_function("dot")