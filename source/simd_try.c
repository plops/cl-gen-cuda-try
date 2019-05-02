// gcc -std=c11 -Ofast -flto -ffast-math -march=skylake -msse2  -ftree-vectorize
// -mfma -mavx2

// clang -std=c11 -Ofast -flto -ffast-math -march=skylake -msse2
// -Rpass-analysis=loop-vectorize -Rpass=loop-vectorize
// -Rpass-missed=loop-vectorize

// icc -std=c11 -O2 -D NOFUNCCALL -qopt-report=1 -qopt-report-phase=vec
// -guide-vec -parallel

#include <complex.h>
#include <math.h>
#include <stdalign.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef CMPLXF
#define CMPLXF(real, imag) ((real) + _Complex_I * (imag))
#endif
typedef float vsf __attribute__((vector_size(64)));
;
float simd_16_fft_112_16_7(vsf *__restrict__ re_in, vsf *__restrict__ im_in,
                           vsf *__restrict__ re_out, vsf *__restrict__ im_out) {
  re_in = __builtin_assume_aligned(re_in, 64);
  im_in = __builtin_assume_aligned(im_in, 64);
  re_out = __builtin_assume_aligned(re_out, 64);
  im_out = __builtin_assume_aligned(im_out, 64);
  {
    static alignas(64) vsf x1_re[7];
    static alignas(64) vsf x1_im[7];
    const alignas(64) vsf con = {(0.0e+0f), (1.e+0f),  (2.e+0f),  (3.e+0f),
                                 (4.e+0f),  (5.e+0f),  (6.e+0f),  (7.e+0f),
                                 (8.e+0f),  (9.e+0f),  (1.e+1f),  (1.1e+1f),
                                 (1.2e+1f), (1.3e+1f), (1.4e+1f), (1.5e+1f)};
    x1_re[0] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                (con * re_in[96]));
    x1_re[16] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[32] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[48] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[64] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[80] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[96] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    memcpy(re_out, x1_re, sizeof(x1_re));
    return x1_re[0][0];
  }
}
void simd_driver() {
  {
    static vsf in_re[7];
    static vsf in_im[7];
    static vsf out_re[7];
    static vsf out_im[7];
    simd_16_fft_112_16_7(in_re, in_im, out_re, out_im);
    for (int i = 0; (i < 16); i += 1) {
      printf("%f\n", out_re[0][i]);
    }
  }
}
int main() {
  simd_driver();
  return 0;
}