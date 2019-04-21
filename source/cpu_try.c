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
float complex *dft16_slow(float complex *__restrict__ a) {
  a = __builtin_assume_aligned(a, 64);
  {
    static alignas(64) float complex y[16] = {0.0fi};
    memset(y, 0, (16 * sizeof(complex float)));
    for (int j = 0; (j < 16); j += 1) {
      for (int k = 0; (k < 16); k += 1) {
        y[j] =
            (y[j] + (a[k] * cexpf((1.0fi * (-3.9269908169872414e-1) * j * k))));
      }
    }
    return y;
  }
}
float complex *dft256_slow(float complex *__restrict__ a) {
  a = __builtin_assume_aligned(a, 64);
  {
    static alignas(64) float complex y[256] = {0.0fi};
    memset(y, 0, (256 * sizeof(complex float)));
    for (int j = 0; (j < 256); j += 1) {
      for (int k = 0; (k < 256); k += 1) {
        y[j] =
            (y[j] + (a[k] * cexpf((1.0fi * (-2.454369260617026e-2) * j * k))));
      }
    }
    return y;
  }
}
float complex *fft_21_3_7(float complex *__restrict__ x) {
  // tell compiler that argument ins 64byte aligned;
  x = __builtin_assume_aligned(x, 64);
  // n1 DFTs of size n2 in the column direction;
  {
    static alignas(64) float complex x1[21] = {(0.0e+0f)};
    x1[0] = (x[0] + x[1] + x[2]);
    x1[3] = (x[3] + x[4] + x[5]);
    x1[6] = (x[6] + x[7] + x[8]);
    x1[9] = (x[9] + x[10] + x[11]);
    x1[12] = (x[12] + x[13] + x[14]);
    x1[15] = (x[15] + x[16] + x[17]);
    x1[18] = (x[18] + x[19] + x[20]);
    x1[1] = (x[0] + x[1] + x[2]);
    x1[4] = ((x[3] * w7m1_7) + (x[4] * w7m1_7) + (x[5] * w7m1_7));
    x1[7] = ((x[6] * w7p5_7) + (x[7] * w7p5_7) + (x[8] * w7p5_7));
    x1[10] = ((x[9] * w7p4_7) + (x[10] * w7p4_7) + (x[11] * w7p4_7));
    x1[13] = ((x[12] * w7p3_7) + (x[13] * w7p3_7) + (x[14] * w7p3_7));
    x1[16] = ((x[15] * w7p2_7) + (x[16] * w7p2_7) + (x[17] * w7p2_7));
    x1[19] = ((x[18] * w7p1_7) + (x[19] * w7p1_7) + (x[20] * w7p1_7));
    x1[2] = (x[0] + x[1] + x[2]);
    x1[5] = ((x[3] * w7p5_7) + (x[4] * w7p5_7) + (x[5] * w7p5_7));
    x1[8] = ((x[6] * w7p3_7) + (x[7] * w7p3_7) + (x[8] * w7p3_7));
    x1[11] = ((x[9] * w7p1_7) + (x[10] * w7p1_7) + (x[11] * w7p1_7));
    x1[14] = ((x[12] * w7m1_7) + (x[13] * w7m1_7) + (x[14] * w7m1_7));
    x1[17] = ((x[15] * w7p4_7) + (x[16] * w7p4_7) + (x[17] * w7p4_7));
    x1[20] = ((x[18] * w7p2_7) + (x[19] * w7p2_7) + (x[20] * w7p2_7));
  }
}