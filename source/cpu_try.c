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
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm;
float complex *dft_21(float complex *__restrict__ a) {
  a = __builtin_assume_aligned(a, 64);
  {
    static alignas(64) float complex y[21] = {0.0fi};
    for (int j = 0; (j < 21); j += 1) {
      for (int k = 0; (k < 21); k += 1) {
        y[j] =
            (y[j] + (a[k] * cexpf((1.0fi * (-2.99199300341885e-1) * j * k))));
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
    static alignas(64) float complex x1[21];
    const float complex w7m1_7 =
        ((6.234898018587335e-1) + (-7.818314824680297e-1i));
    const float complex w7p5_7 =
        ((-2.2252093395631434e-1) + (-9.749279121818235e-1i));
    const float complex w7p4_7 =
        ((-9.009688679024189e-1) + (-4.3388373911755823e-1i));
    const float complex w7p3_7 =
        ((-9.009688679024191e-1) + (4.33883739117558e-1i));
    const float complex w7p2_7 =
        ((-2.2252093395631461e-1) + (9.749279121818235e-1i));
    const float complex w7p1_7 =
        ((6.234898018587334e-1) + (7.818314824680298e-1i));
    x1[0] = (x[0] + x[3] + x[6] + x[9] + x[12] + x[15] + x[18]);
    x1[1] = (x[1] + x[4] + x[7] + x[10] + x[13] + x[16] + x[19]);
    x1[2] = (x[2] + x[5] + x[8] + x[11] + x[14] + x[17] + x[20]);
    x1[3] = (x[0] + (x[3] * w7m1_7) + (x[6] * w7p5_7) + (x[9] * w7p4_7) +
             (x[12] * w7p3_7) + (x[15] * w7p2_7) + (x[18] * w7p1_7));
    x1[4] = (x[1] + (x[4] * w7m1_7) + (x[7] * w7p5_7) + (x[10] * w7p4_7) +
             (x[13] * w7p3_7) + (x[16] * w7p2_7) + (x[19] * w7p1_7));
    x1[5] = (x[2] + (x[5] * w7m1_7) + (x[8] * w7p5_7) + (x[11] * w7p4_7) +
             (x[14] * w7p3_7) + (x[17] * w7p2_7) + (x[20] * w7p1_7));
    x1[6] = (x[0] + (x[3] * w7p5_7) + (x[6] * w7p3_7) + (x[9] * w7p1_7) +
             (x[12] * w7m1_7) + (x[15] * w7p4_7) + (x[18] * w7p2_7));
    x1[7] = (x[1] + (x[4] * w7p5_7) + (x[7] * w7p3_7) + (x[10] * w7p1_7) +
             (x[13] * w7m1_7) + (x[16] * w7p4_7) + (x[19] * w7p2_7));
    x1[8] = (x[2] + (x[5] * w7p5_7) + (x[8] * w7p3_7) + (x[11] * w7p1_7) +
             (x[14] * w7m1_7) + (x[17] * w7p4_7) + (x[20] * w7p2_7));
    x1[9] = (x[0] + (x[3] * w7p4_7) + (x[6] * w7p1_7) + (x[9] * w7p5_7) +
             (x[12] * w7p2_7) + (x[15] * w7m1_7) + (x[18] * w7p3_7));
    x1[10] = (x[1] + (x[4] * w7p4_7) + (x[7] * w7p1_7) + (x[10] * w7p5_7) +
              (x[13] * w7p2_7) + (x[16] * w7m1_7) + (x[19] * w7p3_7));
    x1[11] = (x[2] + (x[5] * w7p4_7) + (x[8] * w7p1_7) + (x[11] * w7p5_7) +
              (x[14] * w7p2_7) + (x[17] * w7m1_7) + (x[20] * w7p3_7));
    x1[12] = (x[0] + (x[3] * w7p3_7) + (x[6] * w7m1_7) + (x[9] * w7p2_7) +
              (x[12] * w7p5_7) + (x[15] * w7p1_7) + (x[18] * w7p4_7));
    x1[13] = (x[1] + (x[4] * w7p3_7) + (x[7] * w7m1_7) + (x[10] * w7p2_7) +
              (x[13] * w7p5_7) + (x[16] * w7p1_7) + (x[19] * w7p4_7));
    x1[14] = (x[2] + (x[5] * w7p3_7) + (x[8] * w7m1_7) + (x[11] * w7p2_7) +
              (x[14] * w7p5_7) + (x[17] * w7p1_7) + (x[20] * w7p4_7));
    x1[15] = (x[0] + (x[3] * w7p2_7) + (x[6] * w7p4_7) + (x[9] * w7m1_7) +
              (x[12] * w7p1_7) + (x[15] * w7p3_7) + (x[18] * w7p5_7));
    x1[16] = (x[1] + (x[4] * w7p2_7) + (x[7] * w7p4_7) + (x[10] * w7m1_7) +
              (x[13] * w7p1_7) + (x[16] * w7p3_7) + (x[19] * w7p5_7));
    x1[17] = (x[2] + (x[5] * w7p2_7) + (x[8] * w7p4_7) + (x[11] * w7m1_7) +
              (x[14] * w7p1_7) + (x[17] * w7p3_7) + (x[20] * w7p5_7));
    x1[18] = (x[0] + (x[3] * w7p1_7) + (x[6] * w7p2_7) + (x[9] * w7p3_7) +
              (x[12] * w7p4_7) + (x[15] * w7p5_7) + (x[18] * w7m1_7));
    x1[19] = (x[1] + (x[4] * w7p1_7) + (x[7] * w7p2_7) + (x[10] * w7p3_7) +
              (x[13] * w7p4_7) + (x[16] * w7p5_7) + (x[19] * w7m1_7));
    x1[20] = (x[2] + (x[5] * w7p1_7) + (x[8] * w7p2_7) + (x[11] * w7p3_7) +
              (x[14] * w7p4_7) + (x[17] * w7p5_7) + (x[20] * w7m1_7));
    // multiply with twiddle factors and transpose;
    {
      static alignas(64) float complex x2[21];
      const float complex w21m1_21 =
          ((9.555728057861407e-1) + (-2.947551744109042e-1i));
      const float complex w21p19_21 =
          ((8.262387743159949e-1) + (-5.63320058063622e-1i));
      const float complex w21p17_21 =
          ((3.65341024366395e-1) + (-9.308737486442042e-1i));
      const float complex w21p6_7 =
          ((6.234898018587335e-1) + (-7.818314824680297e-1i));
      const float complex w21p5_7 =
          ((-2.2252093395631434e-1) + (-9.749279121818235e-1i));
      const float complex w21p13_21 =
          ((-7.330518718298263e-1) + (-6.801727377709194e-1i));
      const float complex w21p16_21 =
          ((7.473009358642417e-2) + (-9.9720379718118e-1i));
      const float complex w21p11_21 =
          ((-9.888308262251285e-1) + (-1.4904226617617428e-1i));
      const float complex w21p3_7 =
          ((-9.009688679024191e-1) + (4.33883739117558e-1i));
      x2[0] = x1[0];
      x2[7] = x1[1];
      x2[14] = x1[2];
      x2[1] = x1[3];
      x2[8] = (x1[4] * w21m1_21);
      x2[15] = (x1[5] * w21p19_21);
      x2[2] = x1[6];
      x2[9] = (x1[7] * w21p19_21);
      x2[16] = (x1[8] * w21p17_21);
      x2[3] = x1[9];
      x2[10] = (x1[10] * w21p6_7);
      x2[17] = (x1[11] * w21p5_7);
      x2[4] = x1[12];
      x2[11] = (x1[13] * w21p17_21);
      x2[18] = (x1[14] * w21p13_21);
      x2[5] = x1[15];
      x2[12] = (x1[16] * w21p16_21);
      x2[19] = (x1[17] * w21p11_21);
      x2[6] = x1[18];
      x2[13] = (x1[19] * w21p5_7);
      x2[20] = (x1[20] * w21p3_7);
      // another dft;
      {
        static alignas(64) float complex x3[21];
        const float complex w3m1_3 =
            ((-4.999999999999997e-1) + (-8.660254037844386e-1i));
        const float complex w3p1_3 =
            ((-5.000000000000004e-1) + (8.660254037844384e-1i));
        x3[0] = (x2[0] + x2[7] + x2[14]);
        x3[7] = (x2[0] + (x2[7] * w3m1_3) + (x2[14] * w3p1_3));
        x3[14] = (x2[0] + (x2[7] * w3p1_3) + (x2[14] * w3m1_3));
        x3[1] = (x2[1] + x2[8] + x2[15]);
        x3[8] = (x2[1] + (x2[8] * w3m1_3) + (x2[15] * w3p1_3));
        x3[15] = (x2[1] + (x2[8] * w3p1_3) + (x2[15] * w3m1_3));
        x3[2] = (x2[2] + x2[9] + x2[16]);
        x3[9] = (x2[2] + (x2[9] * w3m1_3) + (x2[16] * w3p1_3));
        x3[16] = (x2[2] + (x2[9] * w3p1_3) + (x2[16] * w3m1_3));
        x3[3] = (x2[3] + x2[10] + x2[17]);
        x3[10] = (x2[3] + (x2[10] * w3m1_3) + (x2[17] * w3p1_3));
        x3[17] = (x2[3] + (x2[10] * w3p1_3) + (x2[17] * w3m1_3));
        x3[4] = (x2[4] + x2[11] + x2[18]);
        x3[11] = (x2[4] + (x2[11] * w3m1_3) + (x2[18] * w3p1_3));
        x3[18] = (x2[4] + (x2[11] * w3p1_3) + (x2[18] * w3m1_3));
        x3[5] = (x2[5] + x2[12] + x2[19]);
        x3[12] = (x2[5] + (x2[12] * w3m1_3) + (x2[19] * w3p1_3));
        x3[19] = (x2[5] + (x2[12] * w3p1_3) + (x2[19] * w3m1_3));
        x3[6] = (x2[6] + x2[13] + x2[20]);
        x3[13] = (x2[6] + (x2[13] * w3m1_3) + (x2[20] * w3p1_3));
        x3[20] = (x2[6] + (x2[13] * w3p1_3) + (x2[20] * w3m1_3));
        return x3;
      }
    }
  }
}
int main() {
  simd_driver();
  {
    alignas(64) float complex a_in[21];
    float complex *a_out;
    float complex *a_out_slow;
    memset(a_in, 0, (21 * sizeof(complex float)));
    for (int i = 0; (i < 21); i += 1) {
      a_in[i] = sinf(((-7.183775341024617e+0) * i));
    }
    a_out = fft_21_3_7(a_in);
    for (int j = 0; (j < 100000); j += 1) {
      a_out_slow = dft_21(a_in);
    }
    printf("idx     fft                    dft\n");
    for (int i = 0; (i < 21); i += 1) {
      printf("%02d   %6.3f+(%6.3f)i       %6.3f+(%6.3f)i\n", i,
             crealf(a_out[i]), cimagf(a_out[i]), crealf(a_out_slow[i]),
             cimagf(a_out_slow[i]));
    }
  }
  return 0;
}