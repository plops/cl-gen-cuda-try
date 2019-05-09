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
    const float w0_0_1 /* 0.0 */ = 0x0.0p23;
    const float w14933127_m26_1 /* 0.22252093 */ = 0x1.C7B90Ep-3;
    const float w14558722_m25_1 /* 0.43388373 */ = 0x1.BC4C04p-2;
    const float w10460423_m24_1 /* 0.6234898 */ = 0x1.3F3A0Ep-1;
    const float w13116956_m24_1 /* 0.7818315 */ = 0x1.904C38p-1;
    const float w15115749_m24_1 /* 0.90096885 */ = 0x1.CD4BCAp-1;
    const float w16356576_m24_1 /* 0.9749279 */ = 0x1.F329C0p-1;
    const float w8388608_m23_1 /* 1.0 */ = 0x1.0p0;
    {
      const alignas(64) vsf coef_re = {
          (1 * w8388608_m23_1), (1 * w8388608_m23_1), (1 * w8388608_m23_1),
          (1 * w8388608_m23_1), (1 * w8388608_m23_1), (1 * w8388608_m23_1),
          (1 * w8388608_m23_1), (1 * w8388608_m23_1), (1 * w8388608_m23_1),
          (1 * w8388608_m23_1), (1 * w8388608_m23_1), (1 * w8388608_m23_1),
          (1 * w8388608_m23_1), (1 * w8388608_m23_1), (1 * w8388608_m23_1),
          (1 * w8388608_m23_1)};
      const alignas(64) vsf coef_im = {
          w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1,
          w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1, w0_0_1};
      x1_re[0] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                  ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                  ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                  ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                  ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                  ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                  ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[0] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                  ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                  ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                  ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                  ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                  ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                  ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
    {
      const alignas(64)
          vsf coef_re = {(1 * w8388608_m23_1),   (1 * w10460423_m24_1),
                         (-1 * w14933127_m26_1), (-1 * w15115749_m24_1),
                         (-1 * w15115749_m24_1), (-1 * w14933127_m26_1),
                         (1 * w10460423_m24_1),  (1 * w8388608_m23_1),
                         (1 * w10460423_m24_1),  (-1 * w14933127_m26_1),
                         (-1 * w15115749_m24_1), (-1 * w15115749_m24_1),
                         (-1 * w14933127_m26_1), (1 * w10460423_m24_1),
                         (1 * w8388608_m23_1),   (1 * w10460423_m24_1)};
      const alignas(64) vsf coef_im = {w0_0_1,
                                       (-1 * w13116956_m24_1),
                                       (-1 * w16356576_m24_1),
                                       (-1 * w14558722_m25_1),
                                       w14558722_m25_1,
                                       w16356576_m24_1,
                                       w13116956_m24_1,
                                       w0_0_1,
                                       (-1 * w13116956_m24_1),
                                       (-1 * w16356576_m24_1),
                                       (-1 * w14558722_m25_1),
                                       w14558722_m25_1,
                                       w16356576_m24_1,
                                       w13116956_m24_1,
                                       w0_0_1,
                                       (-1 * w13116956_m24_1)};
      x1_re[16] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                   ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                   ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                   ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                   ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                   ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                   ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[16] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                   ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                   ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                   ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                   ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                   ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                   ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
    {
      const alignas(64)
          vsf coef_re = {(1 * w8388608_m23_1),   (-1 * w14933127_m26_1),
                         (-1 * w15115749_m24_1), (1 * w10460423_m24_1),
                         (1 * w10460423_m24_1),  (-1 * w15115749_m24_1),
                         (-1 * w14933127_m26_1), (1 * w8388608_m23_1),
                         (-1 * w14933127_m26_1), (-1 * w15115749_m24_1),
                         (1 * w10460423_m24_1),  (1 * w10460423_m24_1),
                         (-1 * w15115749_m24_1), (-1 * w14933127_m26_1),
                         (1 * w8388608_m23_1),   (-1 * w14933127_m26_1)};
      const alignas(64) vsf coef_im = {w0_0_1,
                                       (-1 * w16356576_m24_1),
                                       w14558722_m25_1,
                                       w13116956_m24_1,
                                       (-1 * w13116956_m24_1),
                                       (-1 * w14558722_m25_1),
                                       w16356576_m24_1,
                                       w0_0_1,
                                       (-1 * w16356576_m24_1),
                                       w14558722_m25_1,
                                       w13116956_m24_1,
                                       (-1 * w13116956_m24_1),
                                       (-1 * w14558722_m25_1),
                                       w16356576_m24_1,
                                       w0_0_1,
                                       (-1 * w16356576_m24_1)};
      x1_re[32] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                   ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                   ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                   ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                   ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                   ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                   ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[32] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                   ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                   ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                   ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                   ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                   ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                   ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
    {
      const alignas(64)
          vsf coef_re = {(1 * w8388608_m23_1),   (-1 * w15115749_m24_1),
                         (1 * w10460423_m24_1),  (-1 * w14933127_m26_1),
                         (-1 * w14933127_m26_1), (1 * w10460423_m24_1),
                         (-1 * w15115749_m24_1), (1 * w8388608_m23_1),
                         (-1 * w15115749_m24_1), (1 * w10460423_m24_1),
                         (-1 * w14933127_m26_1), (-1 * w14933127_m26_1),
                         (1 * w10460423_m24_1),  (-1 * w15115749_m24_1),
                         (1 * w8388608_m23_1),   (-1 * w15115749_m24_1)};
      const alignas(64) vsf coef_im = {w0_0_1,
                                       (-1 * w14558722_m25_1),
                                       w13116956_m24_1,
                                       (-1 * w16356576_m24_1),
                                       w16356576_m24_1,
                                       (-1 * w13116956_m24_1),
                                       w14558722_m25_1,
                                       w0_0_1,
                                       (-1 * w14558722_m25_1),
                                       w13116956_m24_1,
                                       (-1 * w16356576_m24_1),
                                       w16356576_m24_1,
                                       (-1 * w13116956_m24_1),
                                       w14558722_m25_1,
                                       w13879754_m73_1,
                                       (-1 * w14558722_m25_1)};
      x1_re[48] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                   ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                   ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                   ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                   ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                   ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                   ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[48] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                   ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                   ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                   ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                   ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                   ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                   ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
    {
      const alignas(64)
          vsf coef_re = {(1 * w8388608_m23_1),   (-1 * w15115749_m24_1),
                         (1 * w10460423_m24_1),  (-1 * w14933127_m26_1),
                         (-1 * w14933127_m26_1), (1 * w10460423_m24_1),
                         (-1 * w15115749_m24_1), (1 * w8388608_m23_1),
                         (-1 * w15115749_m24_1), (1 * w10460423_m24_1),
                         (-1 * w14933127_m26_1), (-1 * w14933127_m26_1),
                         (1 * w10460423_m24_1),  (-1 * w15115749_m24_1),
                         (1 * w8388608_m23_1),   (-1 * w15115749_m24_1)};
      const alignas(64) vsf coef_im = {w0_0_1,
                                       w14558722_m25_1,
                                       (-1 * w13116956_m24_1),
                                       w16356576_m24_1,
                                       (-1 * w16356576_m24_1),
                                       w13116956_m24_1,
                                       (-1 * w14558722_m25_1),
                                       w0_0_1,
                                       w14558722_m25_1,
                                       (-1 * w13116956_m24_1),
                                       w16356576_m24_1,
                                       (-1 * w16356576_m24_1),
                                       w13116956_m24_1,
                                       (-1 * w14558722_m25_1),
                                       w9253170_m72_1,
                                       w14558722_m25_1};
      x1_re[64] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                   ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                   ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                   ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                   ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                   ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                   ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[64] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                   ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                   ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                   ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                   ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                   ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                   ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
    {
      const alignas(64)
          vsf coef_re = {(1 * w8388608_m23_1),   (-1 * w14933127_m26_1),
                         (-1 * w15115749_m24_1), (1 * w10460423_m24_1),
                         (1 * w10460423_m24_1),  (-1 * w15115749_m24_1),
                         (-1 * w14933127_m26_1), (1 * w8388608_m23_1),
                         (-1 * w14933127_m26_1), (-1 * w15115749_m24_1),
                         (1 * w10460423_m24_1),  (1 * w10460423_m24_1),
                         (-1 * w15115749_m24_1), (-1 * w14933127_m26_1),
                         (1 * w8388608_m23_1),   (-1 * w14933127_m26_1)};
      const alignas(64) vsf coef_im = {w0_0_1,
                                       w16356576_m24_1,
                                       (-1 * w14558722_m25_1),
                                       (-1 * w13116956_m24_1),
                                       w13116956_m24_1,
                                       w14558722_m25_1,
                                       (-1 * w16356576_m24_1),
                                       w11566462_m73_1,
                                       w16356576_m24_1,
                                       (-1 * w14558722_m25_1),
                                       (-1 * w13116956_m24_1),
                                       w13116956_m24_1,
                                       w14558722_m25_1,
                                       (-1 * w16356576_m24_1),
                                       w11566462_m72_1,
                                       w16356576_m24_1};
      x1_re[80] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                   ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                   ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                   ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                   ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                   ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                   ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[80] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                   ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                   ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                   ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                   ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                   ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                   ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
    {
      const alignas(64)
          vsf coef_re = {(1 * w8388608_m23_1),   (1 * w10460423_m24_1),
                         (-1 * w14933127_m26_1), (-1 * w15115749_m24_1),
                         (-1 * w15115749_m24_1), (-1 * w14933127_m26_1),
                         (1 * w10460423_m24_1),  (1 * w8388608_m23_1),
                         (1 * w10460423_m24_1),  (-1 * w14933127_m26_1),
                         (-1 * w15115749_m24_1), (-1 * w15115749_m24_1),
                         (-1 * w14933127_m26_1), (1 * w10460423_m24_1),
                         (1 * w8388608_m23_1),   (1 * w10460423_m24_1)};
      const alignas(64) vsf coef_im = {w0_0_1,
                                       w13116956_m24_1,
                                       w16356576_m24_1,
                                       w14558722_m25_1,
                                       (-1 * w14558722_m25_1),
                                       (-1 * w16356576_m24_1),
                                       (-1 * w13116956_m24_1),
                                       w13879754_m73_1,
                                       w13116956_m24_1,
                                       w16356576_m24_1,
                                       w14558722_m25_1,
                                       (-1 * w14558722_m25_1),
                                       (-1 * w16356576_m24_1),
                                       (-1 * w13116956_m24_1),
                                       w13879754_m72_1,
                                       w13116956_m24_1};
      x1_re[96] = (((coef_re * re_in[0]) - (coef_im * im_in[0])) +
                   ((coef_re * re_in[16]) - (coef_im * im_in[16])) +
                   ((coef_re * re_in[32]) - (coef_im * im_in[32])) +
                   ((coef_re * re_in[48]) - (coef_im * im_in[48])) +
                   ((coef_re * re_in[64]) - (coef_im * im_in[64])) +
                   ((coef_re * re_in[80]) - (coef_im * im_in[80])) +
                   ((coef_re * re_in[96]) - (coef_im * im_in[96])));
      x1_im[96] = (((coef_im * re_in[0]) + (coef_re * im_in[0])) +
                   ((coef_im * re_in[16]) + (coef_re * im_in[16])) +
                   ((coef_im * re_in[32]) + (coef_re * im_in[32])) +
                   ((coef_im * re_in[48]) + (coef_re * im_in[48])) +
                   ((coef_im * re_in[64]) + (coef_re * im_in[64])) +
                   ((coef_im * re_in[80]) + (coef_re * im_in[80])) +
                   ((coef_im * re_in[96]) + (coef_re * im_in[96])));
    }
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
  {
    float a = 0x1.99999Ap-4;
    simd_driver();
  }
  return 0;
}