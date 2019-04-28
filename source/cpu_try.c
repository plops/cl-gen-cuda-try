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
typedef float v16sf __attribute__((vector_size(64)));
;
void simd_16_fft_112_16_7(v16sf *__restrict__ re_in, v16sf *__restrict__ im_in,
                          v16sf *__restrict__ re_out,
                          v16sf *__restrict__ im_out) {
  re_in = __builtin_assume_aligned(re_in, 64);
  im_in = __builtin_assume_aligned(im_in, 64);
  re_out = __builtin_assume_aligned(re_out, 64);
  im_out = __builtin_assume_aligned(im_out, 64);
  {
    static alignas(64) v16sf x1_re[7];
    static alignas(64) v16sf x1_im[7];
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
    x1_re[0] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                re_in[80] + re_in[96]);
    x1_re[1] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                re_in[81] + re_in[97]);
    x1_re[2] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                re_in[82] + re_in[98]);
    x1_re[3] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                re_in[83] + re_in[99]);
    x1_re[4] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                re_in[84] + re_in[100]);
    x1_re[5] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                re_in[85] + re_in[101]);
    x1_re[6] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                re_in[86] + re_in[102]);
    x1_re[7] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                re_in[87] + re_in[103]);
    x1_re[8] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                re_in[88] + re_in[104]);
    x1_re[9] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                re_in[89] + re_in[105]);
    x1_re[10] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                 re_in[90] + re_in[106]);
    x1_re[11] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                 re_in[91] + re_in[107]);
    x1_re[12] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                 re_in[92] + re_in[108]);
    x1_re[13] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                 re_in[93] + re_in[109]);
    x1_re[14] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                 re_in[94] + re_in[110]);
    x1_re[15] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                 re_in[95] + re_in[111]);
    x1_re[16] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                 re_in[80] + re_in[96]);
    x1_re[17] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                 re_in[81] + re_in[97]);
    x1_re[18] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                 re_in[82] + re_in[98]);
    x1_re[19] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                 re_in[83] + re_in[99]);
    x1_re[20] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                 re_in[84] + re_in[100]);
    x1_re[21] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                 re_in[85] + re_in[101]);
    x1_re[22] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                 re_in[86] + re_in[102]);
    x1_re[23] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                 re_in[87] + re_in[103]);
    x1_re[24] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                 re_in[88] + re_in[104]);
    x1_re[25] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                 re_in[89] + re_in[105]);
    x1_re[26] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                 re_in[90] + re_in[106]);
    x1_re[27] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                 re_in[91] + re_in[107]);
    x1_re[28] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                 re_in[92] + re_in[108]);
    x1_re[29] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                 re_in[93] + re_in[109]);
    x1_re[30] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                 re_in[94] + re_in[110]);
    x1_re[31] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                 re_in[95] + re_in[111]);
    x1_re[32] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                 re_in[80] + re_in[96]);
    x1_re[33] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                 re_in[81] + re_in[97]);
    x1_re[34] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                 re_in[82] + re_in[98]);
    x1_re[35] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                 re_in[83] + re_in[99]);
    x1_re[36] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                 re_in[84] + re_in[100]);
    x1_re[37] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                 re_in[85] + re_in[101]);
    x1_re[38] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                 re_in[86] + re_in[102]);
    x1_re[39] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                 re_in[87] + re_in[103]);
    x1_re[40] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                 re_in[88] + re_in[104]);
    x1_re[41] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                 re_in[89] + re_in[105]);
    x1_re[42] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                 re_in[90] + re_in[106]);
    x1_re[43] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                 re_in[91] + re_in[107]);
    x1_re[44] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                 re_in[92] + re_in[108]);
    x1_re[45] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                 re_in[93] + re_in[109]);
    x1_re[46] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                 re_in[94] + re_in[110]);
    x1_re[47] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                 re_in[95] + re_in[111]);
    x1_re[48] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                 re_in[80] + re_in[96]);
    x1_re[49] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                 re_in[81] + re_in[97]);
    x1_re[50] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                 re_in[82] + re_in[98]);
    x1_re[51] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                 re_in[83] + re_in[99]);
    x1_re[52] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                 re_in[84] + re_in[100]);
    x1_re[53] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                 re_in[85] + re_in[101]);
    x1_re[54] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                 re_in[86] + re_in[102]);
    x1_re[55] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                 re_in[87] + re_in[103]);
    x1_re[56] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                 re_in[88] + re_in[104]);
    x1_re[57] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                 re_in[89] + re_in[105]);
    x1_re[58] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                 re_in[90] + re_in[106]);
    x1_re[59] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                 re_in[91] + re_in[107]);
    x1_re[60] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                 re_in[92] + re_in[108]);
    x1_re[61] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                 re_in[93] + re_in[109]);
    x1_re[62] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                 re_in[94] + re_in[110]);
    x1_re[63] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                 re_in[95] + re_in[111]);
    x1_re[64] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                 re_in[80] + re_in[96]);
    x1_re[65] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                 re_in[81] + re_in[97]);
    x1_re[66] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                 re_in[82] + re_in[98]);
    x1_re[67] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                 re_in[83] + re_in[99]);
    x1_re[68] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                 re_in[84] + re_in[100]);
    x1_re[69] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                 re_in[85] + re_in[101]);
    x1_re[70] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                 re_in[86] + re_in[102]);
    x1_re[71] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                 re_in[87] + re_in[103]);
    x1_re[72] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                 re_in[88] + re_in[104]);
    x1_re[73] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                 re_in[89] + re_in[105]);
    x1_re[74] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                 re_in[90] + re_in[106]);
    x1_re[75] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                 re_in[91] + re_in[107]);
    x1_re[76] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                 re_in[92] + re_in[108]);
    x1_re[77] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                 re_in[93] + re_in[109]);
    x1_re[78] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                 re_in[94] + re_in[110]);
    x1_re[79] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                 re_in[95] + re_in[111]);
    x1_re[80] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                 re_in[80] + re_in[96]);
    x1_re[81] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                 re_in[81] + re_in[97]);
    x1_re[82] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                 re_in[82] + re_in[98]);
    x1_re[83] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                 re_in[83] + re_in[99]);
    x1_re[84] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                 re_in[84] + re_in[100]);
    x1_re[85] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                 re_in[85] + re_in[101]);
    x1_re[86] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                 re_in[86] + re_in[102]);
    x1_re[87] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                 re_in[87] + re_in[103]);
    x1_re[88] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                 re_in[88] + re_in[104]);
    x1_re[89] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                 re_in[89] + re_in[105]);
    x1_re[90] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                 re_in[90] + re_in[106]);
    x1_re[91] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                 re_in[91] + re_in[107]);
    x1_re[92] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                 re_in[92] + re_in[108]);
    x1_re[93] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                 re_in[93] + re_in[109]);
    x1_re[94] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                 re_in[94] + re_in[110]);
    x1_re[95] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                 re_in[95] + re_in[111]);
    x1_re[96] = (re_in[0] + re_in[16] + re_in[32] + re_in[48] + re_in[64] +
                 re_in[80] + re_in[96]);
    x1_re[97] = (re_in[1] + re_in[17] + re_in[33] + re_in[49] + re_in[65] +
                 re_in[81] + re_in[97]);
    x1_re[98] = (re_in[2] + re_in[18] + re_in[34] + re_in[50] + re_in[66] +
                 re_in[82] + re_in[98]);
    x1_re[99] = (re_in[3] + re_in[19] + re_in[35] + re_in[51] + re_in[67] +
                 re_in[83] + re_in[99]);
    x1_re[100] = (re_in[4] + re_in[20] + re_in[36] + re_in[52] + re_in[68] +
                  re_in[84] + re_in[100]);
    x1_re[101] = (re_in[5] + re_in[21] + re_in[37] + re_in[53] + re_in[69] +
                  re_in[85] + re_in[101]);
    x1_re[102] = (re_in[6] + re_in[22] + re_in[38] + re_in[54] + re_in[70] +
                  re_in[86] + re_in[102]);
    x1_re[103] = (re_in[7] + re_in[23] + re_in[39] + re_in[55] + re_in[71] +
                  re_in[87] + re_in[103]);
    x1_re[104] = (re_in[8] + re_in[24] + re_in[40] + re_in[56] + re_in[72] +
                  re_in[88] + re_in[104]);
    x1_re[105] = (re_in[9] + re_in[25] + re_in[41] + re_in[57] + re_in[73] +
                  re_in[89] + re_in[105]);
    x1_re[106] = (re_in[10] + re_in[26] + re_in[42] + re_in[58] + re_in[74] +
                  re_in[90] + re_in[106]);
    x1_re[107] = (re_in[11] + re_in[27] + re_in[43] + re_in[59] + re_in[75] +
                  re_in[91] + re_in[107]);
    x1_re[108] = (re_in[12] + re_in[28] + re_in[44] + re_in[60] + re_in[76] +
                  re_in[92] + re_in[108]);
    x1_re[109] = (re_in[13] + re_in[29] + re_in[45] + re_in[61] + re_in[77] +
                  re_in[93] + re_in[109]);
    x1_re[110] = (re_in[14] + re_in[30] + re_in[46] + re_in[62] + re_in[78] +
                  re_in[94] + re_in[110]);
    x1_re[111] = (re_in[15] + re_in[31] + re_in[47] + re_in[63] + re_in[79] +
                  re_in[95] + re_in[111]);
  }
}
float complex *dft_21(float complex *__restrict__ a) {
  a = __builtin_assume_aligned(a, 64);
  {
    static alignas(64) float complex y[21] = {0.0fi};
    memset(y, 0, (21 * sizeof(complex float)));
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
  {
    alignas(64) float complex a_in[21];
    float complex *a_out;
    float complex *a_out_slow;
    memset(a_in, 0, (21 * sizeof(complex float)));
    memset(a_out, 0, (21 * sizeof(complex float)));
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