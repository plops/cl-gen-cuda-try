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
extern void simd_16_fft_112_16_7(v16sf *__restrict__ re_in,
                                 v16sf *__restrict__ im_in,
                                 v16sf *__restrict__ re_out,
                                 v16sf *__restrict__ im_out) {
  re_in = __builtin_assume_aligned(re_in, 64);
  im_in = __builtin_assume_aligned(im_in, 64);
  re_out = __builtin_assume_aligned(re_out, 64);
  im_out = __builtin_assume_aligned(im_out, 64);
  {
    static alignas(64) v16sf x1_re[7];
    static alignas(64) v16sf x1_im[7];
    const alignas(64) v16sf con = {(0.0e+0f), (1.e+0f),  (2.e+0f),  (3.e+0f),
                                   (4.e+0f),  (5.e+0f),  (6.e+0f),  (7.e+0f),
                                   (8.e+0f),  (9.e+0f),  (1.e+1f),  (1.1e+1f),
                                   (1.2e+1f), (1.3e+1f), (1.4e+1f), (1.5e+1f)};
    x1_re[0] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                (con * re_in[96]));
    x1_re[1] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                (con * re_in[97]));
    x1_re[2] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                (con * re_in[98]));
    x1_re[3] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                (con * re_in[99]));
    x1_re[4] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                (con * re_in[100]));
    x1_re[5] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                (con * re_in[101]));
    x1_re[6] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                (con * re_in[102]));
    x1_re[7] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                (con * re_in[103]));
    x1_re[8] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                (con * re_in[104]));
    x1_re[9] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                (con * re_in[105]));
    x1_re[10] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                 (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                 (con * re_in[106]));
    x1_re[11] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                 (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                 (con * re_in[107]));
    x1_re[12] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                 (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                 (con * re_in[108]));
    x1_re[13] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                 (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                 (con * re_in[109]));
    x1_re[14] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                 (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                 (con * re_in[110]));
    x1_re[15] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                 (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                 (con * re_in[111]));
    x1_re[16] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[17] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                 (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                 (con * re_in[97]));
    x1_re[18] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                 (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                 (con * re_in[98]));
    x1_re[19] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                 (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                 (con * re_in[99]));
    x1_re[20] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                 (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                 (con * re_in[100]));
    x1_re[21] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                 (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                 (con * re_in[101]));
    x1_re[22] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                 (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                 (con * re_in[102]));
    x1_re[23] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                 (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                 (con * re_in[103]));
    x1_re[24] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                 (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                 (con * re_in[104]));
    x1_re[25] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                 (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                 (con * re_in[105]));
    x1_re[26] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                 (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                 (con * re_in[106]));
    x1_re[27] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                 (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                 (con * re_in[107]));
    x1_re[28] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                 (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                 (con * re_in[108]));
    x1_re[29] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                 (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                 (con * re_in[109]));
    x1_re[30] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                 (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                 (con * re_in[110]));
    x1_re[31] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                 (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                 (con * re_in[111]));
    x1_re[32] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[33] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                 (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                 (con * re_in[97]));
    x1_re[34] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                 (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                 (con * re_in[98]));
    x1_re[35] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                 (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                 (con * re_in[99]));
    x1_re[36] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                 (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                 (con * re_in[100]));
    x1_re[37] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                 (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                 (con * re_in[101]));
    x1_re[38] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                 (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                 (con * re_in[102]));
    x1_re[39] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                 (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                 (con * re_in[103]));
    x1_re[40] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                 (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                 (con * re_in[104]));
    x1_re[41] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                 (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                 (con * re_in[105]));
    x1_re[42] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                 (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                 (con * re_in[106]));
    x1_re[43] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                 (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                 (con * re_in[107]));
    x1_re[44] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                 (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                 (con * re_in[108]));
    x1_re[45] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                 (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                 (con * re_in[109]));
    x1_re[46] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                 (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                 (con * re_in[110]));
    x1_re[47] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                 (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                 (con * re_in[111]));
    x1_re[48] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[49] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                 (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                 (con * re_in[97]));
    x1_re[50] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                 (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                 (con * re_in[98]));
    x1_re[51] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                 (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                 (con * re_in[99]));
    x1_re[52] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                 (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                 (con * re_in[100]));
    x1_re[53] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                 (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                 (con * re_in[101]));
    x1_re[54] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                 (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                 (con * re_in[102]));
    x1_re[55] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                 (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                 (con * re_in[103]));
    x1_re[56] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                 (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                 (con * re_in[104]));
    x1_re[57] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                 (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                 (con * re_in[105]));
    x1_re[58] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                 (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                 (con * re_in[106]));
    x1_re[59] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                 (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                 (con * re_in[107]));
    x1_re[60] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                 (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                 (con * re_in[108]));
    x1_re[61] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                 (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                 (con * re_in[109]));
    x1_re[62] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                 (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                 (con * re_in[110]));
    x1_re[63] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                 (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                 (con * re_in[111]));
    x1_re[64] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[65] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                 (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                 (con * re_in[97]));
    x1_re[66] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                 (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                 (con * re_in[98]));
    x1_re[67] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                 (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                 (con * re_in[99]));
    x1_re[68] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                 (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                 (con * re_in[100]));
    x1_re[69] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                 (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                 (con * re_in[101]));
    x1_re[70] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                 (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                 (con * re_in[102]));
    x1_re[71] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                 (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                 (con * re_in[103]));
    x1_re[72] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                 (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                 (con * re_in[104]));
    x1_re[73] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                 (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                 (con * re_in[105]));
    x1_re[74] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                 (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                 (con * re_in[106]));
    x1_re[75] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                 (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                 (con * re_in[107]));
    x1_re[76] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                 (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                 (con * re_in[108]));
    x1_re[77] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                 (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                 (con * re_in[109]));
    x1_re[78] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                 (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                 (con * re_in[110]));
    x1_re[79] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                 (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                 (con * re_in[111]));
    x1_re[80] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[81] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                 (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                 (con * re_in[97]));
    x1_re[82] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                 (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                 (con * re_in[98]));
    x1_re[83] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                 (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                 (con * re_in[99]));
    x1_re[84] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                 (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                 (con * re_in[100]));
    x1_re[85] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                 (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                 (con * re_in[101]));
    x1_re[86] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                 (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                 (con * re_in[102]));
    x1_re[87] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                 (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                 (con * re_in[103]));
    x1_re[88] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                 (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                 (con * re_in[104]));
    x1_re[89] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                 (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                 (con * re_in[105]));
    x1_re[90] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                 (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                 (con * re_in[106]));
    x1_re[91] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                 (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                 (con * re_in[107]));
    x1_re[92] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                 (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                 (con * re_in[108]));
    x1_re[93] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                 (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                 (con * re_in[109]));
    x1_re[94] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                 (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                 (con * re_in[110]));
    x1_re[95] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                 (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                 (con * re_in[111]));
    x1_re[96] = ((con * re_in[0]) + (con * re_in[16]) + (con * re_in[32]) +
                 (con * re_in[48]) + (con * re_in[64]) + (con * re_in[80]) +
                 (con * re_in[96]));
    x1_re[97] = ((con * re_in[1]) + (con * re_in[17]) + (con * re_in[33]) +
                 (con * re_in[49]) + (con * re_in[65]) + (con * re_in[81]) +
                 (con * re_in[97]));
    x1_re[98] = ((con * re_in[2]) + (con * re_in[18]) + (con * re_in[34]) +
                 (con * re_in[50]) + (con * re_in[66]) + (con * re_in[82]) +
                 (con * re_in[98]));
    x1_re[99] = ((con * re_in[3]) + (con * re_in[19]) + (con * re_in[35]) +
                 (con * re_in[51]) + (con * re_in[67]) + (con * re_in[83]) +
                 (con * re_in[99]));
    x1_re[100] = ((con * re_in[4]) + (con * re_in[20]) + (con * re_in[36]) +
                  (con * re_in[52]) + (con * re_in[68]) + (con * re_in[84]) +
                  (con * re_in[100]));
    x1_re[101] = ((con * re_in[5]) + (con * re_in[21]) + (con * re_in[37]) +
                  (con * re_in[53]) + (con * re_in[69]) + (con * re_in[85]) +
                  (con * re_in[101]));
    x1_re[102] = ((con * re_in[6]) + (con * re_in[22]) + (con * re_in[38]) +
                  (con * re_in[54]) + (con * re_in[70]) + (con * re_in[86]) +
                  (con * re_in[102]));
    x1_re[103] = ((con * re_in[7]) + (con * re_in[23]) + (con * re_in[39]) +
                  (con * re_in[55]) + (con * re_in[71]) + (con * re_in[87]) +
                  (con * re_in[103]));
    x1_re[104] = ((con * re_in[8]) + (con * re_in[24]) + (con * re_in[40]) +
                  (con * re_in[56]) + (con * re_in[72]) + (con * re_in[88]) +
                  (con * re_in[104]));
    x1_re[105] = ((con * re_in[9]) + (con * re_in[25]) + (con * re_in[41]) +
                  (con * re_in[57]) + (con * re_in[73]) + (con * re_in[89]) +
                  (con * re_in[105]));
    x1_re[106] = ((con * re_in[10]) + (con * re_in[26]) + (con * re_in[42]) +
                  (con * re_in[58]) + (con * re_in[74]) + (con * re_in[90]) +
                  (con * re_in[106]));
    x1_re[107] = ((con * re_in[11]) + (con * re_in[27]) + (con * re_in[43]) +
                  (con * re_in[59]) + (con * re_in[75]) + (con * re_in[91]) +
                  (con * re_in[107]));
    x1_re[108] = ((con * re_in[12]) + (con * re_in[28]) + (con * re_in[44]) +
                  (con * re_in[60]) + (con * re_in[76]) + (con * re_in[92]) +
                  (con * re_in[108]));
    x1_re[109] = ((con * re_in[13]) + (con * re_in[29]) + (con * re_in[45]) +
                  (con * re_in[61]) + (con * re_in[77]) + (con * re_in[93]) +
                  (con * re_in[109]));
    x1_re[110] = ((con * re_in[14]) + (con * re_in[30]) + (con * re_in[46]) +
                  (con * re_in[62]) + (con * re_in[78]) + (con * re_in[94]) +
                  (con * re_in[110]));
    x1_re[111] = ((con * re_in[15]) + (con * re_in[31]) + (con * re_in[47]) +
                  (con * re_in[63]) + (con * re_in[79]) + (con * re_in[95]) +
                  (con * re_in[111]));
    memcpy(x1_re, re_out, sizeof(x1_re));
  }
}
void simd_driver() {
  {
    static v16sf in_re[7];
    static v16sf in_im[7];
    static v16sf out_re[7];
    static v16sf out_im[7];
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