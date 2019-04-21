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
    for (unsigned int j = 0; (j < 16); j += 1) {
      for (unsigned int k = 0; (k < 16); k += 1) {
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
    for (unsigned int j = 0; (j < 256); j += 1) {
      for (unsigned int k = 0; (k < 256); k += 1) {
        y[j] =
            (y[j] + (a[k] * cexpf((1.0fi * (-2.454369260617026e-2) * j * k))));
      }
    }
    return y;
  }
}
float complex *fft16_radix4(float complex *__restrict__ x,
                            float complex *__restrict__ out_y) {
  x = __builtin_assume_aligned(x, 64);
  // dft on each row;
  {
    static alignas(64) float complex s[(4 * 4)] = {0.0fi};
    s[0] = (x[0] + x[4] + x[8] + x[12]);
    s[1] = (x[1] + x[5] + x[9] + x[13]);
    s[2] = (x[2] + x[6] + x[10] + x[14]);
    s[3] = (x[3] + x[7] + x[11] + x[15]);
    s[4] = (x[0] + (CMPLXF(cimagf(x[4]), (-1 * crealf(x[4])))) + (-1 * x[8]) +
            (CMPLXF((-1 * cimagf(x[12])), crealf(x[12]))));
    s[5] = (x[1] + (CMPLXF(cimagf(x[5]), (-1 * crealf(x[5])))) + (-1 * x[9]) +
            (CMPLXF((-1 * cimagf(x[13])), crealf(x[13]))));
    s[6] = (x[2] + (CMPLXF(cimagf(x[6]), (-1 * crealf(x[6])))) + (-1 * x[10]) +
            (CMPLXF((-1 * cimagf(x[14])), crealf(x[14]))));
    s[7] = (x[3] + (CMPLXF(cimagf(x[7]), (-1 * crealf(x[7])))) + (-1 * x[11]) +
            (CMPLXF((-1 * cimagf(x[15])), crealf(x[15]))));
    s[8] = (x[0] + (-1 * x[4]) + x[8] + (-1 * x[12]));
    s[9] = (x[1] + (-1 * x[5]) + x[9] + (-1 * x[13]));
    s[10] = (x[2] + (-1 * x[6]) + x[10] + (-1 * x[14]));
    s[11] = (x[3] + (-1 * x[7]) + x[11] + (-1 * x[15]));
    s[12] = (x[0] + (CMPLXF((-1 * cimagf(x[4])), crealf(x[4]))) + (-1 * x[8]) +
             (CMPLXF(cimagf(x[12]), (-1 * crealf(x[12])))));
    s[13] = (x[1] + (CMPLXF((-1 * cimagf(x[5])), crealf(x[5]))) + (-1 * x[9]) +
             (CMPLXF(cimagf(x[13]), (-1 * crealf(x[13])))));
    s[14] = (x[2] + (CMPLXF((-1 * cimagf(x[6])), crealf(x[6]))) + (-1 * x[10]) +
             (CMPLXF(cimagf(x[14]), (-1 * crealf(x[14])))));
    s[15] = (x[3] + (CMPLXF((-1 * cimagf(x[7])), crealf(x[7]))) + (-1 * x[11]) +
             (CMPLXF(cimagf(x[15]), (-1 * crealf(x[15])))));
    // transpose and elementwise multiplication;
    // Twiddle factors are named by their angle in the unit turn turn
    // https://en.wikipedia.org/wiki/Turn_(geometry). Storing it as a rational
    // number doesn't loose precision.;
    {
      static alignas(64) float complex z[(4 * 4)] = {0.0fi};
      const float complex w16m1_16 =
          ((9.238795325112866e-1) + (-3.826834323650897e-1i));
      const float complex w16p7_8 =
          ((7.071067811865475e-1) + (-7.071067811865475e-1i));
      const float complex w16p13_16 =
          ((3.826834323650898e-1) + (-9.238795325112866e-1i));
      const float complex w16p5_8 =
          ((-7.071067811865475e-1) + (-7.071067811865475e-1i));
      const float complex w16p7_16 =
          ((-9.238795325112867e-1) + (3.8268343236508967e-1i));
      z[0] = s[0];
      z[1] = s[4];
      z[2] = s[8];
      z[3] = s[12];
      z[4] = s[1];
      z[5] = (s[5] * w16m1_16);
      z[6] = (s[9] * w16p7_8);
      z[7] = (s[13] * w16p13_16);
      z[8] = s[2];
      z[9] = (s[6] * w16p7_8);
      z[10] = (CMPLXF(cimagf(s[10]), (-1 * crealf(s[10]))));
      z[11] = (s[14] * w16p5_8);
      z[12] = s[3];
      z[13] = (s[7] * w16p13_16);
      z[14] = (s[11] * w16p5_8);
      z[15] = (s[15] * w16p7_16);
      // dft on each row;
      out_y = __builtin_assume_aligned(out_y, 64);
      {

        out_y[0] = (z[0] + z[4] + z[8] + z[12]);
        out_y[1] = (z[1] + z[5] + z[9] + z[13]);
        out_y[2] = (z[2] + z[6] + z[10] + z[14]);
        out_y[3] = (z[3] + z[7] + z[11] + z[15]);
        out_y[4] =
            (z[0] + (CMPLXF(cimagf(z[4]), (-1 * crealf(z[4])))) + (-1 * z[8]) +
             (CMPLXF((-1 * cimagf(z[12])), crealf(z[12]))));
        out_y[5] =
            (z[1] + (CMPLXF(cimagf(z[5]), (-1 * crealf(z[5])))) + (-1 * z[9]) +
             (CMPLXF((-1 * cimagf(z[13])), crealf(z[13]))));
        out_y[6] =
            (z[2] + (CMPLXF(cimagf(z[6]), (-1 * crealf(z[6])))) + (-1 * z[10]) +
             (CMPLXF((-1 * cimagf(z[14])), crealf(z[14]))));
        out_y[7] =
            (z[3] + (CMPLXF(cimagf(z[7]), (-1 * crealf(z[7])))) + (-1 * z[11]) +
             (CMPLXF((-1 * cimagf(z[15])), crealf(z[15]))));
        out_y[8] = (z[0] + (-1 * z[4]) + z[8] + (-1 * z[12]));
        out_y[9] = (z[1] + (-1 * z[5]) + z[9] + (-1 * z[13]));
        out_y[10] = (z[2] + (-1 * z[6]) + z[10] + (-1 * z[14]));
        out_y[11] = (z[3] + (-1 * z[7]) + z[11] + (-1 * z[15]));
        out_y[12] =
            (z[0] + (CMPLXF((-1 * cimagf(z[4])), crealf(z[4]))) + (-1 * z[8]) +
             (CMPLXF(cimagf(z[12]), (-1 * crealf(z[12])))));
        out_y[13] =
            (z[1] + (CMPLXF((-1 * cimagf(z[5])), crealf(z[5]))) + (-1 * z[9]) +
             (CMPLXF(cimagf(z[13]), (-1 * crealf(z[13])))));
        out_y[14] =
            (z[2] + (CMPLXF((-1 * cimagf(z[6])), crealf(z[6]))) + (-1 * z[10]) +
             (CMPLXF(cimagf(z[14]), (-1 * crealf(z[14])))));
        out_y[15] =
            (z[3] + (CMPLXF((-1 * cimagf(z[7])), crealf(z[7]))) + (-1 * z[11]) +
             (CMPLXF(cimagf(z[15]), (-1 * crealf(z[15])))));
        return out_y;
      }
    }
  }
}
float complex *fft256(float complex *__restrict__ x) {
  x = __builtin_assume_aligned(x, 64);
  // fft16 on each row;
  {
    static alignas(64) float complex s[(16 * 16)] = {0.0fi};
    fft16_radix4((&(x[0])), (&(s[0])));
    fft16_radix4((&(x[16])), (&(s[16])));
    fft16_radix4((&(x[32])), (&(s[32])));
    fft16_radix4((&(x[48])), (&(s[48])));
    fft16_radix4((&(x[64])), (&(s[64])));
    fft16_radix4((&(x[80])), (&(s[80])));
    fft16_radix4((&(x[96])), (&(s[96])));
    fft16_radix4((&(x[112])), (&(s[112])));
    fft16_radix4((&(x[128])), (&(s[128])));
    fft16_radix4((&(x[144])), (&(s[144])));
    fft16_radix4((&(x[160])), (&(s[160])));
    fft16_radix4((&(x[176])), (&(s[176])));
    fft16_radix4((&(x[192])), (&(s[192])));
    fft16_radix4((&(x[208])), (&(s[208])));
    fft16_radix4((&(x[224])), (&(s[224])));
    fft16_radix4((&(x[240])), (&(s[240])));
    // transpose and elementwise multiplication;
    ;
    {
      static alignas(64) float complex z[(16 * 16)] = {0.0fi};
      z[0] = s[0];
      z[16] = s[1];
      z[32] = s[2];
      z[48] = s[3];
      z[64] = s[4];
      z[80] = s[5];
      z[96] = s[6];
      z[112] = s[7];
      z[128] = s[8];
      z[144] = s[9];
      z[160] = s[10];
      z[176] = s[11];
      z[192] = s[12];
      z[208] = s[13];
      z[224] = s[14];
      z[240] = s[15];
      z[1] = s[16];
      z[17] = s[17];
      z[33] = s[18];
      z[49] = s[19];
      z[65] = s[20];
      z[81] = s[21];
      z[97] = s[22];
      z[113] = s[23];
      z[129] = s[24];
      z[145] = s[25];
      z[161] = s[26];
      z[177] = s[27];
      z[193] = s[28];
      z[209] = s[29];
      z[225] = s[30];
      z[241] = s[31];
      z[2] = s[32];
      z[18] = s[33];
      z[34] = s[34];
      z[50] = s[35];
      z[66] = s[36];
      z[82] = s[37];
      z[98] = s[38];
      z[114] = s[39];
      z[130] = s[40];
      z[146] = s[41];
      z[162] = s[42];
      z[178] = s[43];
      z[194] = s[44];
      z[210] = s[45];
      z[226] = s[46];
      z[242] = s[47];
      z[3] = s[48];
      z[19] = s[49];
      z[35] = s[50];
      z[51] = s[51];
      z[67] = s[52];
      z[83] = s[53];
      z[99] = s[54];
      z[115] = s[55];
      z[131] = s[56];
      z[147] = s[57];
      z[163] = s[58];
      z[179] = s[59];
      z[195] = s[60];
      z[211] = s[61];
      z[227] = s[62];
      z[243] = s[63];
      z[4] = s[64];
      z[20] = s[65];
      z[36] = s[66];
      z[52] = s[67];
      z[68] = s[68];
      z[84] = s[69];
      z[100] = s[70];
      z[116] = s[71];
      z[132] = s[72];
      z[148] = s[73];
      z[164] = s[74];
      z[180] = s[75];
      z[196] = s[76];
      z[212] = s[77];
      z[228] = s[78];
      z[244] = s[79];
      z[5] = s[80];
      z[21] = s[81];
      z[37] = s[82];
      z[53] = s[83];
      z[69] = s[84];
      z[85] = s[85];
      z[101] = s[86];
      z[117] = s[87];
      z[133] = s[88];
      z[149] = s[89];
      z[165] = s[90];
      z[181] = s[91];
      z[197] = s[92];
      z[213] = s[93];
      z[229] = s[94];
      z[245] = s[95];
      z[6] = s[96];
      z[22] = s[97];
      z[38] = s[98];
      z[54] = s[99];
      z[70] = s[100];
      z[86] = s[101];
      z[102] = s[102];
      z[118] = s[103];
      z[134] = s[104];
      z[150] = s[105];
      z[166] = s[106];
      z[182] = s[107];
      z[198] = s[108];
      z[214] = s[109];
      z[230] = s[110];
      z[246] = s[111];
      z[7] = s[112];
      z[23] = s[113];
      z[39] = s[114];
      z[55] = s[115];
      z[71] = s[116];
      z[87] = s[117];
      z[103] = s[118];
      z[119] = s[119];
      z[135] = s[120];
      z[151] = s[121];
      z[167] = s[122];
      z[183] = s[123];
      z[199] = s[124];
      z[215] = s[125];
      z[231] = s[126];
      z[247] = s[127];
      z[8] = s[128];
      z[24] = s[129];
      z[40] = s[130];
      z[56] = s[131];
      z[72] = s[132];
      z[88] = s[133];
      z[104] = s[134];
      z[120] = s[135];
      z[136] = s[136];
      z[152] = s[137];
      z[168] = s[138];
      z[184] = s[139];
      z[200] = s[140];
      z[216] = s[141];
      z[232] = s[142];
      z[248] = s[143];
      z[9] = s[144];
      z[25] = s[145];
      z[41] = s[146];
      z[57] = s[147];
      z[73] = s[148];
      z[89] = s[149];
      z[105] = s[150];
      z[121] = s[151];
      z[137] = s[152];
      z[153] = s[153];
      z[169] = s[154];
      z[185] = s[155];
      z[201] = s[156];
      z[217] = s[157];
      z[233] = s[158];
      z[249] = s[159];
      z[10] = s[160];
      z[26] = s[161];
      z[42] = s[162];
      z[58] = s[163];
      z[74] = s[164];
      z[90] = s[165];
      z[106] = s[166];
      z[122] = s[167];
      z[138] = s[168];
      z[154] = s[169];
      z[170] = s[170];
      z[186] = s[171];
      z[202] = s[172];
      z[218] = s[173];
      z[234] = s[174];
      z[250] = s[175];
      z[11] = s[176];
      z[27] = s[177];
      z[43] = s[178];
      z[59] = s[179];
      z[75] = s[180];
      z[91] = s[181];
      z[107] = s[182];
      z[123] = s[183];
      z[139] = s[184];
      z[155] = s[185];
      z[171] = s[186];
      z[187] = s[187];
      z[203] = s[188];
      z[219] = s[189];
      z[235] = s[190];
      z[251] = s[191];
      z[12] = s[192];
      z[28] = s[193];
      z[44] = s[194];
      z[60] = s[195];
      z[76] = s[196];
      z[92] = s[197];
      z[108] = s[198];
      z[124] = s[199];
      z[140] = s[200];
      z[156] = s[201];
      z[172] = s[202];
      z[188] = s[203];
      z[204] = s[204];
      z[220] = s[205];
      z[236] = s[206];
      z[252] = s[207];
      z[13] = s[208];
      z[29] = s[209];
      z[45] = s[210];
      z[61] = s[211];
      z[77] = s[212];
      z[93] = s[213];
      z[109] = s[214];
      z[125] = s[215];
      z[141] = s[216];
      z[157] = s[217];
      z[173] = s[218];
      z[189] = s[219];
      z[205] = s[220];
      z[221] = s[221];
      z[237] = s[222];
      z[253] = s[223];
      z[14] = s[224];
      z[30] = s[225];
      z[46] = s[226];
      z[62] = s[227];
      z[78] = s[228];
      z[94] = s[229];
      z[110] = s[230];
      z[126] = s[231];
      z[142] = s[232];
      z[158] = s[233];
      z[174] = s[234];
      z[190] = s[235];
      z[206] = s[236];
      z[222] = s[237];
      z[238] = s[238];
      z[254] = s[239];
      z[15] = s[240];
      z[31] = s[241];
      z[47] = s[242];
      z[63] = s[243];
      z[79] = s[244];
      z[95] = s[245];
      z[111] = s[246];
      z[127] = s[247];
      z[143] = s[248];
      z[159] = s[249];
      z[175] = s[250];
      z[191] = s[251];
      z[207] = s[252];
      z[223] = s[253];
      z[239] = s[254];
      z[255] = s[255];
      // fft16 on each row;
      ;
      {
        static alignas(64) float complex y[(16 * 16)] = {0.0fi};
        fft16_radix4((&(z[0])), (&(y[0])));
        fft16_radix4((&(z[16])), (&(y[16])));
        fft16_radix4((&(z[32])), (&(y[32])));
        fft16_radix4((&(z[48])), (&(y[48])));
        fft16_radix4((&(z[64])), (&(y[64])));
        fft16_radix4((&(z[80])), (&(y[80])));
        fft16_radix4((&(z[96])), (&(y[96])));
        fft16_radix4((&(z[112])), (&(y[112])));
        fft16_radix4((&(z[128])), (&(y[128])));
        fft16_radix4((&(z[144])), (&(y[144])));
        fft16_radix4((&(z[160])), (&(y[160])));
        fft16_radix4((&(z[176])), (&(y[176])));
        fft16_radix4((&(z[192])), (&(y[192])));
        fft16_radix4((&(z[208])), (&(y[208])));
        fft16_radix4((&(z[224])), (&(y[224])));
        fft16_radix4((&(z[240])), (&(y[240])));
        return y;
      }
    }
  }
}
int main() {
  {
    alignas(64) float complex a_in[256];
    float complex *a_out;
    float complex *a_out_slow;
    memset(a_in, 0, (256 * sizeof(complex float)));
    for (unsigned int i = 0; (i < 256); i += 1) {
      a_in[i] = sinf(((-1.1780972450961724e+0) * i));
    }
    a_out = fft256(a_in);
    a_out_slow = dft256_slow(a_in);
    printf("idx     fft256               dft256_slow\n");
    for (unsigned int i = 0; (i < 256); i += 1) {
      printf("%02d   %6.3f+(%6.3f)i       %6.3f+(%6.3f)i\n", i,
             crealf(a_out[i]), cimagf(a_out[i]), crealf(a_out_slow[i]),
             cimagf(a_out_slow[i]));
    }
  }
  return 0;
}