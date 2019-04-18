// gcc -std=c99 -Ofast -flto -ffast-math -march=skylake -msse2  -ftree-vectorize
// icc -std=c99 -O2 -D NOFUNCCALL -qopt-report=1 -qopt-report-phase=vec
// -guide-vec
#include <complex.h>
#include <stdio.h>
float complex *fun(float complex *__restrict__ a) {
  {
    static float complex x[(4 * 4)] = {0.0fi};
    // split 1d into col major n1 x n2 matrix, n1 columns, n2 rows;
    x[(0 + (4 * 0))] = a[0];
    x[(0 + (4 * 1))] = a[1];
    x[(0 + (4 * 2))] = a[2];
    x[(0 + (4 * 3))] = a[3];
    x[(1 + (4 * 0))] = a[4];
    x[(1 + (4 * 1))] = a[5];
    x[(1 + (4 * 2))] = a[6];
    x[(1 + (4 * 3))] = a[7];
    x[(2 + (4 * 0))] = a[8];
    x[(2 + (4 * 1))] = a[9];
    x[(2 + (4 * 2))] = a[10];
    x[(2 + (4 * 3))] = a[11];
    x[(3 + (4 * 0))] = a[12];
    x[(3 + (4 * 1))] = a[13];
    x[(3 + (4 * 2))] = a[14];
    x[(3 + (4 * 3))] = a[15];
    // dft on each row;
    {
      static float complex s[(4 * 4)] = {0.0fi};
      const float complex wn2_1_4 = ((1.e+0) + (0.0e+0i));
      const float complex wn2_0_1 = ((0.0e+0) + (-1.e+0i));
      const float complex wn2_3_4 = ((-1.e+0) + (0.0e+0i));
      const float complex wn2_1_2 = ((0.0e+0) + (1.e+0i));
      s[0] = (x[0] + x[4] + x[8] + x[12]);
      s[1] = (x[1] + x[5] + x[9] + x[13]);
      s[2] = (x[2] + x[6] + x[10] + x[14]);
      s[3] = (x[3] + x[7] + x[11] + x[15]);
      s[4] = (x[0] + (x[4] * wn2_0_1) + (x[8] * wn2_3_4) + (x[12] * wn2_1_2));
      s[5] = (x[1] + (x[5] * wn2_0_1) + (x[9] * wn2_3_4) + (x[13] * wn2_1_2));
      s[6] = (x[2] + (x[6] * wn2_0_1) + (x[10] * wn2_3_4) + (x[14] * wn2_1_2));
      s[7] = (x[3] + (x[7] * wn2_0_1) + (x[11] * wn2_3_4) + (x[15] * wn2_1_2));
      s[8] = (x[0] + (x[4] * wn2_3_4) + (x[8] * wn2_1_4) + (x[12] * wn2_3_4));
      s[9] = (x[1] + (x[5] * wn2_3_4) + (x[9] * wn2_1_4) + (x[13] * wn2_3_4));
      s[10] = (x[2] + (x[6] * wn2_3_4) + (x[10] * wn2_1_4) + (x[14] * wn2_3_4));
      s[11] = (x[3] + (x[7] * wn2_3_4) + (x[11] * wn2_1_4) + (x[15] * wn2_3_4));
      s[12] = (x[0] + (x[4] * wn2_1_2) + (x[8] * wn2_3_4) + (x[12] * wn2_0_1));
      s[13] = (x[1] + (x[5] * wn2_1_2) + (x[9] * wn2_3_4) + (x[13] * wn2_0_1));
      s[14] = (x[2] + (x[6] * wn2_1_2) + (x[10] * wn2_3_4) + (x[14] * wn2_0_1));
      s[15] = (x[3] + (x[7] * wn2_1_2) + (x[11] * wn2_3_4) + (x[15] * wn2_0_1));
      // transpose and elementwise multiplication;
      {
        static float complex z[(4 * 4)] = {0.0fi};
        const float complex wn_m22500 =
            ((9.238795325112866e-1) + (-3.826834323650897e-1i));
        const float complex wn_m45000 =
            ((7.071067811865475e-1) + (-7.071067811865475e-1i));
        const float complex wn_m67500 =
            ((3.826834323650898e-1) + (-9.238795325112866e-1i));
        const float complex wn_m90000 = ((0.0e+0) + (-1.e+0i));
        const float complex wn_m135000 =
            ((-7.071067811865475e-1) + (-7.071067811865475e-1i));
        const float complex wn_p157500 =
            ((-9.238795325112867e-1) + (3.8268343236508967e-1i));
        z[0] = s[0];
        z[1] = s[4];
        z[2] = s[8];
        z[3] = s[12];
        z[4] = s[1];
        z[5] = (s[5] * wn_m22500);
        z[6] = (s[9] * wn_m45000);
        z[7] = (s[13] * wn_m67500);
        z[8] = s[2];
        z[9] = (s[6] * wn_m45000);
        z[10] = (s[10] * wn_m90000);
        z[11] = (s[14] * wn_m135000);
        z[12] = s[3];
        z[13] = (s[7] * wn_m67500);
        z[14] = (s[11] * wn_m135000);
        z[15] = (s[15] * wn_p157500);
        // dft on each row;
        {
          static float complex y[(4 * 4)] = {0.0fi};
          const float complex wn1_m90000 = ((0.0e+0) + (-1.e+0i));
          const float complex wn1_m180000 = ((-1.e+0) + (0.0e+0i));
          const float complex wn1_p90000 = ((0.0e+0) + (1.e+0i));
          const float complex wn1_p0 = ((1.e+0) + (0.0e+0i));
          y[0] = (z[0] + z[4] + z[8] + z[12]);
          y[1] = (z[1] + z[5] + z[9] + z[13]);
          y[2] = (z[2] + z[6] + z[10] + z[14]);
          y[3] = (z[3] + z[7] + z[11] + z[15]);
          y[4] = (z[0] + (wn1_m90000 * z[4]) + (wn1_m180000 * z[8]) +
                  (wn1_p90000 * z[12]));
          y[5] = (z[1] + (wn1_m90000 * z[5]) + (wn1_m180000 * z[9]) +
                  (wn1_p90000 * z[13]));
          y[6] = (z[2] + (wn1_m90000 * z[6]) + (wn1_m180000 * z[10]) +
                  (wn1_p90000 * z[14]));
          y[7] = (z[3] + (wn1_m90000 * z[7]) + (wn1_m180000 * z[11]) +
                  (wn1_p90000 * z[15]));
          y[8] = (z[0] + (wn1_m180000 * z[4]) + (wn1_p0 * z[8]) +
                  (wn1_m180000 * z[12]));
          y[9] = (z[1] + (wn1_m180000 * z[5]) + (wn1_p0 * z[9]) +
                  (wn1_m180000 * z[13]));
          y[10] = (z[2] + (wn1_m180000 * z[6]) + (wn1_p0 * z[10]) +
                   (wn1_m180000 * z[14]));
          y[11] = (z[3] + (wn1_m180000 * z[7]) + (wn1_p0 * z[11]) +
                   (wn1_m180000 * z[15]));
          y[12] = (z[0] + (wn1_p90000 * z[4]) + (wn1_m180000 * z[8]) +
                   (wn1_m90000 * z[12]));
          y[13] = (z[1] + (wn1_p90000 * z[5]) + (wn1_m180000 * z[9]) +
                   (wn1_m90000 * z[13]));
          y[14] = (z[2] + (wn1_p90000 * z[6]) + (wn1_m180000 * z[10]) +
                   (wn1_m90000 * z[14]));
          y[15] = (z[3] + (wn1_p90000 * z[7]) + (wn1_m180000 * z[11]) +
                   (wn1_m90000 * z[15]));
          return y;
        }
      }
    }
  }
}
float complex global_a[(4 * 4)];

int main() {
  fun(global_a);
  return 0;
}