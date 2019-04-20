// gcc -std=c99 -Ofast -flto -ffast-math -march=skylake -msse2  -ftree-vectorize
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
float complex *fun_slow(float complex *__restrict__ a) {
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
float complex *fun(float complex *__restrict__ x) {
  x = __builtin_assume_aligned(x, 64);
  // dft on each row;
  {
    static float complex s[(4 * 4)] = {0.0fi};
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
      static float complex z[(4 * 4)] = {0.0fi};
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
      {
        static float complex y[(4 * 4)] = {0.0fi};
        y[0] = (z[0] + z[4] + z[8] + z[12]);
        y[1] = (z[1] + z[5] + z[9] + z[13]);
        y[2] = (z[2] + z[6] + z[10] + z[14]);
        y[3] = (z[3] + z[7] + z[11] + z[15]);
        y[4] = (z[0] + (CMPLXF(cimagf(z[4]), (-1 * crealf(z[4])))) +
                (-1 * z[8]) + (CMPLXF((-1 * cimagf(z[12])), crealf(z[12]))));
        y[5] = (z[1] + (CMPLXF(cimagf(z[5]), (-1 * crealf(z[5])))) +
                (-1 * z[9]) + (CMPLXF((-1 * cimagf(z[13])), crealf(z[13]))));
        y[6] = (z[2] + (CMPLXF(cimagf(z[6]), (-1 * crealf(z[6])))) +
                (-1 * z[10]) + (CMPLXF((-1 * cimagf(z[14])), crealf(z[14]))));
        y[7] = (z[3] + (CMPLXF(cimagf(z[7]), (-1 * crealf(z[7])))) +
                (-1 * z[11]) + (CMPLXF((-1 * cimagf(z[15])), crealf(z[15]))));
        y[8] = (z[0] + (-1 * z[4]) + z[8] + (-1 * z[12]));
        y[9] = (z[1] + (-1 * z[5]) + z[9] + (-1 * z[13]));
        y[10] = (z[2] + (-1 * z[6]) + z[10] + (-1 * z[14]));
        y[11] = (z[3] + (-1 * z[7]) + z[11] + (-1 * z[15]));
        y[12] = (z[0] + (CMPLXF((-1 * cimagf(z[4])), crealf(z[4]))) +
                 (-1 * z[8]) + (CMPLXF(cimagf(z[12]), (-1 * crealf(z[12])))));
        y[13] = (z[1] + (CMPLXF((-1 * cimagf(z[5])), crealf(z[5]))) +
                 (-1 * z[9]) + (CMPLXF(cimagf(z[13]), (-1 * crealf(z[13])))));
        y[14] = (z[2] + (CMPLXF((-1 * cimagf(z[6])), crealf(z[6]))) +
                 (-1 * z[10]) + (CMPLXF(cimagf(z[14]), (-1 * crealf(z[14])))));
        y[15] = (z[3] + (CMPLXF((-1 * cimagf(z[7])), crealf(z[7]))) +
                 (-1 * z[11]) + (CMPLXF(cimagf(z[15]), (-1 * crealf(z[15])))));
        return y;
      }
    }
  }
}
alignas(64) float complex global_a[(4 * 4)] = {0.0fi};

int main() {
  global_a[0] = (1.e+0);
  global_a[1] = (1.e+0);
  global_a[2] = (1.e+0);
  global_a[3] = (1.e+0);
  global_a[4] = (1.e+0);
  global_a[5] = (1.e+0);
  global_a[6] = (1.e+0);
  global_a[7] = (1.e+0);
  global_a[8] = (1.e+0);
  global_a[9] = (1.e+0);
  global_a[10] = (1.e+0);
  global_a[11] = (1.e+0);
  global_a[12] = (1.e+0);
  global_a[13] = (1.e+0);
  global_a[14] = (1.e+0);
  global_a[15] = (1.e+0);
  {
    complex float *k_slow = fun_slow(global_a);
    float complex *k_fast = fun(global_a);
    printf("idx     global_a          k_slow           k_fast f=0\n");
    for (int i = 0; (i < 16); i += 1) {
      {

        printf("%02d   %6.3f+(%6.3f)i %6.3f+(%6.3f)i %6.3f+(%6.3f)i \n", i,
               crealf(global_a[i]), cimagf(global_a[i]), crealf(k_slow[i]),
               cimagf(k_slow[i]), crealf(k_fast[i]), cimagf(k_fast[i]));
      }
    }
  }
  global_a[0] = (1.e+0);
  global_a[1] = (7.071067811865475e-1);
  global_a[2] = (6.123233995736767e-17);
  global_a[3] = (-7.071067811865475e-1);
  global_a[4] = (-1.e+0);
  global_a[5] = (-7.071067811865476e-1);
  global_a[6] = (-1.8369701987210296e-16);
  global_a[7] = (7.071067811865474e-1);
  global_a[8] = (1.e+0);
  global_a[9] = (7.071067811865476e-1);
  global_a[10] = (3.061616997868383e-16);
  global_a[11] = (-7.071067811865467e-1);
  global_a[12] = (-1.e+0);
  global_a[13] = (-7.07106781186547e-1);
  global_a[14] = (-4.2862637970157363e-16);
  global_a[15] = (7.071067811865465e-1);
  {
    complex float *k_slow = fun_slow(global_a);
    float complex *k_fast = fun(global_a);
    printf("idx     global_a          k_slow           k_fast f=2\n");
    for (int i = 0; (i < 16); i += 1) {
      {

        printf("%02d   %6.3f+(%6.3f)i %6.3f+(%6.3f)i %6.3f+(%6.3f)i \n", i,
               crealf(global_a[i]), cimagf(global_a[i]), crealf(k_slow[i]),
               cimagf(k_slow[i]), crealf(k_fast[i]), cimagf(k_fast[i]));
      }
    }
  }
  global_a[0] = (1.e+0);
  global_a[1] = (5.515475995016242e-1);
  global_a[2] = (-3.915904909679914e-1);
  global_a[3] = (-9.835091900637407e-1);
  global_a[4] = (-6.933137747668947e-1);
  global_a[5] = (2.18718093715559e-1);
  global_a[6] = (9.34580653879671e-1);
  global_a[7] = (8.122133386604224e-1);
  global_a[8] = (-3.8632019436958975e-2);
  global_a[9] = (-8.54828133829132e-1);
  global_a[10] = (-9.043247909628639e-1);
  global_a[11] = (-1.4272820142161857e-1);
  global_a[12] = (7.46881997212307e-1);
  global_a[13] = (9.666101467684727e-1);
  global_a[14] = (3.1938101499582e-1);
  global_a[15] = (-6.143024824737984e-1);
  {
    complex float *k_slow = fun_slow(global_a);
    float complex *k_fast = fun(global_a);
    printf("idx     global_a          k_slow           k_fast f=2.5123\n");
    for (int i = 0; (i < 16); i += 1) {
      {

        printf("%02d   %6.3f+(%6.3f)i %6.3f+(%6.3f)i %6.3f+(%6.3f)i \n", i,
               crealf(global_a[i]), cimagf(global_a[i]), crealf(k_slow[i]),
               cimagf(k_slow[i]), crealf(k_fast[i]), cimagf(k_fast[i]));
      }
    }
  }
  return 0;
}