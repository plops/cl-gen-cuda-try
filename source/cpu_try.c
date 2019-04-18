#include <complex.h>
#include <stdio.h>
void fun(complex *__restrict__ a) {
  {
    static complex x[(4 * 4)] = {0.0fi};
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
      static complex s[(4 * 4)] = {0.0fi};
      const complex w_1_4 = ((1.e+0) + (0.0e+0i));
      const complex w_0_1 = ((0.0e+0) + (-1.e+0i));
      const complex w_3_4 = ((-1.e+0) + (0.0e+0i));
      const complex w_1_2 = ((0.0e+0) + (1.e+0i));
      s[(0 + (4 * 0))] = (x[0] + x[4] + x[8] + x[12]);
      s[(1 + (4 * 0))] = (x[1] + x[5] + x[9] + x[13]);
      s[(2 + (4 * 0))] = (x[2] + x[6] + x[10] + x[14]);
      s[(3 + (4 * 0))] = (x[3] + x[7] + x[11] + x[15]);
      s[(0 + (4 * 1))] =
          (x[0] + (x[4] * w_0_1) + (x[8] * w_3_4) + (x[12] * w_1_2));
      s[(1 + (4 * 1))] =
          (x[1] + (x[5] * w_0_1) + (x[9] * w_3_4) + (x[13] * w_1_2));
      s[(2 + (4 * 1))] =
          (x[2] + (x[6] * w_0_1) + (x[10] * w_3_4) + (x[14] * w_1_2));
      s[(3 + (4 * 1))] =
          (x[3] + (x[7] * w_0_1) + (x[11] * w_3_4) + (x[15] * w_1_2));
      s[(0 + (4 * 2))] =
          (x[0] + (x[4] * w_3_4) + (x[8] * w_1_4) + (x[12] * w_3_4));
      s[(1 + (4 * 2))] =
          (x[1] + (x[5] * w_3_4) + (x[9] * w_1_4) + (x[13] * w_3_4));
      s[(2 + (4 * 2))] =
          (x[2] + (x[6] * w_3_4) + (x[10] * w_1_4) + (x[14] * w_3_4));
      s[(3 + (4 * 2))] =
          (x[3] + (x[7] * w_3_4) + (x[11] * w_1_4) + (x[15] * w_3_4));
      s[(0 + (4 * 3))] =
          (x[0] + (x[4] * w_1_2) + (x[8] * w_3_4) + (x[12] * w_0_1));
      s[(1 + (4 * 3))] =
          (x[1] + (x[5] * w_1_2) + (x[9] * w_3_4) + (x[13] * w_0_1));
      s[(2 + (4 * 3))] =
          (x[2] + (x[6] * w_1_2) + (x[10] * w_3_4) + (x[14] * w_0_1));
      s[(3 + (4 * 3))] =
          (x[3] + (x[7] * w_1_2) + (x[11] * w_3_4) + (x[15] * w_0_1));
    }
  }
}
complex global_a[(4 * 4)];

int main() {
  fun(global_a);
  return 0;
}