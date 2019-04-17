#include <complex.h>
#include <stdio.h>
void fun(complex *__restrict__ a, int n) {
  {
    static complex local_a[(8 * 8)];
    local_a[0] = a[0];
    local_a[4] = a[1];
    local_a[2] = a[2];
    local_a[6] = a[3];
    local_a[1] = a[4];
    local_a[5] = a[5];
    local_a[3] = a[6];
    local_a[7] = a[7];
    local_a[8] = a[8];
    local_a[12] = a[9];
    local_a[10] = a[10];
    local_a[14] = a[11];
    local_a[9] = a[12];
    local_a[13] = a[13];
    local_a[11] = a[14];
    local_a[15] = a[15];
    local_a[16] = a[16];
    local_a[20] = a[17];
    local_a[18] = a[18];
    local_a[22] = a[19];
    local_a[17] = a[20];
    local_a[21] = a[21];
    local_a[19] = a[22];
    local_a[23] = a[23];
    local_a[24] = a[24];
    local_a[28] = a[25];
    local_a[26] = a[26];
    local_a[30] = a[27];
    local_a[25] = a[28];
    local_a[29] = a[29];
    local_a[27] = a[30];
    local_a[31] = a[31];
    local_a[32] = a[32];
    local_a[36] = a[33];
    local_a[34] = a[34];
    local_a[38] = a[35];
    local_a[33] = a[36];
    local_a[37] = a[37];
    local_a[35] = a[38];
    local_a[39] = a[39];
    local_a[40] = a[40];
    local_a[44] = a[41];
    local_a[42] = a[42];
    local_a[46] = a[43];
    local_a[41] = a[44];
    local_a[45] = a[45];
    local_a[43] = a[46];
    local_a[47] = a[47];
    local_a[48] = a[48];
    local_a[52] = a[49];
    local_a[50] = a[50];
    local_a[54] = a[51];
    local_a[49] = a[52];
    local_a[53] = a[53];
    local_a[51] = a[54];
    local_a[55] = a[55];
    local_a[56] = a[56];
    local_a[60] = a[57];
    local_a[58] = a[58];
    local_a[62] = a[59];
    local_a[57] = a[60];
    local_a[61] = a[61];
    local_a[59] = a[62];
    local_a[63] = a[63];
    {
      static complex line[8];
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 0)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 0)] = (line[k] * cexpf((k * (0.0e+0) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 8)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 8)] =
            (line[k] * cexpf((k * (7.853981633974483e-1) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 16)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 16)] =
            (line[k] * cexpf((k * (1.5707963267948966e+0) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 24)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 24)] =
            (line[k] * cexpf((k * (2.3561944901923448e+0) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 32)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 32)] =
            (line[k] * cexpf((k * (3.141592653589793e+0) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 40)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 40)] =
            (line[k] * cexpf((k * (3.9269908169872414e+0) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 48)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 48)] =
            (line[k] * cexpf((k * (4.7123889803846897e+0) * 1.0fi)));
      }
      for (unsigned int i = 0; (i < 8); i += 1) {
        line[i] = local_a[(i + 56)];
      }
      for (unsigned int k = 0; (k < 8); k += 1) {
        local_a[(k + 56)] =
            (line[k] * cexpf((k * (5.497787143782138e+0) * 1.0fi)));
      }
    }
  }
}
complex global_a[256];

int main() {
  fun(global_a, 256);
  return 0;
}