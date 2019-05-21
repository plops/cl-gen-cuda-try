import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import pycuda.compiler
import numpy as np
mod=pycuda.compiler.SourceModule("""
#include <cuComplex.h>
#define cimagf(x) cuCimagf(x)
#define crealf(x) cuCrealf(x)
#define CMPLXF(x, y) make_cuFloatComplex((x), (y))
typedef cuFloatComplex complex;
// https://en.wikipedia.org/wiki/Cooley%E2%80%93Tukey_FFT_algorithm;
__global__ void fft_21_3_7(complex *dst, complex *src) {
  // n1 DFTs of size n2 in the column direction;
  {
    const int i = threadIdx.x;
    complex *x = (src + (21 * i));
    complex x1[21];
    const complex w7m1_7 =
        CMPLXF((6.234898018587335e-1), (-7.818314824680297e-1));
    const complex w7p5_7 =
        CMPLXF((-2.2252093395631434e-1), (-9.749279121818235e-1));
    const complex w7p4_7 =
        CMPLXF((-9.009688679024189e-1), (-4.3388373911755823e-1));
    const complex w7p3_7 =
        CMPLXF((-9.009688679024191e-1), (4.33883739117558e-1));
    const complex w7p2_7 =
        CMPLXF((-2.2252093395631461e-1), (9.749279121818235e-1));
    const complex w7p1_7 =
        CMPLXF((6.234898018587334e-1), (7.818314824680298e-1));
    x1[0] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)), cuCaddf(aref, cuCaddf(x, 3))),
        cuCaddf(
            cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), cuCaddf(aref, cuCaddf(x, 9))),
            cuCaddf(cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                            cuCaddf(aref, cuCaddf(x, 15))),
                    cuCaddf(aref, cuCaddf(x, 18)))));
    x1[1] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)), cuCaddf(aref, cuCaddf(x, 4))),
        cuCaddf(cuCaddf(cuCaddf(aref, cuCaddf(x, 7)),
                        cuCaddf(aref, cuCaddf(x, 10))),
                cuCaddf(cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                cuCaddf(aref, cuCaddf(x, 16))),
                        cuCaddf(aref, cuCaddf(x, 19)))));
    x1[2] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)), cuCaddf(aref, cuCaddf(x, 5))),
        cuCaddf(cuCaddf(cuCaddf(aref, cuCaddf(x, 8)),
                        cuCaddf(aref, cuCaddf(x, 11))),
                cuCaddf(cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                cuCaddf(aref, cuCaddf(x, 17))),
                        cuCaddf(aref, cuCaddf(x, 20)))));
    x1[3] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 3)),
                                                 w7m1_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), w7p5_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 9)),
                                                 w7p4_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                                                w7p3_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 15)),
                                                w7p2_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 18)),
                                                 w7p1_7))))));
    x1[4] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 4)),
                                                 w7m1_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 7)), w7p5_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 10)),
                                                 w7p4_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                                w7p3_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 16)),
                                                w7p2_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 19)),
                                                 w7p1_7))))));
    x1[5] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 5)),
                                                 w7m1_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 8)), w7p5_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 11)),
                                                 w7p4_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                                w7p3_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 17)),
                                                w7p2_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 20)),
                                                 w7p1_7))))));
    x1[6] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 3)),
                                                 w7p5_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), w7p3_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 9)),
                                                 w7p1_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                                                w7m1_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 15)),
                                                w7p4_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 18)),
                                                 w7p2_7))))));
    x1[7] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 4)),
                                                 w7p5_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 7)), w7p3_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 10)),
                                                 w7p1_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                                w7m1_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 16)),
                                                w7p4_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 19)),
                                                 w7p2_7))))));
    x1[8] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 5)),
                                                 w7p5_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 8)), w7p3_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 11)),
                                                 w7p1_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                                w7m1_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 17)),
                                                w7p4_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 20)),
                                                 w7p2_7))))));
    x1[9] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 3)),
                                                 w7p4_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), w7p1_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 9)),
                                                 w7p5_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                                                w7p2_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 15)),
                                                w7m1_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 18)),
                                                 w7p3_7))))));
    x1[10] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 4)),
                                                 w7p4_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 7)), w7p1_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 10)),
                                                 w7p5_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                                w7p2_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 16)),
                                                w7m1_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 19)),
                                                 w7p3_7))))));
    x1[11] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 5)),
                                                 w7p4_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 8)), w7p1_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 11)),
                                                 w7p5_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                                w7p2_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 17)),
                                                w7m1_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 20)),
                                                 w7p3_7))))));
    x1[12] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 3)),
                                                 w7p3_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), w7m1_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 9)),
                                                 w7p2_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                                                w7p5_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 15)),
                                                w7p1_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 18)),
                                                 w7p4_7))))));
    x1[13] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 4)),
                                                 w7p3_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 7)), w7m1_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 10)),
                                                 w7p2_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                                w7p5_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 16)),
                                                w7p1_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 19)),
                                                 w7p4_7))))));
    x1[14] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 5)),
                                                 w7p3_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 8)), w7m1_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 11)),
                                                 w7p2_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                                w7p5_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 17)),
                                                w7p1_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 20)),
                                                 w7p4_7))))));
    x1[15] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 3)),
                                                 w7p2_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), w7p4_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 9)),
                                                 w7m1_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                                                w7p1_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 15)),
                                                w7p3_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 18)),
                                                 w7p5_7))))));
    x1[16] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 4)),
                                                 w7p2_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 7)), w7p4_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 10)),
                                                 w7m1_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                                w7p1_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 16)),
                                                w7p3_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 19)),
                                                 w7p5_7))))));
    x1[17] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 5)),
                                                 w7p2_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 8)), w7p4_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 11)),
                                                 w7m1_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                                w7p1_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 17)),
                                                w7p3_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 20)),
                                                 w7p5_7))))));
    x1[18] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 0)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 3)),
                                                 w7p1_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 6)), w7p2_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 9)),
                                                 w7p3_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 12)),
                                                w7p4_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 15)),
                                                w7p5_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 18)),
                                                 w7m1_7))))));
    x1[19] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 1)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 4)),
                                                 w7p1_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 7)), w7p2_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 10)),
                                                 w7p3_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 13)),
                                                w7p4_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 16)),
                                                w7p5_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 19)),
                                                 w7m1_7))))));
    x1[20] = cuCaddf(
        cuCaddf(cuCaddf(aref, cuCaddf(x, 2)),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 5)),
                                                 w7p1_7)))),
        cuCaddf(
            cuCaddf(
                cuCaddf(funcall,
                        cuCaddf(cuCmulf,
                                cuCaddf(cuCaddf(aref, cuCaddf(x, 8)), w7p2_7))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 11)),
                                                 w7p3_7)))),
            cuCaddf(
                cuCaddf(cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 14)),
                                                w7p4_7))),
                        cuCaddf(funcall,
                                cuCaddf(cuCmulf,
                                        cuCaddf(cuCaddf(aref, cuCaddf(x, 17)),
                                                w7p5_7)))),
                cuCaddf(funcall,
                        cuCaddf(cuCmulf, cuCaddf(cuCaddf(aref, cuCaddf(x, 20)),
                                                 w7m1_7))))));
    // multiply with twiddle factors and transpose;
    {
      complex x2[21];
      const complex w21m1_21 =
          CMPLXF((9.555728057861407e-1), (-2.947551744109042e-1));
      const complex w21p19_21 =
          CMPLXF((8.262387743159949e-1), (-5.63320058063622e-1));
      const complex w21p17_21 =
          CMPLXF((3.65341024366395e-1), (-9.308737486442042e-1));
      const complex w21p6_7 =
          CMPLXF((6.234898018587335e-1), (-7.818314824680297e-1));
      const complex w21p5_7 =
          CMPLXF((-2.2252093395631434e-1), (-9.749279121818235e-1));
      const complex w21p13_21 =
          CMPLXF((-7.330518718298263e-1), (-6.801727377709194e-1));
      const complex w21p16_21 =
          CMPLXF((7.473009358642417e-2), (-9.9720379718118e-1));
      const complex w21p11_21 =
          CMPLXF((-9.888308262251285e-1), (-1.4904226617617428e-1));
      const complex w21p3_7 =
          CMPLXF((-9.009688679024191e-1), (4.33883739117558e-1));
      x2[0] = x1[0];
      x2[7] = x1[1];
      x2[14] = x1[2];
      x2[1] = x1[3];
      x2[8] = cuCmulf(x1[4], w21m1_21);
      x2[15] = cuCmulf(x1[5], w21p19_21);
      x2[2] = x1[6];
      x2[9] = cuCmulf(x1[7], w21p19_21);
      x2[16] = cuCmulf(x1[8], w21p17_21);
      x2[3] = x1[9];
      x2[10] = cuCmulf(x1[10], w21p6_7);
      x2[17] = cuCmulf(x1[11], w21p5_7);
      x2[4] = x1[12];
      x2[11] = cuCmulf(x1[13], w21p17_21);
      x2[18] = cuCmulf(x1[14], w21p13_21);
      x2[5] = x1[15];
      x2[12] = cuCmulf(x1[16], w21p16_21);
      x2[19] = cuCmulf(x1[17], w21p11_21);
      x2[6] = x1[18];
      x2[13] = cuCmulf(x1[19], w21p5_7);
      x2[20] = cuCmulf(x1[20], w21p3_7);
      // another dft;
      {
        complex *x3 = (dst + (21 * i));
        const complex w3m1_3 =
            CMPLXF((-4.999999999999997e-1), (-8.660254037844386e-1));
        const complex w3p1_3 =
            CMPLXF((-5.000000000000004e-1), (8.660254037844384e-1));
        x3[0] = (x2[0] + x2[7] + x2[14]);
        x3[7] = (x2[0] + cuCmulf(x2[7], w3m1_3) + cuCmulf(x2[14], w3p1_3));
        x3[14] = (x2[0] + cuCmulf(x2[7], w3p1_3) + cuCmulf(x2[14], w3m1_3));
        x3[1] = (x2[1] + x2[8] + x2[15]);
        x3[8] = (x2[1] + cuCmulf(x2[8], w3m1_3) + cuCmulf(x2[15], w3p1_3));
        x3[15] = (x2[1] + cuCmulf(x2[8], w3p1_3) + cuCmulf(x2[15], w3m1_3));
        x3[2] = (x2[2] + x2[9] + x2[16]);
        x3[9] = (x2[2] + cuCmulf(x2[9], w3m1_3) + cuCmulf(x2[16], w3p1_3));
        x3[16] = (x2[2] + cuCmulf(x2[9], w3p1_3) + cuCmulf(x2[16], w3m1_3));
        x3[3] = (x2[3] + x2[10] + x2[17]);
        x3[10] = (x2[3] + cuCmulf(x2[10], w3m1_3) + cuCmulf(x2[17], w3p1_3));
        x3[17] = (x2[3] + cuCmulf(x2[10], w3p1_3) + cuCmulf(x2[17], w3m1_3));
        x3[4] = (x2[4] + x2[11] + x2[18]);
        x3[11] = (x2[4] + cuCmulf(x2[11], w3m1_3) + cuCmulf(x2[18], w3p1_3));
        x3[18] = (x2[4] + cuCmulf(x2[11], w3p1_3) + cuCmulf(x2[18], w3m1_3));
        x3[5] = (x2[5] + x2[12] + x2[19]);
        x3[12] = (x2[5] + cuCmulf(x2[12], w3m1_3) + cuCmulf(x2[19], w3p1_3));
        x3[19] = (x2[5] + cuCmulf(x2[12], w3p1_3) + cuCmulf(x2[19], w3m1_3));
        x3[6] = (x2[6] + x2[13] + x2[20]);
        x3[13] = (x2[6] + cuCmulf(x2[13], w3m1_3) + cuCmulf(x2[20], w3p1_3));
        x3[20] = (x2[6] + cuCmulf(x2[13], w3p1_3) + cuCmulf(x2[20], w3m1_3));
      }
    }
  }
}
""")
fft_21_3_7=mod.get_function("fft_21_3_7")
N2=64
n1=3
n2=7
src=np.random.randn([((n1)*(n2)), n2]).astype(np.complex64)
result=np.zeros_like(a)
fft_21_3_7(drv.Out(dst), drv.In(src), block=(N2,1,1,))
print(dst)