#include <iostream>
#include <arm_neon.h>
#include <cmath>

using namespace std;

struct uint32x2x6_t {
    uint32x2_t val[6];
};

int main() {

    int Gx, Gy;
    int numRows = 64;
    int numCols = 256;
    int ele = numRows * numCols;

    int arr[ele];
    fill(arr, arr + ele, 0);

    uint32x4_t row_vect, col_vect;
    uint32x4_t numCol_vect = vdupq_n_u32(numCols);

    uint32x4x4_t Gx_vect, Gx_conv_vect = {2, 1, 0, 2,
                                          1, 0, 0, 0,
                                          2, 2, 2, 0,
                                          0, 0, 0, 0};

    uint32x4x4_t Gy_vect, Gy_conv_vect = {0, 0, 0, 2,
                                          2, 2, 0, 0,
                                          0, 1, 2, 0,
                                          1, 2, 0, 0};

//    row_vect = vaddq_u16(vdupq_n_u16(row), Gx_row_vect);    // row_sum = (row + Gx[i])
//    row_vect = vmulq_n_u16(row_vect, numCols);                  // row_sum *= numCols
//    col_vect = vaddq_u16(vdupq_n_u16(col), Gx_col_vect);    // col_sum = (col + Gx[j])
//    Gx_vect = vaddq_u16(row_vect, col_vect);                    // Gx_vect = row_sum + col_sum

    // Loop through the rows and cols of the image and apply the sobel filter
    for (unsigned int row = 0; row < numRows - 2; row++) {
        for (unsigned int col = 0; col < numCols - 2; col++) {

            // uint32x4x2_t temp = vld2q_u32((unsigned int[]) {row});

            // Put row and col into vectors
            row_vect = vdupq_n_u32(row);
            col_vect = vdupq_n_u32(col);

            // Convolve Gx
            // Calculate [numCols * (row + x) + (col + y)]
            row_vect = vaddq_u32(row_vect, Gx_conv_vect.val[0]);
            col_vect = vaddq_u32(col_vect, Gx_conv_vect.val[2]);
            Gx_vect.val[0] = vmlaq_u32(col_vect, row_vect, numCol_vect);

            row_vect = vaddq_u32(row_vect, Gx_conv_vect.val[1]);
            col_vect = vaddq_u32(col_vect, Gx_conv_vect.val[3]);
            Gx_vect.val[1] = vmlaq_u32(col_vect, row_vect, numCol_vect);

            // Convolve Gy
            row_vect = vaddq_u32(row_vect, Gy_conv_vect.val[0]);
            col_vect = vaddq_u32(col_vect, Gy_conv_vect.val[2]);
            Gy_vect.val[0] = vmlaq_u32(col_vect, row_vect, numCol_vect);

            row_vect = vaddq_u32(row_vect, Gy_conv_vect.val[1]);
            col_vect = vaddq_u32(col_vect, Gy_conv_vect.val[3]);
            Gy_vect.val[1] = vmlaq_u32(col_vect, row_vect, numCol_vect);

            Gx = (arr[vgetq_lane_u16(Gx_vect.val[0], 0)]) +
                 (arr[vgetq_lane_u16(Gx_vect.val[0], 1)] << 1) +
                 (arr[vgetq_lane_u16(Gx_vect.val[0], 2)]) -
                 (arr[vgetq_lane_u16(Gx_vect.val[0], 3)]) -
                 (arr[vgetq_lane_u16(Gx_vect.val[1], 4)] << 1) -
                 (arr[vgetq_lane_u16(Gx_vect.val[1], 5)]);

            Gy = (arr[vgetq_lane_u16(Gy_vect.val[0], 0)]) +
                 (arr[vgetq_lane_u16(Gy_vect.val[0], 1)] << 1) +
                 (arr[vgetq_lane_u16(Gy_vect.val[0], 2)]) -
                 (arr[vgetq_lane_u16(Gy_vect.val[0], 3)]) -
                 (arr[vgetq_lane_u16(Gy_vect.val[1], 4)] << 1) -
                 (arr[vgetq_lane_u16(Gy_vect.val[1], 5)]);

            cout << Gx + Gy;

            return 0;
        }
    }
}
