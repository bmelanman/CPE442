#include <opencv2/opencv.hpp>
#include <string>
#include <arm_neon.h>

using namespace std;
using namespace cv;

#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

typedef struct int8x8x8_t {
    int8x8_t val[8];
} int8x8x8_t;

const uint8x8_t Green_vect = vdup_n_u8(G_CONST * 256);
const uint8x8_t Blue_vect = vdup_n_u8(B_CONST * 256);
const uint8x8_t Red_vect = vdup_n_u8(R_CONST * 256);
const uint16x8_t Overflow_check = vdupq_n_u16(255);

const int16x8_t Gx_kernel_small = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t Gy_kernel_small = {1, 2, 1, 0, 0, -1, -2, -1};

const int8x8x8_t Gx_kernel = {
        {vdup_n_s8(-1), vdup_n_s8(0), vdup_n_s8(1), vdup_n_s8(-2),
         vdup_n_s8(2), vdup_n_s8(-1), vdup_n_s8(0), vdup_n_s8(1)}
};
const int8x8x8_t Gy_kernel = {
        {vdup_n_s8(1), vdup_n_s8(2), vdup_n_s8(1), vdup_n_s8(0),
         vdup_n_s8(0), vdup_n_s8(-1), vdup_n_s8(-2), vdup_n_s8(-1)}
};

void grayscale_filter_vector(Mat *origImg, Mat *grayImg, int start, int stop, int remainder) {

    uchar *origImgData = origImg->data + (start * 3);
    uchar *grayImgData = grayImg->data + start;

    uint8x8x3_t BGR_values;
    uint16x8_t temp;

    for (int i = start; i < stop; /* Operation occurs here */ i += 8, origImgData += 24, grayImgData += 8) {
        BGR_values = vld3_u8(origImgData);

        temp = vmull_u8(BGR_values.val[0], Blue_vect);

        temp = vmlal_u8(temp, BGR_values.val[1], Green_vect);
        temp = vmlal_u8(temp, BGR_values.val[2], Red_vect);

        vst1_u8(grayImgData, vshrn_n_u16(temp, 8));

    }

    for (int i = stop; i < stop + remainder; i++, origImgData += 3, grayImgData++) {
        *grayImgData = (uchar) ((*origImgData * B_CONST) +
                                (*origImgData * G_CONST) +
                                (*origImgData * R_CONST));
    }
}

void sobel_filter_vector(Mat *grayImg, Mat *soblImg, int start, int stop) {

    // init thread variables
    uchar *gray_data = grayImg->data;
    uchar *sobl_data = soblImg->data;
    int numCols = grayImg->cols;

    int remainder = ((numCols - 2) % 8);
    int col_stop = numCols - 2 - remainder;

    int i0, i1, i2, G;
    int8x8x8_t gray_pixels;
    int16x8_t Gx_vect, Gy_vect;
    uint16x8_t G_vect;

    // Loop through the rows and cols of the image and apply the sobel filter
    for (int row = start; row < stop; row++) {

        // Calculate indexes
        i0 = numCols * row;
        i1 = numCols * (row + 1);
        i2 = numCols * (row + 2);

        for (int col = 0; col < col_stop; col += 8) {

            // Load pixel sets into vectors
            gray_pixels.val[0] = vld1_u8(gray_data + i0 + col);
            gray_pixels.val[1] = vld1_u8(gray_data + i0 + 1 + col);
            gray_pixels.val[2] = vld1_u8(gray_data + i0 + 2 + col);

            gray_pixels.val[3] = vld1_u8(gray_data + i1 + col);
            gray_pixels.val[4] = vld1_u8(gray_data + i1 + 2 + col);

            gray_pixels.val[5] = vld1_u8(gray_data + i2 + col);
            gray_pixels.val[6] = vld1_u8(gray_data + i2 + 1 + col);
            gray_pixels.val[7] = vld1_u8(gray_data + i2 + 2 + col);

            // This operation also clears old values
            Gx_vect = vmull_s8(Gx_kernel.val[0], gray_pixels.val[0]);

            for (int i = 1; i < 8; i++) {
                Gx_vect = vmlal_s8(Gx_vect, Gx_kernel.val[i], gray_pixels.val[i]);
            }

            Gy_vect = vmull_s8(Gy_kernel.val[0], gray_pixels.val[0]);

            for (int i = 1; i < 8; i++) {
                Gy_vect = vmlal_s8(Gy_vect, Gy_kernel.val[i], gray_pixels.val[i]);
            }

            // Gradient Approximation
            G_vect = vaddq_u16(vabsq_s16(Gx_vect), vabsq_s16(Gy_vect));

            // Overflow check
            G_vect = vminq_u16(G_vect, Overflow_check);

            // Write the pixel to the sobel image
            vst1q_u8(sobl_data + ((numCols - 2) * row) + col, vreinterpretq_u8_u16(G_vect));
        }

        for (int i = col_stop; i < numCols - 2; i++, i0++, i1++, i2++) {

            // Reusing variables here
            G_vect = (uint16x8_t) {gray_data[i0], gray_data[i0 + 1], gray_data[i0 + 2],
                                   gray_data[i1], gray_data[i1 + 2],
                                   gray_data[i2], gray_data[i2 + 1], gray_data[i2 + 2]};

            // Calculate G from the pixels
            G = abs(vaddlvq_s16(vmulq_s16(Gx_kernel_small, G_vect))) +
                abs(vaddlvq_s16(vmulq_s16(Gy_kernel_small, G_vect)));

            // Overflow check
            if (G > 255) { G = 255; }

            // Write the pixel to the sobel image
            sobl_data[(numCols - 2) * (row) + (col_stop)] = (uchar) G;
        }
    }

}

int main() {

    // Get the name of the file the user is requesting
    string usr_arg = "/Users/brycemelander/Documents/GitHub/CPE442/Media/valve.PNG";
//    string usr_arg = "/Users/brycemelander/Documents/GitHub/CPE442/Media/valve_small.PNG";

    // init image Mats
    Mat usr_img = imread(usr_arg);
    Mat gray_img(usr_img.rows, usr_img.cols, CV_8UC1);
    Mat sobl_img(usr_img.rows - 2, usr_img.cols - 2, CV_8UC1);

    // Calculate starts and stops
    int num_rows = usr_img.rows;
    int num_cols = usr_img.cols;
    int img_size = num_rows * num_cols;
    int remainder_pixels = img_size % (8 * NUM_THREADS);
    int num_pixels = img_size - remainder_pixels;
    int remainder = 0;
    int start;
    int stop;

    // init and run grayscale filter
    for (int i = 0; i < NUM_THREADS; i++) {
        start = i * num_pixels / NUM_THREADS;
        stop = (i + 1) * num_pixels / NUM_THREADS - 1;

        if (i == NUM_THREADS - 1) {
            remainder = remainder_pixels;
        }

        grayscale_filter_vector(&usr_img, &gray_img, start, stop, remainder);
    }

    // init and run sobel filter
    for (int i = 0; i < NUM_THREADS; i++) {

        start = i * num_rows / NUM_THREADS;
        stop = (i + 1) * num_rows / NUM_THREADS;

        if (i == NUM_THREADS - 1) {
            stop = num_rows - 2;
        }

        sobel_filter_vector(&gray_img, &sobl_img, start, stop);
    }

    printf("%dx%d", sobl_img.rows, sobl_img.cols);

//    resize(sobl_img, sobl_img, Size(), 10, 10, INTER_LANCZOS4);

    imshow(usr_arg, sobl_img);
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();

    return 0;
}
