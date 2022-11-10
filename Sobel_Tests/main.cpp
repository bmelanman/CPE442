#include <opencv2/opencv.hpp>
#include <string>
#include <arm_neon.h>

using namespace std;
using namespace cv;

#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

typedef struct int16x8x8_t {
    int16x8_t val[8];
} int16x8x8_t;
typedef struct uint8x8x8_t {
    uint8x8_t val[8];
} uint8x8x8_t;

// These values will be divided by 256 by the grayscale filter to avoid float math
const uint8x8_t Green_vect = vdup_n_u8(183);    // G_CONST * 256
const uint8x8_t Blue_vect = vdup_n_u8(54);      // B_CONST * 256
const uint8x8_t Red_vect = vdup_n_u8(19);       // R_CONST * 256, rounded up

// Overflow check used to make sure vectors are not larger
// than 255 before converting to an 8 bit vector
const uint16x8_t Overflow_check = vdupq_n_u16(255);

// Kernels used for grayscale filtering
// Small kernel is used for single pixel processing
const int16x8_t Gx_kernel_small = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t Gy_kernel_small = {1, 2, 1, 0, 0, -1, -2, -1};

// Large kernel is used for processing 8 pixels at a time
const int16x8x8_t Gx_kernel = {
        {vdupq_n_s16(1), vdupq_n_s16(2), vdupq_n_s16(1), vdupq_n_s16(0),
         vdupq_n_s16(0), vdupq_n_s16(-1), vdupq_n_s16(-2), vdupq_n_s16(-1)}};
const int16x8x8_t Gy_kernel = {
        {vdupq_n_s16(-1), vdupq_n_s16(0), vdupq_n_s16(1), vdupq_n_s16(-2),
         vdupq_n_s16(2), vdupq_n_s16(-1), vdupq_n_s16(0), vdupq_n_s16(1)}};

void filter(Mat *origImg, Mat *grayImg, Mat *soblImg, uint8_t thread_id) {

    /***** Grayscale variables *****/
    uint8x8x3_t BGR_values;
    uint16x8_t temp;

    /***** Sobel variables *****/
    int i0, i1, i2, G;
    uint8x8x8_t gray_pixels;
    int16x8_t Gx_vect, Gy_vect;
    uint16x8_t G_vect;

    /***** Thread variables *****/
    uchar *orig_frame_data;
    uchar *gray_frame_data;
    uchar *sobl_frame_data = soblImg->data;

    int orig_rows = origImg->rows;
    int orig_cols = origImg->cols;
    int orig_num_pixels = orig_rows * orig_cols;
    int sobl_cols = orig_cols - 2;

    // Grayscale filter variables
    int gray_remainder_pixels = orig_num_pixels % (8 * NUM_THREADS);
    int gray_start = thread_id * (orig_num_pixels - gray_remainder_pixels) / NUM_THREADS;
    int gray_stop = (thread_id + 1) * (orig_num_pixels - gray_remainder_pixels) / NUM_THREADS - 1;

    // Sobel filter variables
    int sobl_start = thread_id * orig_rows / NUM_THREADS;
    int sobl_stop = (thread_id + 1) * orig_rows / NUM_THREADS;
    int sobl_remainder = (sobl_cols) % 8;
    int sobl_col_stop = sobl_cols - sobl_remainder;

    if (thread_id == NUM_THREADS - 1) {
        sobl_stop = orig_rows - 2;
    }

    /***** Grayscale conversion *****/

    // Reset data address position for each loop
    orig_frame_data = origImg->data + (gray_start * 3);
    gray_frame_data = grayImg->data + gray_start;

    // Go through the image 8 pixels at a time
    for (int i = gray_start; i < gray_stop; i += 8, orig_frame_data += 24, gray_frame_data += 8) {

        // Load 8 pixels in a vector
        BGR_values = vld3_u8(orig_frame_data);

        // Multiply each pixel by the predefined grayscale ratios
        temp = vmull_u8(BGR_values.val[0], Blue_vect);

        temp = vmlal_u8(temp, BGR_values.val[1], Green_vect);
        temp = vmlal_u8(temp, BGR_values.val[2], Red_vect);

        // Write 8 pixels at a time to grayscale frame
        vst1_u8(gray_frame_data, vshrn_n_u16(temp, 8));
    }

    // If the end of the image has fewer than 8 pixels, process the remainder one pixel at a time
    for (int i = gray_stop; i < gray_stop + gray_remainder_pixels; i++, orig_frame_data += 3, gray_frame_data++) {
        *gray_frame_data = (uchar) ((*orig_frame_data * B_CONST) +
                                    (*orig_frame_data * G_CONST) +
                                    (*orig_frame_data * R_CONST));
    }

    /***** Sobel conversion *****/

    // Reset data address back to beginning
    gray_frame_data = grayImg->data;

    // Loop through the rows and cols of the image and apply the sobel filter
    for (int row = sobl_start; row < sobl_stop; row++) {

        // Calculate indexes
        i0 = orig_cols * row;
        i1 = orig_cols * (row + 1);
        i2 = orig_cols * (row + 2);

        for (int col = 0; col < sobl_col_stop; col += 8) {

            // Load pixel sets into vectors
            gray_pixels.val[0] = vld1_u8(gray_frame_data + i0 + col);
            gray_pixels.val[1] = vld1_u8(gray_frame_data + i0 + 1 + col);
            gray_pixels.val[2] = vld1_u8(gray_frame_data + i0 + 2 + col);

            gray_pixels.val[3] = vld1_u8(gray_frame_data + i1 + col);
            gray_pixels.val[4] = vld1_u8(gray_frame_data + i1 + 2 + col);

            gray_pixels.val[5] = vld1_u8(gray_frame_data + i2 + col);
            gray_pixels.val[6] = vld1_u8(gray_frame_data + i2 + 1 + col);
            gray_pixels.val[7] = vld1_u8(gray_frame_data + i2 + 2 + col);

            // Multiply each pixel by the X kernel and sum them all together
            // This operation also clears old values
            Gx_vect = vmulq_s16(Gx_kernel.val[0], vmovl_u8(gray_pixels.val[0]));

            for (int i = 1; i < 8; i++) {
                Gx_vect = vmlaq_s16(Gx_vect, Gx_kernel.val[i], vmovl_u8(gray_pixels.val[i]));
            }

            // Repeat for the Y kernel
            Gy_vect = vmulq_s16(Gy_kernel.val[0], vmovl_u8(gray_pixels.val[0]));

            for (int i = 1; i < 8; i++) {
                Gy_vect = vmlaq_s16(Gy_vect, Gy_kernel.val[i], vmovl_u8(gray_pixels.val[i]));
            }

            // Gradient Approximation
            G_vect = vaddq_u16(vabsq_s16(Gx_vect), vabsq_s16(Gy_vect));

            // Overflow check
            G_vect = vminq_u16(G_vect, Overflow_check);

            // The pixel values are guaranteed to be less than 256, so they can be converted
            // to uint8_t vectors and written to the sobel image 8 pixels at a time
            vst1_u8(sobl_frame_data + (sobl_cols * row) + col, vmovn_u16(G_vect));
        }

        for (int r_col = sobl_col_stop; r_col < sobl_cols; r_col++) {

            // If there's some remaining pixels, filter them one by one
            // Reusing variables here
            G_vect = (uint16x8_t) {
                    gray_frame_data[i0 + r_col], gray_frame_data[i0 + 1 + r_col], gray_frame_data[i0 + 2 + r_col],
                    gray_frame_data[i1 + r_col], gray_frame_data[i1 + 2 + r_col],
                    gray_frame_data[i2 + r_col], gray_frame_data[i2 + 1 + r_col], gray_frame_data[i2 + 2 + r_col]
            };

            // Calculate G from the pixels, lots of conversions at once to avoid using more variables
            G = abs(vaddlvq_s16(vmulq_s16(Gx_kernel_small, G_vect))) +
                abs(vaddlvq_s16(vmulq_s16(Gy_kernel_small, G_vect)));

            // Overflow check
            if (G > 255) { G = 255; }

            // Write the pixel to the sobel image
            sobl_frame_data[sobl_cols * row + r_col] = (uchar) G;
        }
    }
}

int main() {

    // Get the name of the file the user is requesting
    string usr_arg = "/Users/brycemelander/Documents/GitHub/CPE442/Media/valve.PNG";

    // init image Mats
    Mat usr_img = imread(usr_arg);

    // Calculate starts and stops
    int num_rows = usr_img.rows;
    int num_cols = usr_img.cols;
    Mat gray_img(num_rows, num_cols, CV_8UC1);
    Mat sobl_img(num_rows - 2, num_cols - 2, CV_8UC1);

    printf("%dx%d -> %dx%d", num_rows, num_cols, num_rows - 2, num_cols - 2);

    // init and run grayscale filter
    for (int i = 0; i < NUM_THREADS; i++) {

        filter(&usr_img, &gray_img, &sobl_img, i);
    }

    resize(gray_img, gray_img, Size(), 10, 10, INTER_LANCZOS4);
    resize(sobl_img, sobl_img, Size(), 10, 10, INTER_LANCZOS4);

    imshow(usr_arg, gray_img);
    waitKey(0);
    imshow(usr_arg, sobl_img);
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();

    return 0;
}
