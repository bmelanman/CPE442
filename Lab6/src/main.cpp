/*********************************************************
* File: main.cpp
*
* Description: This program takes full color videos,
*              converts them to grayscale and applies
*              the sobel operator frame by frame. The
*              speed of the video is dependant on the
*              time it takes to process each frame.
*
* Author: Bryce Melander
* Co-Authors: Blase Parker, Johnathan Espiritu
*
* Revisions: V4.1
**********************************************************/

/***** Includes *****/
#include <opencv2/opencv.hpp>
#include "../lib/pthread_barrier.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <arm_neon.h>

/***** Namespaces *****/
using namespace std;
using namespace cv;

/***** Defines *****/
#define B_CONST 0.2126
#define G_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

/***** Structures *****/
struct big_thread_data {
    Mat *orig_frame{};
    Mat *gray_frame{};
    Mat *sobl_frame{};
    int thread_id{};
};
typedef struct int16x8x8_t {
    int16x8_t val[8];
} int16x8x8_t;
typedef struct uint8x8x8_t {
    uint8x8_t val[8];
} uint8x8x8_t;

/***** Global Variables *****/
pthread_barrier_t gray_barrier;
pthread_barrier_t sobl_barrier;

// While-loop lock variable for keeping the threads in their respective filters
bool thread_lock = false;

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

/***** Prototypes *****/

void *filter(void *threadArgs);

/***** Main *****/
int main(int argc, char const *argv[]) {

    // validate input args
    if (argc != 2) {
        cout << "Invalid input" << endl;
        exit(-1);
    }

    // get video file name
    string usrVideo_location = argv[1];
    ifstream ifile;
    ifile.open(usrVideo_location);

    // check file exists
    if (!ifile) {
        cout << "File does not exist" << endl;
        exit(-1);
    }

    // Open the file as a video and display info for user
    VideoCapture usrVideo(usrVideo_location);
    auto fps = (uint8_t) usrVideo.get(CAP_PROP_FPS);
    int num_frames = (int) usrVideo.get(CAP_PROP_FRAME_COUNT);
    printf("Video Resolution: %dx%d\n", (int) usrVideo.get(CAP_PROP_FRAME_HEIGHT),
           (int) usrVideo.get(CAP_PROP_FRAME_WIDTH));
    printf("Video Length: %d.%d seconds\n", num_frames / fps, (num_frames / fps) % 1);
    printf("Number of threads: %d\n", NUM_THREADS);

    // Init Mats for each filter
    int usrVideo_rows = (int) usrVideo.get(4);
    int usrVideo_cols = (int) usrVideo.get(3);
    Mat frame(usrVideo_rows, usrVideo_cols, CV_8UC3);
    Mat gray_frame(usrVideo_rows, usrVideo_cols, CV_8UC1);
    Mat sobl_frame(usrVideo_rows - 2, usrVideo_cols - 2, CV_8UC1);

    // Init pthreads and barriers
    pthread_t filter_threads[NUM_THREADS];
    pthread_barrier_init(&gray_barrier, nullptr, NUM_THREADS + 1);
    pthread_barrier_init(&sobl_barrier, nullptr, NUM_THREADS + 1);

    // Init thread struct variables
    struct big_thread_data thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {

        // init filter thread variables
        thread_data[i] = {
                &frame,
                &gray_frame,
                &sobl_frame,
                i
        };

        pthread_create(&filter_threads[i], nullptr, &filter, (void *) &thread_data[i]);
    }

    // Loop through the image file
    while (usrVideo.isOpened()) {

        // Wait for the image to be processes
        pthread_barrier_wait(&gray_barrier);
        pthread_barrier_wait(&sobl_barrier);

        // Get a frame from the video
        usrVideo >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            // set the done flag high to release the threads from the filters
            thread_lock = true;
            break;
        }

        // Display the frame
        imshow(usrVideo_location, sobl_frame);

        // Hold ESC to exit the video early
        if ((char) waitKey(1) == 27) {
            thread_lock = true;
            break;
        }
    }

    // Rejoin all threads
    pthread_barrier_wait(&gray_barrier);
    pthread_barrier_wait(&sobl_barrier);

    // Clean up threads
    for (auto &filter_thread: filter_threads) {
        pthread_join(filter_thread, nullptr);
    }

    // Clean up thread barriers
    pthread_barrier_destroy(&gray_barrier);
    pthread_barrier_destroy(&sobl_barrier);

    // clean up video stuff
    usrVideo.release();
    destroyAllWindows();

    return 0;
}

/***** Project Functions *****/

/**
 * A grayscale filter for color images, applies the ITU-R (BT.709) grayscale algorithm. The function utilizes ARM neon
 * vectors to optimize the speed of the grayscale algorithm.
 * @param image An image
 * @param grayscale A grayscale image
 * Takes a grayscale image and applies the sobel operator to the given image.
 * @param threadArgs - A struct with variables for the sobel filter
 */
void *filter(void *threadArgs) {

    /***** Filter variables *****/
    uint8x8x3_t BGR_values;
    uint8x8x8_t gray_pixels;
    int16x8_t Gx_vect, Gy_vect;
    uint16x8_t G_vect;
    int i0, i1, i2, G;

    /***** Thread variables *****/
    auto *thread_data = (struct big_thread_data *) threadArgs;
    uchar *orig_frame_data;
    uchar *gray_frame_data;
    uchar *sobl_frame_data = thread_data->sobl_frame->data;
    uint8_t thread_id = thread_data->thread_id;

    int orig_rows = thread_data->orig_frame->rows;
    int orig_cols = thread_data->orig_frame->cols;
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

    /***** Begin filtering *****/

    // Lock the threads in the image processing loop until we've processed every frame of the video
    while (!thread_lock) {

        /***** Grayscale conversion *****/

        // Reset data address position for each loop
        orig_frame_data = thread_data->orig_frame->data + (gray_start * 3);
        gray_frame_data = thread_data->gray_frame->data + gray_start;

        // Go through the image 8 pixels at a time
        for (int i = gray_start; i < gray_stop; i += 8, orig_frame_data += 24, gray_frame_data += 8) {

            // Load 8 pixels in a vector
            BGR_values = vld3_u8(orig_frame_data);

            // Multiply each pixel by the predefined grayscale ratios
            // Reusing G_vect to use fewer variables overall
            G_vect = vmull_u8(BGR_values.val[0], Blue_vect);

            G_vect = vmlal_u8(G_vect, BGR_values.val[1], Green_vect);
            G_vect = vmlal_u8(G_vect, BGR_values.val[2], Red_vect);

            // Write 8 pixels at a time to grayscale frame
            vst1_u8(gray_frame_data, vshrn_n_u16(G_vect, 8));
        }

        // If the end of the image has fewer than 8 pixels, process the remainder one pixel at a time
        // This only runs for a maximum of 7 pixels , so single pixel operation is fine
        for (int i = gray_stop; i < gray_stop + gray_remainder_pixels; i++, orig_frame_data += 3, gray_frame_data++) {
            *gray_frame_data = (uchar) ((*orig_frame_data * B_CONST) +
                                        (*orig_frame_data * G_CONST) +
                                        (*orig_frame_data * R_CONST));
        }

        // Wait until all threads are done with grayscale
        pthread_barrier_wait(&gray_barrier);

        gray_frame_data = thread_data->gray_frame->data;

        /***** Sobel conversion *****/

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

        // Wait until the main loop moves on to the next image
        pthread_barrier_wait(&sobl_barrier);
    }
    pthread_exit(nullptr);
}