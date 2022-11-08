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
#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

/***** Structures *****/
struct thread_data {
    Mat *input{};
    Mat *output{};
    int start{};
    int stop{};
    int remainder{0};
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
const uint8x8_t Green_vect = vdup_n_u8(54); // G_CONST * 256
const uint8x8_t Blue_vect = vdup_n_u8(183); // B_CONST * 256
const uint8x8_t Red_vect = vdup_n_u8(19);   // R_CONST * 256, rounded up

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
void *gray_thread_vector(void *threadArgs);

void *sobl_thread_vector(void *threadArgs);

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
    pthread_t gray_threads[NUM_THREADS];
    pthread_t sobl_threads[NUM_THREADS];
    pthread_barrier_init(&gray_barrier, nullptr, NUM_THREADS + 1);
    pthread_barrier_init(&sobl_barrier, nullptr, NUM_THREADS + 1);

    // Calculate the number of remaining pixels that can't be grouped in sets of 8
    int usrVideo_numPixels = usrVideo_rows * usrVideo_cols;
    int remainder_pixels = usrVideo_numPixels % (8 * NUM_THREADS);
    int num_pixels = usrVideo_numPixels - remainder_pixels;

    // Init thread struct variables
    struct thread_data gray_data[NUM_THREADS], sobl_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {

        // init grayscale filter thread variables
        gray_data[i].input = &frame;
        gray_data[i].output = &gray_frame;
        gray_data[i].start = i * num_pixels / NUM_THREADS;
        gray_data[i].stop = (i + 1) * num_pixels / NUM_THREADS - 1;

        // Give the last thread any remainder pixels
        if (i == NUM_THREADS - 1 && remainder_pixels != 0) {
            gray_data[i].remainder = remainder_pixels;
        }

        // Run the grayscale filter threads
        pthread_create(&gray_threads[i], nullptr, &gray_thread_vector, (void *) &gray_data[i]);

        // init the sobel filter thread variables
        sobl_data[i].input = &gray_frame;
        sobl_data[i].output = &sobl_frame;
        sobl_data[i].start = i * usrVideo_rows / NUM_THREADS;
        sobl_data[i].remainder = (usrVideo_cols - 2) % 8;

        // Special case for the last thread, it must go all the way to the end
        if (i == NUM_THREADS - 1) { sobl_data[i].stop = usrVideo_rows - 2; }
        else { sobl_data[i].stop = (i + 1) * usrVideo_rows / NUM_THREADS; }

        // Run the sobel filter threads
        pthread_create(&sobl_threads[i], nullptr, sobl_thread_vector, (void *) &sobl_data[i]);
    }

    // Allow the grayscale filter to process one frame before the sobel filter runs
    usrVideo >> frame;
    pthread_barrier_wait(&gray_barrier);

    // Loop through the image file
    while (usrVideo.isOpened()) {

        // Get a frame from the video
        usrVideo >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            // set the done flag high to release the threads from the filters
            thread_lock = true;
            break;
        }

        // Process the image
        pthread_barrier_wait(&gray_barrier);
        pthread_barrier_wait(&sobl_barrier);

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
    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(gray_threads[i], nullptr);
        pthread_join(sobl_threads[i], nullptr);
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
 */
void *gray_thread_vector(void *threadArgs) {

    // Function variables
    uchar *origImgData;
    uchar *grayImgData;
    uint8x8x3_t BGR_values;
    uint16x8_t temp;

    // Thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    int remainder = thread_data->remainder;
    int start = thread_data->start;
    int stop = thread_data->stop;

    // Lock the threads in the image processing loop until we've processed every frame of the video
    while (!thread_lock) {

        // Reset data address position for each loop
        origImgData = thread_data->input->data + (start * 3);
        grayImgData = thread_data->output->data + start;

        // Go through the image 8 pixels at a time
        for (int i = start; i < stop; i += 8, origImgData += 24, grayImgData += 8) {

            // Load 8 pixels in a vector
            BGR_values = vld3_u8(origImgData);

            // Multiply each pixel by the predefined grayscale ratios
            temp = vmull_u8(BGR_values.val[0], Blue_vect);

            temp = vmlal_u8(temp, BGR_values.val[1], Green_vect);
            temp = vmlal_u8(temp, BGR_values.val[2], Red_vect);

            // Write 8 pixels at a time to grayscale frame
            vst1_u8(grayImgData, vshrn_n_u16(temp, 8));
        }

        // If the end of the image has fewer than 8 pixels, process the remainder one pixel at a time
        for (int i = stop; i < stop + remainder; i++, origImgData += 3, grayImgData++) {
            *grayImgData = (uchar) ((*origImgData * B_CONST) +
                                    (*origImgData * G_CONST) +
                                    (*origImgData * R_CONST));
        }

        // Wait until the main loop moves on to the next image
        pthread_barrier_wait(&gray_barrier);
    }
    pthread_exit(nullptr);
}

/**
 * Takes a grayscale image and applies the sobel operator to the given image.
 * @param threadArgs - A struct with variables for the sobel filter
 */
void *sobl_thread_vector(void *threadArgs) {

    // Function Variables
    int i0, i1, i2, G;
    uint8x8x8_t gray_pixels;
    int16x8_t Gx_vect, Gy_vect;
    uint16x8_t G_vect;

    // Thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    uchar *gray_data = thread_data->input->data;
    uchar *sobl_data = thread_data->output->data;
    int numCols = thread_data->input->cols;
    int start = thread_data->start;
    int stop = thread_data->stop;
    int remainder = thread_data->remainder;

    int numCols_sobel = numCols - 2;
    int col_stop = numCols_sobel - remainder;

    // Lock the threads in the image processing loop until we've processed every frame of the video
    while (!thread_lock) {

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
                vst1_u8(sobl_data + (numCols_sobel * row) + col, vmovn_u16(G_vect));
            }

            for (int r_col = col_stop; r_col < numCols_sobel; r_col++) {

                // If there's some remaining pixels, filter them one by one
                // Reusing variables here
                G_vect = (uint16x8_t) {
                        gray_data[i0 + r_col], gray_data[i0 + 1 + r_col], gray_data[i0 + 2 + r_col],
                        gray_data[i1 + r_col], gray_data[i1 + 2 + r_col],
                        gray_data[i2 + r_col], gray_data[i2 + 1 + r_col], gray_data[i2 + 2 + r_col]
                };

                // Calculate G from the pixels, lots of conversions at once to avoid using more variables
                G = abs(vaddlvq_s16(vmulq_s16(Gx_kernel_small, G_vect))) +
                    abs(vaddlvq_s16(vmulq_s16(Gy_kernel_small, G_vect)));

                // Overflow check
                if (G > 255) { G = 255; }

                // Write the pixel to the sobel image
                sobl_data[numCols_sobel * row + r_col] = (uchar) G;
            }
        }

        // Wait until the main loop moves on to the next image
        pthread_barrier_wait(&sobl_barrier);
    }
    pthread_exit(nullptr);
}

