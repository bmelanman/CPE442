/*********************************************************
* File: main.cpp
*
* Description: This program takes full color videos,
*              converts them to grayscale and applies
*              the sobel operator frame by frame.
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

/***** Structures *****/
struct thread_data {
    Mat *input{};
    Mat *output{};
    int start{};
    int stop{};
    int remainder{};
};

typedef struct int8x8x8_t {
    int8x8_t val[8];
} int8x8x8_t;

#define NUM_THREADS 4
/***** Global Variables *****/
pthread_barrier_t gray_barrier;
pthread_barrier_t sobl_barrier;

bool done_flag = false;
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

/***** Prototypes *****/
void *gray_thread_vector(void *threadArgs);

void *sobl_thread_vector(void *threadArgs);

void *test_filter(void *threadArgs);

/***** Main *****/
int main(int argc, char const *argv[]) {

    // validate input args
    if (argc != 2 && argc != 3) {
        cout << "Invalid input" << endl;
        exit(-1);
    }

    // get video file name
    string usr_arg = argv[1];
    ifstream ifile;
    ifile.open(usr_arg);

    // check file exists
    if (!ifile) {
        cout << "File does not exist" << endl;
        exit(-1);
    }

    // calculate and display video length and resolution
    VideoCapture usr_vid(usr_arg);
    int fps = (int) usr_vid.get(CAP_PROP_FPS);
    int frame_count = int(usr_vid.get(CAP_PROP_FRAME_COUNT));
    cout << "Video resolution: " << usr_vid.get(CAP_PROP_FRAME_HEIGHT)
         << "x" << usr_vid.get(CAP_PROP_FRAME_WIDTH) << endl;
    cout << "Video length in seconds: " << (frame_count / fps) << "." << ((frame_count / fps) % 1) << endl;
    cout << "Number of threads: " << NUM_THREADS << endl;

    // init image Mats
    int usr_vid_rows = (int) usr_vid.get(4);
    int usr_vid_cols = (int) usr_vid.get(3);
    Mat frame(usr_vid_rows, usr_vid_cols, CV_8UC3);
    Mat gray_frame(usr_vid_rows, usr_vid_cols, CV_8UC1);
    Mat sobl_frame(usr_vid_rows - 2, usr_vid_cols - 2, CV_8UC1);

    // init pthreads and barriers
    pthread_t gray_threads[NUM_THREADS];
    pthread_t sobl_threads[NUM_THREADS];
    pthread_barrier_init(&gray_barrier, nullptr, NUM_THREADS + 1);
    pthread_barrier_init(&sobl_barrier, nullptr, NUM_THREADS + 1);

    // init and fill thread struct variables
    struct thread_data gray_data[NUM_THREADS], sobl_data[NUM_THREADS];
    int img_size = usr_vid_rows * usr_vid_cols;
    int remainder_pixels = img_size % (8 * NUM_THREADS);
    int num_pixels = img_size - remainder_pixels;
    int remainder = 0;

    for (int i = 0; i < NUM_THREADS; i++) {

        // init grayscale filter thread variables
        gray_data[i].input = &frame;
        gray_data[i].output = &gray_frame;
        gray_data[i].start = i * num_pixels / NUM_THREADS;
        gray_data[i].stop = (i + 1) * num_pixels / NUM_THREADS - 1;
        gray_data[i].remainder = remainder;

        if (i == NUM_THREADS - 1) {
            remainder = remainder_pixels;
        }

        // start running the grayscale filter threads
        pthread_create(&gray_threads[i], nullptr, &gray_thread_vector, (void *) &gray_data[i]);

        // init the sobel filter thread variables
        sobl_data[i].input = &gray_frame;
        sobl_data[i].output = &sobl_frame;
        gray_data[i].start = i * usr_vid_rows / NUM_THREADS;

        // special case for the last thread, it must go all the way to the end
        if (i == NUM_THREADS - 1) { sobl_data[i].stop = usr_vid_rows - 2; }
        else { gray_data[i].stop = (i + 1) * usr_vid_rows / NUM_THREADS; }

        // start running the sobel filter threads
        pthread_create(&sobl_threads[i], nullptr, test_filter, (void *) &sobl_data[i]);
    }

    // Loop through the image file
    while (usr_vid.isOpened()) {

        // Get a frame from the video
        usr_vid >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            // set the done flag high to release the threads from the filters
            done_flag = true;
            break;
        }

        // Process the image
        pthread_barrier_wait(&gray_barrier);
        pthread_barrier_wait(&sobl_barrier);

        // Display the frame
        imshow(usr_arg, sobl_frame);

        // Hold ESC to exit the video early
        if ((char) waitKey(2) == 27) {
            done_flag = true;
            break;
        }
    }

    // collect all the filters together and rejoin everything with the main thread
    pthread_barrier_wait(&gray_barrier);
    pthread_barrier_wait(&sobl_barrier);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(gray_threads[i], nullptr);
        pthread_join(sobl_threads[i], nullptr);
    }

    // clean up thread barriers
    pthread_barrier_destroy(&gray_barrier);
    pthread_barrier_destroy(&sobl_barrier);

    // clean up video stuff
    usr_vid.release();
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

    auto *thread_data = (struct thread_data *) threadArgs;
    int remainder = thread_data->remainder;
    int start = thread_data->start;
    int stop = thread_data->stop;

    uchar *origImgData;
    uchar *grayImgData;
    uint8x8x3_t BGR_values;
    uint16x8_t temp;

    while (!done_flag) {
        origImgData = thread_data->input->data + (start * 3);
        grayImgData = thread_data->output->data + start;

        for (int i = start; i < stop; i += 8, origImgData += 24, grayImgData += 8) {
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
        pthread_barrier_wait(&gray_barrier);
    }
    pthread_exit(nullptr);
}

/**
 * Takes a grayscale image and applies the sobel operator to the given image.
 * @param threadArgs - A struct with variables for the sobel filter
 */
void *sobl_thread_vector(void *threadArgs) {

    // init thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    uchar *gray_data = thread_data->input->data;
    uchar *sobl_data = thread_data->output->data;
    int numCols = thread_data->input->cols;
    int start = thread_data->start;
    int stop = thread_data->stop;
    int i0, i1, i2, Gx, Gy, G;
    int16x8_t pixel_vect;

    // Lock the threads in the filter until the main loop leaves the display loop
    while (!done_flag) {

        // Loop through the rows and cols of the image and apply the sobel filter
        for (int row = start; row < stop; row++) {

            // Calculate indexes
            i0 = numCols * row;
            i1 = numCols * (row + 1);
            i2 = numCols * (row + 2);

            for (int col = 0; col < numCols - 2; col++, i0++, i1++, i2++) {

                // Load pixels into a vector
                pixel_vect = (int16x8_t) {gray_data[i0], gray_data[i0 + 1], gray_data[i0 + 2],
                                          gray_data[i1], gray_data[i1 + 2],
                                          gray_data[i2], gray_data[i2 + 1], gray_data[i2 + 2]};

                // Convolve the pixels with Gx and Gy
                Gx = vaddlvq_s16(vmulq_s16(Gx_kernel_small, pixel_vect));
                Gy = vaddlvq_s16(vmulq_s16(Gy_kernel_small, pixel_vect));

                // Gradient approximation
                G = abs(Gx) + abs(Gy);

                // Overflow check
                if (G > 255) { G = 255; }

                // Write the pixel to the sobel image
                sobl_data[(numCols - 2) * (row) + (col)] = (uchar) G;
            }
        }
        // Wait until the main loop moves on to the next image
        pthread_barrier_wait(&sobl_barrier);
    }
    pthread_exit(nullptr);
}

void *test_filter(void *threadArgs) {

    // init thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    uchar *gray_data = thread_data->input->data;
    uchar *sobl_data = thread_data->output->data;
    int numCols = thread_data->input->cols;
    int start = thread_data->start;
    int stop = thread_data->stop;

    int numCols_sobel = numCols - 2;
    int remainder = (numCols_sobel % 8);
    int col_stop = numCols_sobel - remainder;

    int i0, i1, i2, G;
    int8x8x8_t gray_pixels;
    int16x8_t Gx_vect, Gy_vect;
    uint16x8_t G_vect;

    while (!done_flag) {
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
                vst1q_u8(sobl_data + (numCols_sobel * row) + col, vreinterpretq_u8_u16(G_vect));
            }

            for (int r_col = col_stop; r_col < numCols_sobel; r_col++) {

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

