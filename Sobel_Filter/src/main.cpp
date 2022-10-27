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
* Revisions: V3.1
**********************************************************/

/***** Includes *****/
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include "../lib/pthread_barrier.h"
#include <arm_neon.h>

/***** Defines *****/
#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

/***** Namespaces *****/
using namespace std;
using namespace cv;

/***** Structures *****/
struct thread_data {
    Mat *input{};
    Mat *output{};
    int start{};
    int stop{};
};

/***** Prototypes *****/
void *thread_grayscale_filter(void *threadArgs);

void *thread_sobel_filter(void *threadArgs);

/***** Global Variables *****/
pthread_barrier_t gray_barrier;
pthread_barrier_t sobel_barrier;
bool done_flag = false;

/***** Main *****/
int main(int argc, char const *argv[]) {

    if (argc != 2) {
        cout << "Invalid input" << endl;
        exit(-1);
    }

    // Get the file the user is requesting
    string usr_arg = argv[1];
    ifstream ifile;
    ifile.open(usr_arg);

    if (!ifile) {
        cout << "File does not exist" << endl;
        exit(-1);
    }

    // Print how long the video is
    VideoCapture usr_vid(usr_arg);
    int fps = (int) usr_vid.get(CAP_PROP_FPS);
    int frame_count = int(usr_vid.get(CAP_PROP_FRAME_COUNT));
    cout << "Video length in seconds: " << (frame_count / fps) << "." << ((frame_count / fps) % 1) << endl;
    cout << "Thread count set to default: 4 threads" << endl;

    // Read the video
    int usr_vid_rows = (int) usr_vid.get(4);
    int usr_vid_cols = (int) usr_vid.get(3);

    // init image Mats
    Mat frame(usr_vid_rows, usr_vid_cols, CV_8UC3);
    Mat gray_frame(usr_vid_rows, usr_vid_cols, CV_8UC1);
    Mat sobel_frame(usr_vid_rows - 2, usr_vid_cols - 2, CV_8UC1);

    // init pthreads
    pthread_t gray_threads[NUM_THREADS];
    pthread_t sobel_threads[NUM_THREADS];

    // init barriers
    pthread_barrier_init(&gray_barrier, nullptr, NUM_THREADS + 1);
    pthread_barrier_init(&sobel_barrier, nullptr, NUM_THREADS + 1);

    struct thread_data gray_data[NUM_THREADS], sobel_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {

        // init thread variables
        gray_data[i].input = &frame;
        gray_data[i].output = &gray_frame;
        gray_data[i].start = i * usr_vid_cols * (usr_vid_rows / NUM_THREADS);

        if (i == NUM_THREADS - 1) { gray_data[i].stop = usr_vid_rows * usr_vid_cols - 1; }
        else { gray_data[i].stop = (i + 1) * usr_vid_cols * (usr_vid_rows / NUM_THREADS); }

        // run the threads
        pthread_create(&gray_threads[i], nullptr, &thread_grayscale_filter, (void *) &gray_data[i]);

        sobel_data[i].input = &gray_frame;
        sobel_data[i].output = &sobel_frame;

        if (i == 0) { sobel_data[i].start = 0; }
        else { gray_data[i].start = i * (usr_vid_rows / NUM_THREADS) - 2; }

        if (i == NUM_THREADS - 1) { sobel_data[i].stop = usr_vid_rows - 2; }
        else { gray_data[i].stop = (i + 1) * (usr_vid_rows / NUM_THREADS); }

        pthread_create(&sobel_threads[i], nullptr, &thread_sobel_filter, (void *) &sobel_data[i]);
    }

    // Loop through the image file
    while (usr_vid.isOpened()) {
        // Get a frame from the video
        usr_vid >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            done_flag = true;
            break;
        }

        // Process the image
        pthread_barrier_wait(&gray_barrier);
        pthread_barrier_wait(&sobel_barrier);

        // Display the frame
        imshow("sobel", sobel_frame);

        // Hold ESC to exit the video early
        if ((char) waitKey(5) == 27) break;
    }

    pthread_barrier_wait(&gray_barrier);
    pthread_barrier_wait(&sobel_barrier);

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(sobel_threads[i], nullptr);
        pthread_join(gray_threads[i], nullptr);
    }

    pthread_barrier_destroy(&gray_barrier);
    pthread_barrier_destroy(&sobel_barrier);

    // Clean up
    usr_vid.release();
    destroyAllWindows();

    return 0;
}

/***** Project Functions *****/

/**
 * A grayscale filter for color images, applies the ITU-R (BT.709) grayscale algorithm
 * @param image An image
 * @param grayscale A grayscale image
 */
void *thread_grayscale_filter(void *threadArgs) {

    // init thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    uchar *usr_img = thread_data->input->data;
    uchar *gray_img = thread_data->output->data;
    int start = thread_data->start;
    int stop = thread_data->stop;

    while (!done_flag) {
        for (int pos = start; pos < stop; pos++) {
            gray_img[pos] = (uchar) ((usr_img[3 * pos + 0] * B_CONST) +
                                     (usr_img[3 * pos + 1] * G_CONST) +
                                     (usr_img[3 * pos + 2] * R_CONST));
        }
        pthread_barrier_wait(&gray_barrier);
    }
    pthread_exit(nullptr);
}

/**
 * Takes a grayscale image and applies the sobel operator to the given image. The function will default to single
 * thread operation when threading variables are unspecified.
 * @param threadArgs - A struct with variables for the sobel_filter
 */
void *thread_sobel_filter(void *threadArgs) {

    // init thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    int start = thread_data->start;
    int stop = thread_data->stop;
    Mat *grayscale_img = thread_data->input;
    Mat *sobel_img = thread_data->output;

    // init function variables
    uchar *gray_data = grayscale_img->data;
    uchar *sobel_data = sobel_img->data;
    int numCols = grayscale_img->cols;
    int Gx, Gy, G;

    // init NEON vectors
    uint32x4_t row_const, col_const, row_vect, col_vect;
    uint32x4_t numCol_vect = vdupq_n_u32(numCols);
    uint32x4x2_t Gx_vect, Gy_vect;

    uint32x4x4_t Gx_conv_vect = {2, 1, 0, 2, 1, 0, 0, 0,
                                 2, 2, 2, 0, 0, 0, 0, 0};

    uint32x4x4_t Gy_conv_vect = {0, 0, 0, 2, 2, 2, 0, 0,
                                 0, 1, 2, 0, 1, 2, 0, 0};

    while (!done_flag) {

        // Loop through the rows and cols of the image and apply the sobel filter
        for (int row = start; row < stop - 2; row++) {

            // Put the row into a vector
            row_const = vdupq_n_u32(row);

            for (int col = 0; col < numCols - 2; col++) {

                // Put the col into a vector
                col_const = vdupq_n_u32(col);

                // Convolve Gx
                // Calculate [numCols * (row + x) + (col + y)] for bits 0 - 3
                row_vect = vaddq_u32(row_const, Gx_conv_vect.val[0]);                // Vr = (row + i)
                col_vect = vaddq_u32(col_const, Gx_conv_vect.val[2]);                // Vc = (col + j)
                Gx_vect.val[0] = vmlaq_u32(col_vect, row_vect, numCol_vect);     // i = Vc + Vr * V_numCols

                // Calculate bits 4 - 5
                row_vect = vaddq_u32(row_const, Gx_conv_vect.val[1]);
                col_vect = vaddq_u32(col_const, Gx_conv_vect.val[3]);
                Gx_vect.val[1] = vmlaq_u32(col_vect, row_vect, numCol_vect);

                // Convolve Gy
                row_vect = vaddq_u32(row_const, Gy_conv_vect.val[0]);
                col_vect = vaddq_u32(col_const, Gy_conv_vect.val[2]);
                Gy_vect.val[0] = vmlaq_u32(col_vect, row_vect, numCol_vect);

                row_vect = vaddq_u32(row_const, Gy_conv_vect.val[1]);
                col_vect = vaddq_u32(col_const, Gy_conv_vect.val[3]);
                Gy_vect.val[1] = vmlaq_u32(col_vect, row_vect, numCol_vect);

                Gx = (gray_data[vgetq_lane_u32(Gx_vect.val[0], 0)]) +
                     (gray_data[vgetq_lane_u32(Gx_vect.val[0], 1)] << 1) +
                     (gray_data[vgetq_lane_u32(Gx_vect.val[0], 2)]) -
                     (gray_data[vgetq_lane_u32(Gx_vect.val[0], 3)]) -
                     (gray_data[vgetq_lane_u32(Gx_vect.val[1], 0)] << 1) -
                     (gray_data[vgetq_lane_u32(Gx_vect.val[1], 1)]);

                Gy = (gray_data[vgetq_lane_u32(Gy_vect.val[0], 0)]) +
                     (gray_data[vgetq_lane_u32(Gy_vect.val[0], 1)] << 1) +
                     (gray_data[vgetq_lane_u32(Gy_vect.val[0], 2)]) -
                     (gray_data[vgetq_lane_u32(Gy_vect.val[0], 3)]) -
                     (gray_data[vgetq_lane_u32(Gy_vect.val[1], 0)] << 1) -
                     (gray_data[vgetq_lane_u32(Gy_vect.val[1], 1)]);

                // Gradient approximation
                G = abs(Gx) + abs(Gy);

                // Overflow check
                if (G > 255) { G = 255; }

                // Write the pixel to the sobel image
                sobel_data[sobel_img->cols * (row) + (col)] = G;
            }
        }
        pthread_barrier_wait(&gray_barrier);
        pthread_barrier_wait(&sobel_barrier);
    }

    // pthread return function
    pthread_exit(nullptr);
}