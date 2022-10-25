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
void grayscale_filter(Mat *image, Mat *grayscale);

void sobel_filter(Mat *grayscale_img, Mat *sobel_img, int start, int stop);

void *thread_sobel_filter(void *threadArgs);

/***** Global Variables *****/
pthread_barrier_t barrier;
bool cont_flag = false;

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

    if (!ifile){
        cout << "File does not exist" << endl;
        exit (-1);
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
    Mat frame;
    Mat gray_frame(usr_vid_rows, usr_vid_cols, CV_8UC1);
    Mat sobel_frame(usr_vid_rows - 2, usr_vid_cols - 2, CV_8UC1);

    // init pthreads
    pthread_t threads[NUM_THREADS];
    struct thread_data in_out[NUM_THREADS];
    pthread_barrier_init(&barrier, nullptr, NUM_THREADS + 1);

    // Each thread processes half of the image
    for (int i = 0; i < NUM_THREADS; i++) {

        // init thread variables
        in_out[i].input = &gray_frame;
        in_out[i].output = &sobel_frame;
        in_out[i].start = i * (usr_vid_rows / NUM_THREADS) - 2;
        in_out[i].stop = (i + 1) * (usr_vid_rows / NUM_THREADS);

        if (i == 0) { in_out[i].start = 0; }
        if (i == 3) { in_out[i].stop = usr_vid_rows - 2; }

        // run the threads
        pthread_create(&threads[i], nullptr, &thread_sobel_filter, (void *) &in_out[i]);
    }

    // Loop through the image file
    uchar key;
    while (usr_vid.isOpened()) {
        // Get a frame from the video
        usr_vid >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            cont_flag = true;
            break;
        }

        // Process the image
        grayscale_filter(&frame, &gray_frame);

        pthread_barrier_wait(&barrier);

        // Display the frame
        imshow("sobel", sobel_frame);

        // Hold ESC to exit the video early
        key = waitKey(25);
        if (key == 27) {
            cont_flag = true;
            break;
        }
    }

    pthread_barrier_wait(&barrier);
    for (auto &thread: threads) {
        pthread_join(thread, nullptr);
    }
    pthread_barrier_destroy(&barrier);

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
void grayscale_filter(Mat *image, Mat *grayscale) {
    uchar *image_data = image->data;
    uchar *grayscale_data = grayscale->data;

//    uint16x4_t rgb_const = {0.2126, 0.7156, 0.0722};

    // Apply the ITU-R (BT.709) grayscale algorithm
    for (int pos = 0; pos < image->rows * image->cols; pos++) {
        grayscale_data[pos] = (uchar) (
                (image_data[3 * pos + 0] * B_CONST) +
                (image_data[3 * pos + 1] * G_CONST) +
                (image_data[3 * pos + 2] * R_CONST)
        );
    }
}

/**
 * Takes a grayscale image and applies the sobel operator to the given image. The function will default to single
 * thread operation when threading variables are unspecified.
 * @param grayscale_img An image that has been converted to grayscale
 * @param sobel_img A grayscale image with the sobel operator applied to it
 * @param start For use with threading, indicates the starting point for a given thread
 * @param step For use with threading, sets the number of pixels each thread will skip
 */
void sobel_filter(Mat *grayscale_img, Mat *sobel_img, int start, int stop) {

    // index = [numCols * (row + x) + (col + y)]
    // Convolution = img[index] + (img[index] << 1) + img[index] - img[index] - (img[index] << 1) - img[index]
    // 2 V_ld1 instructions: Vr = fill(row) & Vc = fill(col)
    // 2 V + S instructions: Vx = (Vr + Sr) & Vy = (Vc + Sy)
    // 1 V + V * V instruction:  V_Gx = Vc + Vr * V_numRows
    // These operations are per-pixel, meaning that the CPU does 4 * (6 * 2) + (5 * 2) + (6 * 2) = 70 operations per sobel pixel
    // Without converting the 5 addition operations to vectors, we will still be reducing the total number of
    // operations to (2 + 2 + 1) * 4 + (5 * 2) + (6 * 2) = 42 operations (>50% decrease!)

    // init function variables
    int numCols = grayscale_img->cols;
    int Gx, Gy, G;

    uchar *gray_data = grayscale_img->data;
    uchar *sobel_data = sobel_img->data;

    uint32x4_t row_const, col_const, row_vect, col_vect;
    uint32x4_t numCol_vect = vdupq_n_u32(numCols);
    uint32x4x2_t Gx_vect, Gy_vect;

    uint32x4x4_t Gx_conv_vect = {2, 1, 0, 2,
                                 1, 0, 0, 0,
                                 2, 2, 2, 0,
                                 0, 0, 0, 0};

    uint32x4x4_t Gy_conv_vect = {0, 0, 0, 2,
                                 2, 2, 0, 0,
                                 0, 1, 2, 0,
                                 1, 2, 0, 0};

    while (!cont_flag) {
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
        pthread_barrier_wait(&barrier);
    }
}

/**
 * Converts a thread struct into variables for sobel_filter()
 * @param threadArgs - A struct with variables for the sobel_filter
 */
void *thread_sobel_filter(void *threadArgs) {

    // init thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    int start = thread_data->start;
    int step = thread_data->stop;
    Mat *grayscale_img = thread_data->input;
    Mat *sobel_img = thread_data->output;

    // Run the sobel filter
    sobel_filter(grayscale_img, sobel_img, start, step);

    // pThread return functions
    pthread_exit(nullptr);
}