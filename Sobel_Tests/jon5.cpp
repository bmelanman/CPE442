/******************************************************************************
 * File: main.cpp
 *
 * Description: Program will process some given video using the specified
 *              number of threads.
 *
 * Author: Jonathan Espiritu
 * Partners: Bryce Melander, Blaise Parker
 *
 * Revisions:
 *      - (1.1) Original
 *
 *****************************************************************************/

/********** INCLUDES **********/
#include "opencv2/opencv.hpp"
#include "lib/pthread_barrier.h"
#include <fstream>
#include <iostream>
#include <pthread.h>
#include <arm_neon.h>

/********** NAMESPACES **********/
using namespace std;
using namespace cv;

/********** DEFINES **********/
// These constants come from bit shifting the decimal constants left by 8
#define R_CONST 55
#define G_CONST 183
#define B_CONST 18

/********** STRUCTS **********/
typedef struct thread_data {
    Mat *origFrame, *greyFrame, *sobelFrame;
    uint8_t thread_id;
} thread_data;

/********** PROTOTYPES **********/
void *toGreyscale(void *threadArgs);

void *sobelFilter(void *threadArgs);

/********** GLOBALS **********/
uint8_t numThreads;
uint16_t numRows, numCols;
pthread_barrier_t barrier_grey, barrier_sobel;
uint8_t done_flag = false;

const int16x8_t GX_KERNEL = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t GY_KERNEL = {1, 2, 1, 0, 0, -1, -2, -1};

/********** MAIN **********/
int main() {

    // Get the name of the image the user is requesting
    string videoName = "coffee_10.mp4";
    numThreads = 4;

    // Check to make sure the file exists
    ifstream ifile;
    ifile.open(videoName);
    if (!ifile) {
        cout << "Bad" << endl;
        exit(-1);
    }

    // Read the image
    VideoCapture usr_vid(videoName);
    numRows = (uint16_t) usr_vid.get(4);
    numCols = (uint16_t) usr_vid.get(3);

    // Print video properties
    cout << "Video: " << videoName << endl;
    cout << "Length: " << usr_vid.get(7) / usr_vid.get(5) << " seconds" << endl;
    cout << "Threads: " << (int) numThreads << endl;

    // Define Mat objects used
    Mat frameOrig(numRows, numCols, CV_8UC3), frameGrey(numRows, numCols, CV_8UC1);
    Mat frameSobel(numRows - 2, numCols - 2, CV_8UC1);

    // Create threads
    pthread_t threads_grey[numThreads], threads_sobel[numThreads];

    // Initialize barrier
    pthread_barrier_init(&barrier_grey, nullptr, numThreads + 1);
    pthread_barrier_init(&barrier_sobel, nullptr, numThreads + 1);

    // Create greyscale and Sobel filter arguments for each thread
    thread_data td[numThreads];
    for (uint8_t thread = 0; thread < numThreads; ++thread) {
        td[thread] = {
                .origFrame = &frameOrig,
                .greyFrame = &frameGrey,
                .sobelFrame = &frameSobel,
                .thread_id = thread
        };

        pthread_create(&threads_grey[thread], nullptr, &toGreyscale, (void *) &td[thread]);
        pthread_create(&threads_sobel[thread], nullptr, &sobelFilter, (void *) &td[thread]);
    }

    uchar key;
    while (usr_vid.isOpened()) {
        // Pull next frame
        usr_vid >> frameOrig;

        // Check if the video has ended
        if (frameOrig.empty()) {
            break;
        }

        // Wait for all threads to finish their portion of greyscale and main to reach here
        pthread_barrier_wait(&barrier_grey);
        // Wait for all threads to finish their portion of sobel and main to reach here
        pthread_barrier_wait(&barrier_sobel);

        // Display Sobel image
        imshow("Sobel", frameSobel);

        // Check if user quits the video playback
        key = waitKey(1);
        if (key == 'q') {
            cout << "q key was pressed by user. Ending video." << endl;
            break;
        }

    }

    // Set done flag and release all threads from waiting
    done_flag = true;

    pthread_barrier_wait(&barrier_grey);
    pthread_barrier_wait(&barrier_sobel);

    // Destroy threads
    for (uint8_t thread = 0; thread < numThreads; ++thread) {
        pthread_join(threads_grey[thread], nullptr);
        pthread_join(threads_sobel[thread], nullptr);
    }

    // Destroy barriers
    pthread_barrier_destroy(&barrier_grey);
    pthread_barrier_destroy(&barrier_sobel);

    // Release VideoCapture object
    usr_vid.release();

    // Destroy the window created
    destroyAllWindows();

    return 0;
}

/********** USER DEFINED FUNCTIONS **********/

/******************************************************************************
 * Function: toGreyscale
 *
 * Description: This function is meant to be use with Pthread multi-threading,
 *              and will contiue to run until the done_flag is set. Converts a
 *              3 channel RGB image to a 1 channel greyscale image using
 *              ITU-R(BT.709) recommended algorithm.
 *
 *              Grey = 0.2126R + 0.7152G + 0.0722B
 *
 * param threadArgs: void*: Pointer to a thread_grey_data struct
 *
 * return: void*
 *****************************************************************************/
void *toGreyscale(void *threadArgs) {
    auto *frames = (thread_data *) threadArgs;
    uchar *origImgData = frames->origFrame->data, *greyImgData = frames->greyFrame->data;

    uint8x8x3_t origPixels;
    uint16x8_t greyPixels;
    uint8x8_t r_weights = vdup_n_u8(R_CONST);
    uint8x8_t g_weights = vdup_n_u8(G_CONST);
    uint8x8_t b_weights = vdup_n_u8(B_CONST);

    // Chunk input frame based on thread_id
    uint32_t pos;
    uint32_t startPos = frames->thread_id * numCols * (numRows / numThreads);
    uint32_t endPos;
    if (frames->thread_id == numThreads - 1) endPos = numRows * numCols - 1;
    else endPos = (frames->thread_id + 1) * numCols * (numRows / numThreads);

    uint8_t left_over = (endPos - startPos) % 24;
    // uint32_t stopPos = (endPos - left_over) / 8;

    // While video is not done, continue to process frames and wait
    while (true) {
        pthread_barrier_wait(&barrier_grey);

        if (done_flag) break;

        for (pos = startPos; pos < endPos - left_over; pos += 8) {
            origPixels = vld3_u8(origImgData + 3 * pos);

            greyPixels = vmull_u8(origPixels.val[0], b_weights);
            greyPixels = vmlal_u8(greyPixels, origPixels.val[1], g_weights);
            greyPixels = vmlal_u8(greyPixels, origPixels.val[2], r_weights);

            vst1_u8(greyImgData + pos, vshrn_n_u16(greyPixels, 8));
        }
        for (; pos < endPos; ++pos) {
            greyImgData[pos] = (uchar) (
                    ((origImgData[3 * pos + 0] * B_CONST) >> 8) +
                    ((origImgData[3 * pos + 1] * G_CONST) >> 8) +
                    ((origImgData[3 * pos + 2] * R_CONST) >> 8)
            );
        }
    }
    pthread_exit(nullptr);
}

/******************************************************************************
 * Function: sobelFilter
 *
 * Description: This function is meant to be use with Pthread multi-threading,
 *              and will contiue to run until the done_flag is set. Processes
 *              the given 1 channel greyscale image by applying a Sobel filter
 *              over the image. The size of the image will be reduced by 2
 *              pixels in length and width.
 *
 *                   |-1  0  1|       | 1  2  1|
 *              gx = |-2  0  2|, gy = | 0  0  0|, G = |gx| + |gy|
 *                   |-1  0  1|       |-1 -2 -1|
 *
 * param threadArgs: void*: Pointer to a thread_sobel_data struct
 *
 * return: void*
 *****************************************************************************/
void *sobelFilter(void *threadArgs) {
    auto *frames = (thread_data *) threadArgs;
    int16x8_t subarray;
    int16_t G;
    uchar *greyImgData = frames->greyFrame->data, *sobelImgData = frames->sobelFrame->data;

    // Chunk input frame based on thread_id
    uint16_t startRow, endRow;
    if (frames->thread_id == 0) startRow = 0;
    else startRow = frames->thread_id * (numRows / numThreads) - 2;
    if (frames->thread_id == numThreads - 1) endRow = numRows - 2;
    else endRow = (frames->thread_id + 1) * (numRows / numThreads) - 1;

    // While video is not done, continue to process frames and wait
    uint32_t row_0, row_1, row_2;
    while (true) {
        pthread_barrier_wait(&barrier_sobel);

        if (done_flag) break;

        for (int row = startRow + 1; row < endRow; ++row) {
            row_0 = numCols * (row - 1);
            row_1 = numCols * row;
            row_2 = numCols * (row + 1);

            for (int col = 1; col < numCols - 1; ++col) {
                subarray = (int16x8_t) {greyImgData[(row_0 + col - 1)], greyImgData[(row_0 + col)],
                                        greyImgData[(row_0 + col + 1)],
                                        greyImgData[(row_1 + col - 1)], greyImgData[(row_1 + col + 1)],
                                        greyImgData[(row_2 + col - 1)], greyImgData[(row_2 + col)],
                                        greyImgData[(row_2 + col + 1)]};

                // Approximate gradient
                G = (int16_t) (
                        abs(vaddvq_s16(vmulq_s16(subarray, GX_KERNEL))) +
                        abs(vaddvq_s16(vmulq_s16(subarray, GY_KERNEL)))
                );

                // Check that the gradient will not overflow 8-bits
                if (G > 255) { G = 255; }

                sobelImgData[(frames->sobelFrame->cols * (row - 1) + (col - 1))] = (uchar) G;
            }
        }
    }
    pthread_exit(nullptr);
}