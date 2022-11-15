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
 *      - (1.3) Sobel redone to improve speed and lower computations
 *      - (1.2) Sobel redone to process multiple pixels at once
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
#include <sys/time.h>

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

typedef struct int8x8x8_t{
    int8x8_t val[8];
} int8x8x8_t;
typedef struct int16x8x8_t{
    int16x8_t val[8];
} int16x8x8_t;
typedef struct uint16x8x8_t{
    uint16x8_t val[8];
} uint16x8x8_t;

/********** PROTOTYPES **********/
void* toGreyscale(void* threadArgs);
void* sobelFilter(void* threadArgs);

/********** GLOBALS **********/
uint8_t numThreads;
uint16_t numRows, numCols;
pthread_barrier_t barrier_grey, barrier_sobel;
uint8_t done_flag = false;

const uint8x8_t R_WEIGHT_VECT = vdup_n_u8(R_CONST);
const uint8x8_t G_WEIGHT_VECT = vdup_n_u8(G_CONST);
const uint8x8_t B_WEIGHT_VECT = vdup_n_u8(B_CONST);

const int16x8_t GX_KERNEL = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t GY_KERNEL = {1, 2, 1, 0, 0, -1, -2, -1};

/********** MAIN **********/
int main(int argc, char const *argv[])
{
    // Check for valid input
    if (argc != 2) {
        cout << "Invalid input, please try again" << endl;
        return 0;
    }

    // Get the name of the image the user is requesting
    string videoName = argv[1];
    numThreads = 4;

    // Check to make sure the file exists
    ifstream ifile;
    ifile.open(videoName);
    if (!ifile) {
        cout << "The specified file does not exist" << endl;
        return -1;
    }

    // Read the image
    VideoCapture usr_vid(videoName);
    numRows = usr_vid.get(4);
    numCols = usr_vid.get(3);

    // Print video properties
    cout << "Video: " << videoName << endl;
    cout << "Length: " << usr_vid.get(7)/usr_vid.get(5) << " seconds" << endl;
    cout << "Threads: " << (int)numThreads << endl;

    // Define Mat objects used
    Mat frameOrig(numRows, numCols, CV_8UC3), frameGrey(numRows, numCols, CV_8UC1);
    Mat frameSobel(numRows - 2, numCols - 2, CV_8UC1);

    // Create threads
    pthread_t threads_grey[numThreads], threads_sobel[numThreads];

    // Initialize barrier
    pthread_barrier_init(&barrier_grey, NULL, numThreads + 1);
    pthread_barrier_init(&barrier_sobel ,NULL ,numThreads + 1);

    // Create greyscale and Sobel filter arguments for each thread
    thread_data td[numThreads];
    for (uint8_t thread = 0; thread < numThreads; ++thread) {
        td[thread] = {
                .origFrame = &frameOrig,
                .greyFrame = &frameGrey,
                .sobelFrame = &frameSobel,
                .thread_id = thread
        };

        pthread_create(&threads_grey[thread], NULL, &toGreyscale, (void*) &td[thread]);
        pthread_create(&threads_sobel[thread], NULL, &sobelFilter, (void*) &td[thread]);
    }


    struct timeval start, end;
    /* Continue reading and processing frames until reaching the end of
        the video or until the user quits.
    */
    uchar key;
    gettimeofday(&start, 0);
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
    gettimeofday(&end, 0);

    // Set done flag and release all threads from waiting
    done_flag = true;
    pthread_barrier_wait(&barrier_grey);
    pthread_barrier_wait(&barrier_sobel);

    // Destroy threads
    for (uint8_t thread = 0; thread < numThreads; ++thread) {
        pthread_join(threads_grey[thread], NULL);
        pthread_join(threads_sobel[thread], NULL);
    }

    // Destroy barriers
    pthread_barrier_destroy(&barrier_grey);
    pthread_barrier_destroy(&barrier_sobel);

    // Release VideoCapture object
    usr_vid.release();

    // Destroy the window created
    destroyAllWindows();

    // Calculate processing time
    long secs = end.tv_sec - start.tv_sec;
    long usecs = end.tv_usec - start.tv_usec;
    cout << "Processing Time: " << (double) (secs + usecs * 1e-6) << " secs" << endl;

    return 0;
}

/********** USER DEFINED FUNCTIONS **********/

/******************************************************************************
 * Function: toGreyscale
 *
 * Description: This function is meant to be use with Pthread multi-threading
 *              and Arm Neon SIMD vectorization. This function will contiue to
 *              run until the done_flag is set. Converts a 3 channel RGB image
 *              to a 1 channel greyscale image using ITU-R(BT.709) recommended
 *              algorithm.
 *
 *              Grey = (55R + 183G + 18B) >> 8
 *
 * param threadArgs: void*: Pointer to a thread_data
 *
 * return: void*
 *****************************************************************************/
void* toGreyscale(void* threadArgs) {
    thread_data *frames = (thread_data*) threadArgs;
    uchar *origImgData = frames->origFrame->data, *greyImgData = frames->greyFrame->data;
    uint8x8x3_t origPixels;
    uint8x8_t greyPixels;

    // Chunk input frame based on thread_id
    uint32_t pos;
    uint32_t startPos = frames->thread_id * numCols * (numRows/numThreads);
    uint32_t endPos;
    if (frames->thread_id == numThreads - 1) endPos = numRows * numCols - 1;
    else endPos = (frames->thread_id + 1) * numCols * (numRows/numThreads);

    // Number of pixels that cannot be calculated in a group of 8
    uint8_t leftover = (endPos - startPos) % 24;

    // While video is not done, continue to process frames and wait
    while (1) {
        // Wait for dependent processes to end before performing Sobel filtering
        pthread_barrier_wait(&barrier_grey);
        if (done_flag) break;

        // Calculate greyscale pixels 8 at a time, leave leftover pixels to be calculate afterwards
        for (pos = startPos; pos < endPos - leftover; pos += 8) {
            // Read in pixels striding by 3 bytes to get RGB values for 8 pixels
            origPixels = vld3_u8(origImgData + 3 * pos);

            // Apply RGB weighting and add values together
            // greyPixels = vmull_u8(origPixels.val[0], B_WEIGHT_VECT);
            // greyPixels = vmlal_u8(greyPixels, origPixels.val[1], G_WEIGHT_VECT);
            // greyPixels = vmlal_u8(greyPixels, origPixels.val[2], R_WEIGHT_VECT);

            // BSR by 8 then store in greyscale image
            vst1_u8(greyImgData + pos, vshrn_n_u16(
                    vmlal_u8(
                            vmlal_u8(
                                    vmull_u8(origPixels.val[0], B_WEIGHT_VECT),
                                    origPixels.val[1], G_WEIGHT_VECT
                            ),
                            origPixels.val[2], R_WEIGHT_VECT
                    ), 8
            ));
        }
        // Calculate leftover greyscale pixels
        for (; pos < endPos; ++pos) {
            // Apply RGB weights, add values together, and BSR by 8
            greyImgData[pos] = (uchar) ((((origImgData[3*pos + 0] * B_CONST) +
                                          (origImgData[3*pos + 1] * G_CONST) +
                                          (origImgData[3*pos + 2] * R_CONST))) >> 8);
        }
    }
    pthread_exit(NULL);
}

/******************************************************************************
 * Function: sobelFilter
 *
 * Description: This function is meant to be use with Pthread multi-threading
 *              and Arm Neon SIMD vectorization. This funciton will contiue to
 *              run until the done_flag is set. Processes the given 1 channel
 *              greyscale image by applying a Sobel filter over the image. The
 *              size of the image will be reduced by 2 pixels in length and
 *              width.
 *
 *                   |-1  0  1|       | 1  2  1|
 *              gx = |-2  0  2|, gy = | 0  0  0|, G = |gx| + |gy|
 *                   |-1  0  1|       |-1 -2 -1|
 *
 * param threadArgs: void*: Pointer to a thread_data
 *
 * return: void*
 *****************************************************************************/
void* sobelFilter(void* threadArgs) {
    thread_data *frames = (thread_data*) threadArgs;
    uint16x8x8_t pixels;
    int16x8_t pixels_leftover;
    uint8x8_t G_vect;
    uint16_t G;
    uchar *greyImgData = frames->greyFrame->data, *sobelImgData = frames->sobelFrame->data;

    // Chunk input frame based on thread_id
    uint16_t startRow, endRow;
    if (frames->thread_id == 0) startRow = 0;
    else startRow = frames->thread_id*(numRows/numThreads) - 2;
    if (frames->thread_id == numThreads - 1) endRow = numRows - 2;
    else endRow = (frames->thread_id + 1)*(numRows/numThreads) - 1;

    // Number of pixels that cannot be calculated in a group of 8
    uint8_t leftover = (numCols - 2) % 8;

    // While video is not done, continue to process frames and wait
    uint32_t row_0, row_1, row_2;
    int row, col;
    while (1) {
        // Wait for dependent processes to end before performing Sobel filtering
        pthread_barrier_wait(&barrier_sobel);
        if (done_flag) break;

        for (row = startRow + 1; row < endRow; ++row) {
            row_0 = numCols*(row - 1);
            row_1 = numCols*row;
            row_2 = numCols*(row + 1);

            // Calculate Sobel pixels 8 at a time, leave leftover pixels to be calculate afterwards
            for (col = 1; col < numCols - leftover - 1; col += 8) {
                // Read in pixels required to processed 8 consective sobel pixels
                // and change type from uint8x8_t to uint16x8_t
                // Row 1
                pixels.val[0] = vmovl_u8(vld1_u8(greyImgData + row_0 + col - 1));
                pixels.val[1] = vmovl_u8(vld1_u8(greyImgData + row_0 + col));
                pixels.val[2] = vmovl_u8(vld1_u8(greyImgData + row_0 + col + 1));

                // Row 2
                pixels.val[3] = vmovl_u8(vld1_u8(greyImgData + row_1 + col - 1));
                pixels.val[4] = vmovl_u8(vld1_u8(greyImgData + row_1 + col + 1));

                // Row 3
                pixels.val[5] = vmovl_u8(vld1_u8(greyImgData + row_2 + col - 1));
                pixels.val[6] = vmovl_u8(vld1_u8(greyImgData + row_2 + col));
                pixels.val[7] = vmovl_u8(vld1_u8(greyImgData + row_2 + col + 1));

                // Calculate gradient approximation, G = abs(gx) + abs(gy)
                G_vect = vqmovn_u16(                    // G = saturated cast<uchar>(G_u16)
                        vaddq_u16(                      // G_u16 = gx7 + gy7
                                vabsq_s16(                      // gx7 = abs(gx6)
                                        vsubq_s16(                      // gx6 = gx4 - gx5
                                                vaddq_u16(                      // gx4 = gx0 + gx1
                                                        vaddq_u16(                      // gx0 = p2 + p7
                                                                pixels.val[2],
                                                                pixels.val[7]
                                                        ),
                                                        vshlq_n_u16(pixels.val[4], 1)   // gx1 = p4 << 1 (2*p4)
                                                ),
                                                vaddq_u16(                      // gx5 = gx2 + gx3
                                                        vaddq_u16(                      // gx2 = p0 + p5
                                                                pixels.val[0],
                                                                pixels.val[5]
                                                        ),
                                                        vshlq_n_u16(pixels.val[3], 1)   // gx3 = p3 << 1 (2*p3)
                                                )
                                        )
                                ),
                                vabsq_s16(                      // gy7 = abs(gy6)
                                        vsubq_s16(                      // gy6 = gy4 - gy5
                                                vaddq_u16(                      // gy4 = gy0 + gy1
                                                        vaddq_u16(                      // gy0 = p0 + p2
                                                                pixels.val[0],
                                                                pixels.val[2]
                                                        ),
                                                        vshlq_n_u16(pixels.val[1], 1)   // gy1 = p1 << 1 (2*p1)
                                                ),
                                                vaddq_u16(                  // gy5 = py2 + gy3
                                                        vaddq_u16(                  // gy2 = p5 + p7
                                                                pixels.val[5],
                                                                pixels.val[7]
                                                        ),
                                                        vshlq_n_u16(pixels.val[6], 1)   // gy3 = p6 << 1 (2*pi)
                                                )
                                        )
                                )
                        )
                );

                // Store in Sobel image
                vst1_u8(sobelImgData + frames->sobelFrame->cols*(row - 1) + (col - 1), G_vect);
            }

            // Calculate leftover Sobel pixels
            for (; col < numCols - 1; ++col) {
                // Read in pixels required to processed 1 sobel pixel and change
                // type from uint8x8_t to int16x8_t
                pixels_leftover = vreinterpretq_s16_u16(vmovl_u8(
                        (uint8x8_t) {greyImgData[(row_0 + col - 1)], greyImgData[(row_0 + col)], greyImgData[(row_0 + col + 1)],
                                     greyImgData[(row_1 + col - 1)],                             greyImgData[(row_1 + col + 1)],
                                     greyImgData[(row_2 + col - 1)], greyImgData[(row_2 + col)], greyImgData[(row_2 + col + 1)]}
                ));
                // Calculate the gradient approximation, G = abs(gx) + abs(gy)
                G = abs(vaddvq_s16(vmulq_s16(pixels_leftover, GX_KERNEL))) + abs(vaddvq_s16(vmulq_s16(pixels_leftover, GY_KERNEL)));

                // Check that the gradient will not overflow 8-bits
                if (G <= UCHAR_MAX) {
                    sobelImgData[(frames->sobelFrame->cols*(row - 1) + (col - 1))] = (uchar) G;
                } else {
                    sobelImgData[(frames->sobelFrame->cols*(row - 1) + (col - 1))] = (uchar) UCHAR_MAX;
                }
            }
        }
    }
    pthread_exit(NULL);
}