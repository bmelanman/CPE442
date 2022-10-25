/******************************************************************************
 * File: main.cpp
 * 
 * Description: This program will process the given video and display the video
 *              after applying both a greyscale and Sobel filter.
 * 
 * Author: Jonathan Espiritu
 * Partners: Bryce Melander, Blaise Parker
 * 
 * Revisions: 1.1
 * 
 *****************************************************************************/

/********** INCLUDES **********/
#include "opencv2/opencv.hpp"
#include "../lib/pthread_barrier.h"
#include <fstream>
#include <iostream>
#include <pthread.h>

/********** NAMESPACES **********/
using namespace std;
using namespace cv;

/********** DEFINES **********/
#define R_CONST 0.2126
#define G_CONST 0.7152
#define B_CONST 0.0722

/********** STRUCTS **********/
struct thread_grey_data {
    Mat *inputFrame;
    Mat *outputFrame;
    uint32_t startPos;
    uint32_t endPos;
    uint16_t numRows;
    uint16_t numCols;
    uint8_t thread_id;
};

//struct thread_sobel_data {
//    Mat* inputFrame;
//    Mat* outputFrame;
//    uint16_t startRow;
//    uint16_t endRow;
//    uint16_t numRows;
//    uint16_t numCols;
//    uint8_t thread_id;
//};

/********** PROTOTYPES **********/
void toGreyscale(Mat *origImg, Mat *greyImg);

void *toGreyscale(void *threadArgs);

void sobelFilter(Mat *greyImg, Mat *sobelImg);

void *sobelFilter(void *threadArgs);

/********** GLOBALS **********/
uint8_t numThreads;
pthread_barrier_t barrier_grey;
pthread_barrier_t barrier_sobel;
uint8_t done_flag = false;

/********** MAIN **********/
int main(int argc, char const *argv[]) {
    // Check for valid input
    if (argc != 3) {
        cout << "Invalid input, please try again\n";
        return 0;
    }

    // Get the name of the image the user is requesting
    string videoName = argv[1];
    numThreads = strtol(argv[2], nullptr, 10);

    // Check to make sure the file exists
    ifstream ifile;
    ifile.open(videoName);
    if (ifile) {
        // Read the image
        VideoCapture usr_vid(videoName);
        uint16_t numRows = usr_vid.get(4);
        uint16_t numCols = usr_vid.get(3);

        // Define Mat objects used
        Mat frameOrig(numRows, numCols, CV_8UC1);
        Mat frameGrey(numRows, numCols, CV_8UC1);
        Mat frameSobel(numRows - 2, numCols - 2, CV_8UC1);

        // Create threads
        pthread_t threads_grey[numThreads];

        // Initialize barrier
        pthread_barrier_init(&barrier_grey, nullptr, numThreads + 1);

        // Create greyscale and Sobel filter arguments for each thread
        struct thread_grey_data td_orig2grey[numThreads];

        for (uint8_t thread = 0; thread < numThreads; ++thread) {
            td_orig2grey[thread].inputFrame = &frameOrig;
            td_orig2grey[thread].outputFrame = &frameGrey;
            td_orig2grey[thread].startPos = thread * numCols * (numRows / numThreads);
            td_orig2grey[thread].endPos = (thread + 1) * numCols * (numRows / numThreads) - 1;
            td_orig2grey[thread].numRows = numRows;
            td_orig2grey[thread].numCols = numCols;
            td_orig2grey[thread].thread_id = thread;

            pthread_create(&threads_grey[thread], nullptr, &toGreyscale, (void *) &td_orig2grey[thread]);

        }

        uchar key;
        while (usr_vid.isOpened()) {
            // Pull next frame
            usr_vid >> frameOrig;

            // Check if the video has ended
            if (frameOrig.empty()) {
                done_flag = true;
                break;
            }

            pthread_barrier_wait(&barrier_grey);
            // pthread_barrier_wait(&barrier_sobel);

            // // Run greyscale filter on a portion of the original image using each thread
            // for (uint8_t thread = 0; thread < numThreads; ++ thread) {
            //     pthread_create(&threads[thread], nullptr, &toGreyscale, (void*) &td_orig2grey[thread]);
            // }

            // Display Sobel image
            imshow("Sobel", frameGrey);

            // Check if user quits the video playback
            key = waitKey(10);
            if (key == 'q') {
                cout << "q key was pressed by user. Ending video." << endl;
                done_flag = true;
                break;
            }

        }

        pthread_barrier_wait(&barrier_grey);
        // pthread_barrier_wait(&barrier_sobel);

        // Destroy threads
        for (uint8_t thread = 0; thread < numThreads; ++thread) {
            pthread_join(threads_grey[thread], nullptr);
            // pthread_join(threads_sobel[thread], nullptr);
        }

        // Destroy barrier
        pthread_barrier_destroy(&barrier_grey);
        // pthread_barrier_destroy(&barrier_sobel);

        // Release VideoCapture object
        usr_vid.release();

        // Destroy the window created
        destroyAllWindows();
    } else {
        cout << "The specified file does not exist";
    }

    return 0;
}

/********** USER DEFINED FUNCTIONS **********/

/******************************************************************************
 * Function: toGreyscale
 *
 * Description: Converts a 3 channel RGB image to a 1 channel greyscale image
 *              using ITU-R(BT.709) recommended algorithm.
 *
 *              Grey = 0.2126R + 0.7152G + 0.0722B
 *
 * param origImg:   Mat*: Pointer to 3 channel RGB image Mat
 * param greyImg:   Mat*: Pointer to 1 channel Mat to save modified image
 *
 * return: void
 *****************************************************************************/
//void toGreyscale(Mat *origImg, Mat *greyImg) {
//    uchar *origImgData = origImg->data;
//    uchar *greyImgData = greyImg->data;
//
//    for (int pos = 0; pos < origImg->rows * origImg->cols; ++pos) {
//        greyImgData[pos] = (uchar) ((origImgData[3 * pos + 0] * B_CONST) +
//                                    (origImgData[3 * pos + 1] * G_CONST) +
//                                    (origImgData[3 * pos + 2] * R_CONST));
//    }
//}

void *toGreyscale(void *threadArgs) {
    struct thread_grey_data *frames = (struct thread_grey_data *) threadArgs;
    uchar *origImgData = frames->inputFrame->data;
    uchar *greyImgData = frames->outputFrame->data;

    if (frames->thread_id == numThreads - 1) frames->endPos = frames->numRows * frames->numCols - 1;

    while (!done_flag) {
        for (int pos = frames->startPos; pos < frames->endPos; ++pos) {
            greyImgData[pos] = (uchar) (origImgData[3 * pos + 0] * B_CONST) +
                               (uchar) (origImgData[3 * pos + 1] * G_CONST) +
                               (uchar) (origImgData[3 * pos + 2] * R_CONST);
        }
        pthread_barrier_wait(&barrier_grey);
    }
    pthread_exit(nullptr);
}

/******************************************************************************
 * Function: sobelFilter
 *
 * Description: Processes the given 1 channel greyscale image by applying a
 *              Sobel filter over the image. The size of the image will be
 *              reduced by 2 pixels in length and width.
 *
 *                   |-1  0  1|       | 1  2  1|
 *              gx = |-2  0  2|, gy = | 0  0  0|, G = |gx| + |gy|
 *                   |-1  0  1|       |-1 -2 -1|
 *
 * param greyImg:   Mat*: Pointer to 1 channel greyscale image Mat
 * param sobelImg:  Mat*: Pointer to 1 channel Mat to save modified image
 *
 * return: void
 *****************************************************************************/
//void sobelFilter(Mat* greyImg, Mat* sobelImg) {
//    int numRows = greyImg->rows;
//    int numCols = greyImg->cols;
//    int gx, gy, G;
//    uchar* greyImgData = greyImg->data;
//    uchar* sobelImgData = sobelImg->data;
//
//    for (int row = 1; row < numRows - 1; ++row) {
//        for (int col = 1; col < numCols - 1; ++col) {
//            // Convolve gx with surrounding pixels
//            gx = (greyImgData[(numCols*(row + 1) + (col + 1))]) +
//                 (greyImgData[(numCols*row + (col + 1))] << 1) +
//                 (greyImgData[(numCols*(row - 1) + (col + 1))]) -
//                 (greyImgData[(numCols*(row + 1) + (col - 1))]) -
//                 (greyImgData[(numCols*row + (col - 1))] << 1) -
//                 (greyImgData[(numCols*(row - 1) + (col - 1))]);
//
//            // Convolve gy with surrounding pixels
//            gy = (greyImgData[(numCols*(row - 1) + (col - 1))]) +
//                 (greyImgData[(numCols*(row - 1) + col)] << 1) +
//                 (greyImgData[(numCols*(row - 1) + (col + 1))]) -
//                 (greyImgData[(numCols*(row + 1) + (col - 1))]) -
//                 (greyImgData[(numCols*(row + 1) + col)] << 1) -
//                 (greyImgData[(numCols*(row + 1) + (col + 1))]);
//
//            // Approximate gradient
//            G = abs(gx) + abs(gy);
//
//            // Check that the gradient will not overflow 8-bits
//            if (G <= 255) {
//                sobelImgData[(sobelImg->cols*(row - 1) + (col - 1))] = (uchar)G;
//            } else {
//                sobelImgData[(sobelImg->cols*(row - 1) + (col - 1))] = 255;
//            }
//        }
//    }
//}

/******************************************************************************
 * Function: sobelFilter
 *
 * Description: This function is meant to be use with Pthread multi-threading.
 *              Processes the given 1 channel greyscale image by applying a
 *              Sobel filter over the image. The size of the image will be
 *              reduced by 2 pixels in length and width.
 *
 *                   |-1  0  1|       | 1  2  1|
 *              gx = |-2  0  2|, gy = | 0  0  0|, G = |gx| + |gy|
 *                   |-1  0  1|       |-1 -2 -1|
 *
 * param threadArgs: void*: Pointer to a thread_data struct
 *
 * return: void*
 *****************************************************************************/
//void* sobelFilter(void* threadArgs) {
//    struct thread_sobel_data *frames = (struct thread_sobel_data*) threadArgs;
//    short gx, gy, G;
//    uchar* greyImgData = frames->inputFrame->data;
//    uchar* sobelImgData = frames->outputFrame->data;
//
//    if (frames->thread_id == 0) frames->startRow = 0;
//    if (frames->thread_id == numThreads - 1) frames->endRow = frames->numRows - 2;
//
//    while (!done_flag) {
//        for (int row = frames->startRow + 1; row < frames->endRow; ++row) {
//            for (int col = 1; col < frames->numCols - 1; ++col) {
//                // Convolve gx with surrounding pixels
//                gx = (greyImgData[(frames->numCols*(row + 1) + (col + 1))]) +
//                     (greyImgData[(frames->numCols*row + (col + 1))] << 1) +
//                     (greyImgData[(frames->numCols*(row - 1) + (col + 1))]) -
//                     (greyImgData[(frames->numCols*(row + 1) + (col - 1))]) -
//                     (greyImgData[(frames->numCols*row + (col - 1))] << 1) -
//                     (greyImgData[(frames->numCols*(row - 1) + (col - 1))]);
//
//                // Convolve gy with surrounding pixels
//                gy = (greyImgData[(frames->numCols*(row - 1) + (col - 1))]) +
//                     (greyImgData[(frames->numCols*(row - 1) + col)] << 1) +
//                     (greyImgData[(frames->numCols*(row - 1) + (col + 1))]) -
//                     (greyImgData[(frames->numCols*(row + 1) + (col - 1))]) -
//                     (greyImgData[(frames->numCols*(row + 1) + col)] << 1) -
//                     (greyImgData[(frames->numCols*(row + 1) + (col + 1))]);
//
//                // Approximate gradient
//                G = abs(gx) + abs(gy);
//
//                // Check that the gradient will not overflow 8-bits
//                if (G <= 255) {
//                    sobelImgData[(frames->outputFrame->cols*(row - 1) + (col - 1))] = (uchar)G;
//                } else {
//                    sobelImgData[(frames->outputFrame->cols*(row - 1) + (col - 1))] = 255;
//                }
//            }
//        }
//        pthread_barrier_wait(&barrier_sobel);
//    }
//
//    pthread_exit(nullptr);
//}