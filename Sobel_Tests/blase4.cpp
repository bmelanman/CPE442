/******************************************************************************
 * Filename: main.cpp
 * 
 * Desc: This function takes in a .mp4 file, converts it from rgb to grayscale,
 *       then passes the grayscale image through a sobel filter. This filtered
 *       video is then "played"
 * 
 * Author: Blaise Parker 
 * Co-Authors: Bryce Melander, Jonathan Espiritu
 * 
 * Versions:  1.0 (10/3/22) - Built to apply grayscale/sobel to single image
 *            1.1 (10/4/22) - Updated to work with a video/sequence of images
 * ***************************************************************************/


// Include Libraries
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include "lib/pthread_barrier.h"
#include <pthread.h>

// Namespaces being used in this program
using namespace std;
using namespace cv;

//****************** Globals ************************ 
pthread_barrier_t barrier;
//pthread_barrier_t sobel_barrier;
//pthread_barrier_t start_barrier;

bool doneFlag = false;

//****************** Struct Declarations ******************
struct ThreadArg {
    Mat *inputImage;
    Mat *grayImage;
    Mat *sobelImage;
    int sobel_start;
    int sobel_end;
    int gray_start;
    int gray_end;
};

//******************** Function Declarations ****************

void *sobelFilter(void *threadArgs);

void *processFrame(void *threadArgs);

int main(int argc, char const *argv[]) {
    // Check for valid input
    if (argc != 2) {
        cout << "Invalid input, please try again\n";
        return 0;
    }

    // Get the name of the image the user is requesting
    string usr_arg = argv[1];

    // Check to make sure the file exsists
    ifstream ifile;
    ifile.open(usr_arg);
    if (ifile) {

        // Read the image
        VideoCapture usrVid(usr_arg);

        // Ensure a connection is established to the video file. Exit if uncon.
        if (!usrVid.isOpened()) {
            cout << "Error Opening File";
            return -1;
        }

        int numRows = (int) usrVid.get(4);
        int numCols = (int) usrVid.get(3);

        // Initialize Mat to store single frame from video a
        Mat frame(numRows, numCols, CV_8UC3);
        Mat grayscale(numRows, numCols, CV_8UC1);
        Mat sobelImage(numRows - 2, numCols - 2, CV_8UC1);

        //********** Initialize Stuff for pthreads ***********
        int numThreads = 4;
        pthread_t myThreads[numThreads];
        struct ThreadArg sobelArg[numThreads];
        uint8_t barrierCount = (uint8_t) numThreads + 1;

        // One Barrier used multiple times
        pthread_barrier_init(&barrier, nullptr, barrierCount);

        int start;
        int sobel_end;
        int gray_end;

        // Create the threads
        for (int i = 0; i < numThreads; i++) {

            // Ternary Operator for performance
            // Establish Start point for both gray and sobel.
            start = (i == 0) ? 0 : (i * numRows / numThreads) - 2;

            // Establish End points for sobel filter
            if (i == numThreads - 1) {
                sobel_end = ((i + 1) * numRows / numThreads) - 2;
            } else {
                sobel_end = ((i + 1) * numRows / numThreads) - 1;
            }

            // Establish End Points for grayscale filter/image
            if (i == numThreads - 1) {
                gray_end = ((i + 1) * numRows / numThreads) - 1;
            } else {
                gray_end = ((i + 1) * numRows / numThreads);

            }

            // Create P-Threads
            sobelArg[i] = {&frame,
                           &grayscale,
                           &sobelImage,
                           start,
                           sobel_end,
                           start,
                           gray_end};

            pthread_create(&myThreads[i],
                           nullptr,
                           processFrame,
                           (void *) &sobelArg[i]);
        }

        // Main while-loop, runs while video is "playing"
        while (usrVid.isOpened()) {

            usrVid >> frame;                // Grab a single frame from the video

            if (frame.empty()) {
                break;
            }

            pthread_barrier_wait(&barrier);   // Threads wait for in data
            pthread_barrier_wait(&barrier);   // Main waits for grayscale
            pthread_barrier_wait(&barrier);   // Main waits for sobel

            imshow("Sobel Video", sobelImage);  // Display the sobel-ized image

            // Allows user to press ESC key to exit early
            char c = (char) waitKey(1);
            if (c == 27) {
                cout << "Program Exited Successfully \n";
                break;
            }

        }   // end main while loop

        doneFlag = true;   // tells threads to exit/terminate

        pthread_barrier_wait(&barrier); // Allow time for threads to exit loop

        // Destroy Threads
        for (int thread = 0; thread < numThreads; thread++) {
            pthread_join(myThreads[thread], nullptr);
        }

        pthread_barrier_destroy(&barrier);

        // Release video capture object
        usrVid.release();
        destroyAllWindows();

    } else {
        // Error message if filename isn't valid
        cout << "The specified file does not exist\n";
    }

    return 0;
}


void *processFrame(void *threadArgs) {
    auto *threadData = (struct ThreadArg *) threadArgs;

    // Create Pointers to image data
    uchar *pInput = threadData->inputImage->data;
    uchar *pGray = threadData->grayImage->data;
    uchar *pSobel = threadData->sobelImage->data;

    int numCols = threadData->inputImage->cols;

    uchar Blue, Green, Red, gray;
    //int16_t tempGray;
    int Gx, Gy, G;

    int row_g, col_g, row_0, row_1, row_2, col_1, col_2;

    // wait for first frame to arrive from main
    pthread_barrier_wait(&barrier);

    while (!doneFlag) {

        // ************************* Apply Grayscale ******************************
        for (int row = threadData->gray_start; row < threadData->gray_end; row++) {
            row_g = 3 * row * numCols;

            for (int col = 0; col < numCols; col++) {
                col_g = 3 * col;
                // Grab Blue Data
                Blue = pInput[row_g + col_g];

                // Green Data
                Green = pInput[row_g + col_g + 1];

                // Red Data
                Red = pInput[row_g + col_g + 2];

                // Calculate/Create grayscale pixel
                //tempGray = Blue * 19 +   Green * 183 + Red * 54;
                //gray = tempGray >> 8;

                gray = (uchar) (0.0722 * Blue + 0.7152 * Green + 0.2126 * Red);

                // Store grayscale pixel
                pGray[row * numCols + col] = gray;

            }
        }

        // Main waits for grayscale to finish
        pthread_barrier_wait(&barrier);

        // ******************** Apply Sobel Filter *******************************
        for (int row = threadData->sobel_start; row < threadData->sobel_end; row++) {
            row_0 = row * numCols;
            row_1 = (row + 1) * numCols;
            row_2 = (row + 2) * numCols;

            for (int col = 0; col < numCols - 2; col++) {
                col_1 = col + 1;
                col_2 = col + 2;

                Gx = (pGray[row_0 + col_2]) +
                     (pGray[row_1 + col_2] << 1) +
                     (pGray[row_2 + col_2]) -
                     (pGray[row_0 + (col)]) -
                     (pGray[row_1 + (col)] << 1) -
                     (pGray[row_2 + (col)]);

                Gy = (pGray[row_0 + (col)]) +
                     (pGray[row_0 + col_1] << 1) +
                     (pGray[row_0 + col_2]) -
                     (pGray[row_2 + (col)]) -
                     (pGray[row_2 + col_1] << 1) -
                     (pGray[row_2 + col_2]);


                G = abs(Gx) + abs(Gy);

                if (G > 255) {
                    pSobel[row * (numCols - 2) + col] = 255;
                } else {
                    pSobel[row * (numCols - 2) + col] = (uchar) G;
                }

            }
        }

        // Main waits until sobel is finished
        pthread_barrier_wait(&barrier);

        // Threads wait until a new Frame is grabbed
        pthread_barrier_wait(&barrier);
    }

    // Threads are done, exit
    pthread_exit(nullptr);

}

/*
void *getGrayscale(void *threadArgs) {
   
} */

void *sobelFilter(void *threadArgs) {

    auto *sobelArgs = (struct ThreadArg *) threadArgs;

    uchar *pGray = sobelArgs->inputImage->data;
    uchar *pSobel = sobelArgs->sobelImage->data;

    int numCols = sobelArgs->inputImage->cols;

    int Gx, Gy, G;

    while (!doneFlag) {
        for (int row = sobelArgs->sobel_start; row < sobelArgs->sobel_end; row++) {
            for (int col = 0; col < numCols - 2; col++) {


                Gx = (pGray[(row) * numCols + (col) + 2]) +
                     (pGray[(row + 1) * numCols + (col) + 2] << 1) +
                     (pGray[(row + 2) * numCols + (col) + 2]) -
                     (pGray[(row) * numCols + (col)]) -
                     (pGray[(row + 1) * numCols + (col)] << 1) -
                     (pGray[(row + 2) * numCols + (col)]);

                Gy = (pGray[(row) * numCols + (col)]) +
                     (pGray[(row) * numCols + (col) + 1] << 1) +
                     (pGray[(row) * numCols + (col) + 2]) -
                     (pGray[(row + 2) * numCols + (col)]) -
                     (pGray[(row + 2) * numCols + (col) + 1] << 1) -
                     (pGray[(row + 2) * numCols + (col) + 2]);


                G = abs(Gx) + abs(Gy);

                if (G > 255) {
                    pSobel[row * (numCols - 2) + col] = 255;
                } else {
                    pSobel[row * (numCols - 2) + col] = (uchar) G;
                }

            }
        }

        //Wait until main() grabs a new frame
        //r_wait(&barrier);
    }

    pthread_exit(nullptr);
}
