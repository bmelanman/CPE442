/*********************************************************
* File: main.cpp
*
* Description: This program converts images and videos from
*               full color RGB to grayscale and then applies
*               a Sobel filter before displaying the final
*               image.
*
* Author: Bryce Melander
* Co-Authors: Blase Parker, Johnathan Espiritu
*
* Revisions: V1.2
*
**********************************************************/

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <pthread.h>

#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722

// Namespace nullifies the use of std::fucntion() and cv::function();
using namespace std;
using namespace cv;

struct thread_data {
    Mat* input;
    Mat* output;
    int start;
    int step;
};

void grayscale_filter(Mat* image, Mat* grayscale) {
    uchar* image_data = image->data;
    uchar* grayscale_data = grayscale->data;

    for (int pos = 0; pos < image->rows * image->cols; pos++) {
        grayscale_data[pos] = (
            (image_data[3 * pos + 0] * B_CONST) +
            (image_data[3 * pos + 1] * G_CONST) +
            (image_data[3 * pos + 2] * R_CONST)
            );
    }

}

void* threaded_sobel(void* threadArgs) {

    // init thread variables
    struct thread_data* thread_data = (struct thread_data*)threadArgs;
    Mat* grayscale_img = thread_data->input;
    Mat* sobel_img = thread_data->output;
    int start = thread_data->start;
    int step = thread_data->step;

    // init function varaibles 
    int numRows = grayscale_img->rows;
    int numCols = grayscale_img->cols;
    int16_t Gx, Gy, G;
    uchar* grayscale_data = grayscale_img->data;
    uchar* sobel_data = sobel_img->data;

    // Make sure start and step aren't big enough to get out of bounds
    if (start > numRows - 2 || step > numRows) {
        cout << "Invalid threaded_sobel() configuration";
        exit(-1);
    }

    // Loop through the rows and cols of the image and apply the sobel filter
    for (int row = start; row < (numRows - 2 - step); row += step) {
        for (int col = 0; col < numCols - 2; col++) {

            // Convolve Gx
            Gx =
                (grayscale_data[(numCols * (row + 2) + (col + 2))]) +
                (grayscale_data[(numCols * (row + 1) + (col + 2))] << 1) +
                (grayscale_data[(numCols * (row + 0) + (col + 2))]) -
                (grayscale_data[(numCols * (row + 2) + (col + 0))]) -
                (grayscale_data[(numCols * (row + 1) + (col + 0))] << 1) -
                (grayscale_data[(numCols * (row + 0) + (col + 0))]);

            // Convolve Gy
            Gy =
                (grayscale_data[(numCols * (row + 0) + (col + 0))]) +
                (grayscale_data[(numCols * (row + 0) + (col + 1))] << 1) +
                (grayscale_data[(numCols * (row + 0) + (col + 2))]) -
                (grayscale_data[(numCols * (row + 2) + (col + 0))]) -
                (grayscale_data[(numCols * (row + 2) + (col + 1))] << 1) -
                (grayscale_data[(numCols * (row + 2) + (col + 2))]);

            // Gradient approximation
            G = abs(Gx) + abs(Gy);

            // Overflow check
            if (G > 255) {
                sobel_data[(sobel_img->cols * (row - 1) + (col - 1))] = 255;
            }
            else {
                sobel_data[(sobel_img->cols * (row - 1) + (col - 1))] = (uchar)G;
            }
        }
    }

    pthread_exit(NULL);
}

int main(int argc, char const* argv[]) {

    // Check for valid input
    if (argc != 2) {
        cout << "Invalid input, please try again\n";
        exit(-1);
    }

    // Get the name of the file the user is requesting
    string usr_arg = argv[1];

    // Check to make sure the file exsists, quit if it does not
    ifstream ifile;
    ifile.open(usr_arg);
    if (!ifile) {
        cout << "The specified file does not exist";
        exit(-1);
    }

    if (usr_arg.substr(usr_arg.size() - 4) == ".mp4") {
        // Read the video
        VideoCapture usr_vid = VideoCapture(usr_arg);
        int usr_vid_rows = usr_vid.get(4);
        int usr_vid_cols = usr_vid.get(3);

        // init the Mats for each image
        Mat frame;
        Mat gray_frame(usr_vid_rows, usr_vid_cols, CV_8UC1);
        Mat sobel_frame(usr_vid_rows - 2, usr_vid_cols - 2, CV_8UC1);

        // init pThreads
        pthread_t thread1, thread2;

        // init data used by threads
        struct thread_data in_out1;
        struct thread_data in_out2;
        in_out1.input = in_out2.input = &gray_frame;
        in_out1.output = in_out2.input = &sobel_frame;
        in_out1.step = in_out2.step = 2;
        in_out1.start = 0;
        in_out2.start = 1;

        // Loop through the image file
        while (1) {
            // Get a frame from the video
            usr_vid >> frame;

            // If we're all out of frames, the video is over
            if (frame.empty()) {
                break;
            }

            // Process the image
            grayscale_filter(&frame, &gray_frame);

            // Each thread processes half of the image
            pthread_create(&thread1, NULL, threaded_sobel, (void*)&in_out1);    // Even pixels
            pthread_create(&thread2, NULL, threaded_sobel, (void*)&in_out2);    // Odd pixels

            // threaded_sobel(&gray_frame, &sobel_frame, 0, 2);
            // threaded_sobel(&gray_frame, &sobel_frame, 1, 2);

            // Wait for the threads to finish before displaying 
            pthread_join(thread1, NULL);
            pthread_join(thread2, NULL);

            // Dislplay the frame
            imshow(usr_arg, sobel_frame);

            // Hold ESC to exit the video early
            char c = (char)waitKey(25);
            if (c == 27) {
                break;
            }
        }

        // Clean up
        usr_vid.release();
        destroyAllWindows();

        return 0;
    }
    else {
        cout << "This file is not supported";
        exit(-1);
    }

    cout << "Breakout!\n";
    return -1;
}
