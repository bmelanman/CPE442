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
#include <string>
#include <cmath>
#include <pthread.h>
#include <chrono>
#include "../lib/pthread_barrier.h"
#include <arm_neon.h>

/***** Defines *****/
#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722

/***** Namespaces *****/
using namespace std;
using namespace cv;
using namespace chrono;

/***** Structures *****/
struct thread_data {
    Mat *input{};
    Mat *output{};
    int start{};
    int step{};
};

/***** Global Variables *****/
pthread_barrier_t barrier;

/**
 * A grayscale filter for color images, applies the ITU-R (BT.709) grayscale algorithm
 * @param image An image
 * @param grayscale A grayscale image
 */
void grayscale_filter(Mat *image, Mat *grayscale) {
    unsigned char *image_data = image->data;
    unsigned char *grayscale_data = grayscale->data;

    // Apply the ITU-R (BT.709) grayscale algorithm
    for (int pos = 0; pos < image->rows * image->cols; pos++) {
        grayscale_data[pos] = (unsigned char) (
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
void sobel_filter(Mat *grayscale_img, Mat *sobel_img, int start = 0, int step = 1) {

    // index = [nunCols * (row + x) + (col + y)]
    // Convolution = img[index] + (img[index] << 1) + img[index] - img[index] - (img[index] << 1) - img[index]
    // 2 V + S instructions: Vx = (Vx + Sr) & Vy = (Vy + Sy)
    // 1 V * S instruction:  Vx = Vx * Sc
    // 1 V + V instruction:  Vx = Vx + Vy
    // This is done on 6 different pixels, which are then all added/subtracted together
    // These operations are per-pixel, meaning that the CPU does 12 * 4 * 10 = 480 operations per sobel pixel
    // Without converting the 5 addition operations to vectors, we will still be reducing the total number of
    // operations to (2 + 1 + 1) * 10 = 40 operations -> That's 92% fewer operations!

    // init function variables
    int numRows = grayscale_img->rows;
    int numCols = grayscale_img->cols;
    int Gx, Gy, G;
    unsigned char *grayscale_data = grayscale_img->data;
    unsigned char *sobel_data = sobel_img->data;

    uint16x8_t Gx_vect, Gy_vect, row_vect, col_vect;
    uint16x8_t Gx_row_vect = {2, 1, 0, 2, 1, 0};
    uint16x8_t Gx_col_vect = {2, 2, 2, 0, 0, 0};
    uint16x8_t Gy_row_vect = {0, 0, 0, 2, 2, 2};
    uint16x8_t Gy_col_vect = {0, 1, 2, 0, 1, 2};

    // Loop through the rows and cols of the image and apply the sobel filter
    for (int row = start; row < numRows - 2; row += step) {
        for (int col = 0; col < numCols - 2; col++) {

            // Convolve Gx
            row_vect = vaddq_u16(vdupq_n_u16(row), Gx_row_vect);      // row_sum = (row + Gx[i])
            row_vect = vmulq_n_u16(row_vect, numCols);                // row_sum *= numCols
            col_vect = vaddq_u16(vdupq_n_u16(col), Gx_col_vect);      // col_sum = (col + Gx[j])
            Gx_vect = vaddq_u16(row_vect, col_vect);                  // Gx_vect = row_sum + col_sum

            Gx =
                    (grayscale_data[vgetq_lane_u16(Gx_vect, 0)]) +
                    (grayscale_data[vgetq_lane_u16(Gx_vect, 1)] << 1) +
                    (grayscale_data[vgetq_lane_u16(Gx_vect, 2)]) -
                    (grayscale_data[vgetq_lane_u16(Gx_vect, 3)]) -
                    (grayscale_data[vgetq_lane_u16(Gx_vect, 4)] << 1) -
                    (grayscale_data[vgetq_lane_u16(Gx_vect, 5)]);

            // Convolve Gy
            row_vect = vaddq_u16(vdupq_n_u16(row), Gy_row_vect);
            row_vect = vmulq_n_u16(row_vect, numCols);
            col_vect = vaddq_u16(vdupq_n_u16(col), Gy_col_vect);
            Gy_vect = vaddq_u16(row_vect, col_vect);

            Gy =
                    (grayscale_data[vgetq_lane_u16(Gy_vect, 0)]) +
                    (grayscale_data[vgetq_lane_u16(Gy_vect, 1)] << 1) +
                    (grayscale_data[vgetq_lane_u16(Gy_vect, 2)]) -
                    (grayscale_data[vgetq_lane_u16(Gy_vect, 3)]) -
                    (grayscale_data[vgetq_lane_u16(Gy_vect, 4)] << 1) -
                    (grayscale_data[vgetq_lane_u16(Gy_vect, 5)]);

//            Gx = (grayscale_data[(numCols * (row + 2) + (col + 2))]) +
//                 (grayscale_data[(numCols * (row + 1) + (col + 2))] << 1) +
//                 (grayscale_data[(numCols * (row + 0) + (col + 2))]) -
//                 (grayscale_data[(numCols * (row + 2) + (col + 0))]) -
//                 (grayscale_data[(numCols * (row + 1) + (col + 0))] << 1) -
//                 (grayscale_data[(numCols * (row + 0) + (col + 0))]);
//
//            Gy = (grayscale_data[(numCols * (row + 0) + (col + 0))]) +
//                 (grayscale_data[(numCols * (row + 0) + (col + 1))] << 1) +
//                 (grayscale_data[(numCols * (row + 0) + (col + 2))]) -
//                 (grayscale_data[(numCols * (row + 2) + (col + 0))]) -
//                 (grayscale_data[(numCols * (row + 2) + (col + 1))] << 1) -
//                 (grayscale_data[(numCols * (row + 2) + (col + 2))]);


            // Gradient approximation
            G = abs(Gx) + abs(Gy);

            // Overflow check
            if (G > 255) { G = 255; }

            sobel_data[(sobel_img->cols * (row) + (col))] = G;

        }
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
    int step = thread_data->step;
    Mat *grayscale_img = thread_data->input;
    Mat *sobel_img = thread_data->output;

    // Run the sobel filter
    sobel_filter(grayscale_img, sobel_img, start, step);

    // pThread return functions
    pthread_barrier_wait(&barrier);
    pthread_exit(nullptr);
}

/**
 * Takes a video file and applies the sobel operator to it frame by frame.
 * This operation is threaded, so a number of threads may be specified if desired.
 * The default number of threads is 1
 * @param video_file The name of a video file
 * @param num_threads A number of threads
 */
void video_processor(const string &usr_vid, int num_threads = 1) {

    // Make sure num_threads is 1 or more
    if (num_threads < 1) {
        cout << "Invalid number of threads, num_threads has been set to 1" << endl;
        num_threads = 1;
    }

    // init pthreads
    pthread_t threads[num_threads];
    struct thread_data in_out[num_threads];
    pthread_barrier_init(&barrier, nullptr, num_threads);

    // Read the video
    VideoCapture video_file(usr_vid);
    int usr_vid_rows = (int) video_file.get(4);
    int usr_vid_cols = (int) video_file.get(3);

    // init image Mats
    Mat frame;
    Mat gray_frame(usr_vid_rows, usr_vid_cols, CV_8UC1);
    Mat sobel_frame(usr_vid_rows - 2, usr_vid_cols - 2, CV_8UC1);

    // Loop through the image file
    while (video_file.isOpened()) {
        // Get a frame from the video
        video_file >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            break;
        }

        // Process the image
        grayscale_filter(&frame, &gray_frame);

        // Each thread processes half of the image
        for (int i = 0; i < num_threads; i++) {

            // init thread variables
            in_out[i].input = &gray_frame;
            in_out[i].output = &sobel_frame;
            in_out[i].step = num_threads;
            in_out[i].start = i;

            // run the threads
            pthread_create(&threads[i], nullptr, &thread_sobel_filter, (void *) &in_out[i]);
        }

        // Display the frame
        imshow(usr_vid, sobel_frame);

        // Hold ESC to exit the video early
        if ((char) waitKey(25) == 27) break;
    }

    // Clean up
    video_file.release();
    destroyAllWindows();
    pthread_barrier_destroy(&barrier);
}

/**
 * Tracks the time it takes to run the video processor functions
 * @param usr_vid
 * @param num_threads
 */
float time_function(const string &usr_vid, int num_threads = 4) {

    // Start the timer
    auto start = high_resolution_clock::now();

    // Process the video
    video_processor(usr_vid, num_threads);

    // Stop the timer and calculate duration
    auto stop = high_resolution_clock::now();
    return (float) (duration_cast<microseconds>(stop - start)).count() / 1000000;
}

int main(int argc, char const *argv[]) {

    // Check for valid input
    if (argc != 2 && argc != 3) {
        cout << "Invalid input, please try again\n";
        exit(-1);
    }

    // Get the file the user is requesting
    string usr_arg = argv[1];
    ifstream ifile;
    ifile.open(usr_arg);

    // Check to make sure the file exists, quit if it does not
    if (!ifile) {
        cout << "The specified file does not exist";
        exit(-1);
    } else if (usr_arg.substr(usr_arg.size() - 4) != ".mp4") {
        cout << "This file is not supported";
        exit(-1);
    }

    // Print how long the video is
    VideoCapture usr_vid(usr_arg);
    int fps = (int) usr_vid.get(CAP_PROP_FPS);
    int frame_count = int(usr_vid.get(CAP_PROP_FRAME_COUNT));
    usr_vid.release();

    cout << "Video length in seconds: " << (frame_count / fps) << "." << ((frame_count / fps) % 1) << endl;

    // Check user options
    if (argc == 3 && argv[2][0] == 'a') {

        cout << "Thread count set to default: 4 threads" << endl;

        float time;
        float avg = 0;

        for (int i = 0; i < 10; i++) {
            time = time_function(usr_arg, 4);
            cout << "Processing time: " << time << endl;
            avg += time;
        }
        cout << "Average time: " << (avg / 10) << endl;

    } else if (argc == 3 && isdigit(argv[2][0])) {

        cout << "Processing time: " << time_function(usr_arg, (int) argv[2][0]) << endl;

    } else {

        cout << "Thread count set to default: 4 threads" << endl;
        cout << "Processing time: " << time_function(usr_arg) << endl;
    }
    return 0;
}
