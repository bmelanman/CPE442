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

/***** Prototypes *****/
void grayscale_filter(Mat *image, Mat *grayscale);
void sobel_filter(Mat *grayscale_img, Mat *sobel_img, int start, int step);
void *thread_sobel_filter(void *threadArgs);
void video_processor(const string &usr_vid, int num_threads);
float process_timer(const string &usr_vid, int num_threads);

/***** Global Variables *****/
pthread_barrier_t barrier;

/***** Main *****/
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
            time = process_timer(usr_arg, 4);
            cout << "Processing time: " << time << endl;
            avg += time;
        }
        cout << "Average time: " << (avg / 10) << endl;

    } else if (argc == 3 && isdigit(argv[2][0])) {

        volatile int num = (uchar) argv[2][0];

        cout << "Processing time: " << process_timer(usr_arg, num) << endl;

    } else {

        cout << "Thread count set to default: 4 threads" << endl;
        cout << "Processing time: " << process_timer(usr_arg, 4) << endl;
    }
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
void sobel_filter(Mat *grayscale_img, Mat *sobel_img, int start = 0, int step = 1) {

    // index = [numCols * (row + x) + (col + y)]
    // Convolution = img[index] + (img[index] << 1) + img[index] - img[index] - (img[index] << 1) - img[index]
    // 2 V + S instructions: Vx = (Vx + Sr) & Vy = (Vy + Sy)
    // 1 V * S instruction:  Vx = Vx * Sc
    // 1 V + V instruction:  Vx = Vx + Vy
    // These operations are per-pixel, meaning that the CPU does 4 * (6 * 2) * (5 * 2) = 480 operations per sobel pixel
    // Without converting the 5 addition operations to vectors, we will still be reducing the total number of
    // operations to (2 + 1 + 1) * (5 * 2) = 40 operations -> That's 92% fewer operations!

    // init function variables
    int numRows = grayscale_img->rows;
    int numCols = grayscale_img->cols;
    int Gx, Gy, G;
    uchar *grayscale_data = grayscale_img->data;
    uchar *sobel_data = sobel_img->data;

    uint32x4_t row_vect, col_vect, row_temp, col_temp;
    uint32x4_t numCol_vect = vdupq_n_u32(numCols);

    uint32x4x4_t Gx_vect, Gx_conv_vect = {2, 1, 0, 2,
                                          1, 0, 0, 0,
                                          2, 2, 2, 0,
                                          0, 0, 0, 0};

    uint32x4x4_t Gy_vect, Gy_conv_vect = {0, 0, 0, 2,
                                          2, 2, 0, 0,
                                          0, 1, 2, 0,
                                          1, 2, 0, 0};

    // Loop through the rows and cols of the image and apply the sobel filter
    for (int row = start; row < numRows - 2; row += step) {

        // Put the row into a vector
        row_vect = vdupq_n_u32(row);

        for (int col = 0; col < numCols - 2; col++) {

            // Put the col into a vector
            col_vect = vdupq_n_u32(col);

            // Convolve Gx
            // Calculate [numCols * (row + x) + (col + y)] for bits 0 - 3
            row_temp = vaddq_u32(row_vect, Gx_conv_vect.val[0]);                // Vr = (row + i)
            col_temp = vaddq_u32(col_vect, Gx_conv_vect.val[2]);                // Vc = (col + j)
            Gx_vect.val[0] = vmlaq_u32(col_temp, row_temp, numCol_vect);    // i = Vc + Vr * V_numCols

            row_temp = vaddq_u32(row_vect, Gx_conv_vect.val[1]);
            col_temp = vaddq_u32(col_vect, Gx_conv_vect.val[3]);
            Gx_vect.val[1] = vmlaq_u32(col_temp, row_temp, numCol_vect);

            // Convolve Gy
            // Calculate bits 4 - 5
            row_temp = vaddq_u32(row_vect, Gy_conv_vect.val[0]);
            col_temp = vaddq_u32(col_vect, Gy_conv_vect.val[2]);
            Gy_vect.val[0] = vmlaq_u32(col_temp, row_temp, numCol_vect);

            row_temp = vaddq_u32(row_vect, Gy_conv_vect.val[1]);
            col_temp = vaddq_u32(col_vect, Gy_conv_vect.val[3]);
            Gy_vect.val[1] = vmlaq_u32(col_temp, row_temp, numCol_vect);

            Gx = (grayscale_data[vgetq_lane_u32(Gx_vect.val[0], 0)]) +
                 (grayscale_data[vgetq_lane_u32(Gx_vect.val[0], 1)] << 1) +
                 (grayscale_data[vgetq_lane_u32(Gx_vect.val[0], 2)]) -
                 (grayscale_data[vgetq_lane_u32(Gx_vect.val[0], 3)]) -
                 (grayscale_data[vgetq_lane_u32(Gx_vect.val[1], 0)] << 1) -
                 (grayscale_data[vgetq_lane_u32(Gx_vect.val[1], 1)]);

            Gy = (grayscale_data[vgetq_lane_u32(Gy_vect.val[0], 0)]) +
                 (grayscale_data[vgetq_lane_u32(Gy_vect.val[0], 1)] << 1) +
                 (grayscale_data[vgetq_lane_u32(Gy_vect.val[0], 2)]) -
                 (grayscale_data[vgetq_lane_u32(Gy_vect.val[0], 3)]) -
                 (grayscale_data[vgetq_lane_u32(Gy_vect.val[1], 0)] << 1) -
                 (grayscale_data[vgetq_lane_u32(Gy_vect.val[1], 1)]);

            // Gradient approximation
            G = abs(Gx) + abs(Gy);

            // Overflow check
            if (G > 255) { G = 255; }

            // Write the pixel to the sobel image
            sobel_data[sobel_img->cols * (row) + (col)] = G;
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
void video_processor(const string &usr_vid, int num_threads) {

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
float process_timer(const string &usr_vid, int num_threads = 4) {

    // Start the timer
    auto start = high_resolution_clock::now();

    // Process the video
    video_processor(usr_vid, num_threads);

    // Stop the timer and calculate duration
    auto stop = high_resolution_clock::now();
    return (float) (duration_cast<microseconds>(stop - start)).count() / 1000000;
}
