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

/***** Global Variables *****/
pthread_barrier_t gray_barrier;
pthread_barrier_t sobl_barrier;
bool done_flag = false;
const float32x4_t GBR_const = {G_CONST, B_CONST, R_CONST, 0};
const int16x8_t Gx_kernel = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t Gy_kernel = {1, 2, 1, 0, 0, -1, -2, -1};

/***** Prototypes *****/
void *thread_gray_filter(void *threadArgs);

void *test_sobl(void *threadArgs);

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
    cout << "Video resolution: " << usr_vid.get(CAP_PROP_FRAME_HEIGHT) << "x" << usr_vid.get(CAP_PROP_FRAME_WIDTH)
         << endl;
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

    for (int i = 0; i < NUM_THREADS; i++) {

        // init grayscale filter thread variables
        gray_data[i].input = &frame;
        gray_data[i].output = &gray_frame;
        gray_data[i].start = i * usr_vid_cols * (usr_vid_rows / NUM_THREADS);

        // special case for the last thread, it must go all the way to the end
        if (i == NUM_THREADS - 1) { gray_data[i].stop = usr_vid_cols * usr_vid_rows - 1; }
        else { gray_data[i].stop = (i + 1) * usr_vid_cols * usr_vid_rows / NUM_THREADS; }

        // start running the grayscale filter threads
        pthread_create(&gray_threads[i], nullptr, &thread_gray_filter, (void *) &gray_data[i]);

        // init the sobel filter thread variables
        sobl_data[i].input = &gray_frame;
        sobl_data[i].output = &sobl_frame;

        // special case for the first thread, start at 0 instead of -2
        if (i == 0) { sobl_data[i].start = 0; }
        else { gray_data[i].start = i * (usr_vid_rows / NUM_THREADS) - 2; }

        // special case for the last thread, it must go all the way to the end
        if (i == NUM_THREADS - 1) { sobl_data[i].stop = usr_vid_rows - 2; }
        else { gray_data[i].stop = (i + 1) * (usr_vid_rows / NUM_THREADS); }

        // start running the sobel filter threads
        pthread_create(&sobl_threads[i], nullptr, test_sobl, (void *) &sobl_data[i]);
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
 * A grayscale filter for color images, applies the ITU-R (BT.709) grayscale algorithm
 * @param image An image
 * @param grayscale A grayscale image
 */
void *thread_gray_filter(void *threadArgs) {

    // init thread variables
    auto *thread_data = (struct thread_data *) threadArgs;
    uchar *user_img_data = thread_data->input->data;
    uchar *gray_img_data = thread_data->output->data;
    int start = thread_data->start;
    int stop = thread_data->stop;

    float32x4_t pixel_vect;

    // Lock the threads into the filter loop until the main loop exits the display loop
    while (!done_flag) {

        // Iterate through each pixel and apply the grayscale filter
        for (int pos = start; pos < stop; pos++) {

            pixel_vect = (float32x4_t) {(float32_t) user_img_data[3 * pos],
                                        (float32_t) user_img_data[3 * pos + 1],
                                        (float32_t) user_img_data[3 * pos + 2],
                                        0};

            gray_img_data[pos] = (uchar) vaddvq_f32(vmulq_f32(pixel_vect, GBR_const));
        }
        // wait until the main loop fetches another image to process
        pthread_barrier_wait(&gray_barrier);
    }
    pthread_exit(nullptr);
}

/**
 * Takes a grayscale image and applies the sobel operator to the given image.
 * @param threadArgs - A struct with variables for the sobel filter
 */
void *test_sobl(void *threadArgs) {

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
                Gx = vaddlvq_s16(vmulq_s16(Gx_kernel, pixel_vect));
                Gy = vaddlvq_s16(vmulq_s16(Gy_kernel, pixel_vect));

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
