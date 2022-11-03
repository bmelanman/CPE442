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
#include "../lib/pthread_barrier.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <arm_neon.h>

/***** Namespaces *****/
using namespace std;
using namespace cv;

/***** Defines *****/
#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

/***** Global Variables *****/
pthread_barrier_t gray_barrier;
pthread_barrier_t sobl_barrier;
bool done_flag = false;

const uint8x8_t G_vect = vdup_n_u8(G_CONST * 256);
const uint8x8_t B_vect = vdup_n_u8(B_CONST * 256);
const uint8x8_t R_vect = vdup_n_u8(R_CONST * 256);

const int16x8_t Gx_kernel = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t Gy_kernel = {1, 2, 1, 0, 0, -1, -2, -1};

/***** Structures *****/
struct thread_data {
    Mat *input{};
    Mat *output{};
    int start{};
    int stop{};
    int remainder{};
};

/***** Prototypes *****/
void *thread_gray_filter_vector(void *threadArgs);

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
    cout << "Video resolution: " << usr_vid.get(CAP_PROP_FRAME_HEIGHT)
         << "x" << usr_vid.get(CAP_PROP_FRAME_WIDTH) << endl;
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
    int img_size = usr_vid_rows * usr_vid_cols;
    int remainder_pixels = img_size % (8 * NUM_THREADS);
    int num_pixels = img_size - remainder_pixels;
    int remainder = 0;

    for (int i = 0; i < NUM_THREADS; i++) {

        // init grayscale filter thread variables
        gray_data[i].input = &frame;
        gray_data[i].output = &gray_frame;
        gray_data[i].start = i * num_pixels / NUM_THREADS;
        gray_data[i].stop = (i + 1) * num_pixels / NUM_THREADS - 1;
        gray_data[i].remainder = remainder;

        if (i == NUM_THREADS - 1) {
            remainder = remainder_pixels;
        }

        // start running the grayscale filter threads
        pthread_create(&gray_threads[i], nullptr, &thread_gray_filter_vector, (void *) &gray_data[i]);

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
 * A grayscale filter for color images, applies the ITU-R (BT.709) grayscale algorithm. The function utilizes ARM neon
 * vectors to optimize the speed of the grayscale algorithm.
 * @param image An image
 * @param grayscale A grayscale image
 */
void *thread_gray_filter_vector(void *threadArgs) {

    auto *thread_data = (struct thread_data *) threadArgs;
    int remainder = thread_data->remainder;
    int start = thread_data->start;
    int stop = thread_data->stop;

    uchar *origImgData;
    uchar *grayImgData;
    uint8x8x3_t BGR_values;
    uint16x8_t temp;

    while (!done_flag) {
        origImgData = thread_data->input->data + (start * 3);
        grayImgData = thread_data->output->data + start;

        for (int i = start; i < stop; i += 8, origImgData += 24, grayImgData += 8) {
            BGR_values = vld3_u8(origImgData);

            temp = vmull_u8(BGR_values.val[0], B_vect);

            temp = vmlal_u8(temp, BGR_values.val[1], G_vect);
            temp = vmlal_u8(temp, BGR_values.val[2], R_vect);

            vst1_u8(grayImgData, vshrn_n_u16(temp, 8));
        }

        for (int i = stop; i < stop + remainder; i++, origImgData += 3, grayImgData++) {
            *grayImgData = (uchar) ((*origImgData * B_CONST) +
                                    (*origImgData * G_CONST) +
                                    (*origImgData * R_CONST));
        }
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
