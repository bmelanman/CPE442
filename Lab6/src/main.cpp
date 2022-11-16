/*********************************************************
* File: main.cpp
*
* Description: This program takes full color videos,
*              converts them to grayscale and applies
*              the sobel operator frame by frame. The
*              speed of the video is dependant on the
*              time it takes to process each frame.
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
using namespace chrono;
using namespace cv;

/***** Defines *****/
#define G_CONST 183
#define B_CONST 19
#define R_CONST 54
#define NUM_THREADS 4
#define DATA_SIZE 8

/***** Structures *****/
struct big_thread_data {
    Mat *orig_frame{};
    Mat *gray_frame{};
    Mat *sobl_frame{};
    int thread_id{};
};
typedef struct uint16x8x8_t {
    uint16x8_t val[8];
} uint16x8x8_t;

/***** Global Variables *****/
pthread_barrier_t filter_barrier;

// While-loop lock variable for keeping the threads in their respective filters
bool thread_lock = false;

// These values will be divided by 256 by the grayscale filter to avoid float math
const uint8x8_t Green_vect = vdup_n_u8(G_CONST);    // 0.7152 * 256
const uint8x8_t Blue_vect = vdup_n_u8(B_CONST);     // 0.0722 * 256, rounded up
const uint8x8_t Red_vect = vdup_n_u8(R_CONST);      // 0.2126 * 256

// Kernels used for single pixel grayscale filtering
const int16x8_t Gx_kernel_small = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t Gy_kernel_small = {1, 2, 1, 0, 0, -1, -2, -1};

/***** Prototypes *****/

void *filter(void *threadArgs);

/***** Main *****/
int main(int argc, char const *argv[]) {

    // validate input args
    if (argc != 2) {
        cout << "Invalid input" << endl;
        exit(-1);
    }

    // get video file name
    string usrVideo_location = argv[1];
    ifstream ifile;
    ifile.open(usrVideo_location);

    // check file exists
    if (!ifile) {
        cout << "File does not exist" << endl;
        exit(-1);
    }

    // Open the file as a video and display info for user
    VideoCapture usrVideo(usrVideo_location);
    auto fps = (uint8_t) usrVideo.get(CAP_PROP_FPS);
    int num_frames = (int) usrVideo.get(CAP_PROP_FRAME_COUNT);

    printf("Video Resolution: %dx%d\n", (int) usrVideo.get(CAP_PROP_FRAME_HEIGHT),
           (int) usrVideo.get(CAP_PROP_FRAME_WIDTH));
    printf("Number of frames: %d\n", num_frames);
    printf("Video Length: %d.%d seconds\n\n", num_frames / fps, (num_frames / fps) % 1);

    // Init Mats for each filter
    int usrVideo_rows = (int) usrVideo.get(4);
    int usrVideo_cols = (int) usrVideo.get(3);
    Mat frame(usrVideo_rows, usrVideo_cols, CV_8UC3);
    Mat gray_frame(usrVideo_rows, usrVideo_cols, CV_8UC1);
    Mat sobl_frame(usrVideo_rows - 2, usrVideo_cols - 2, CV_8UC1);

    // Init pthreads and barriers
    pthread_t filter_threads[NUM_THREADS];
    pthread_barrier_init(&filter_barrier, nullptr, NUM_THREADS + 1);

    // Init thread struct variables
    struct big_thread_data thread_data[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {

        // init filter thread variables
        thread_data[i] = {
                &frame,
                &gray_frame,
                &sobl_frame,
                i
        };

        // run the threads
        pthread_create(&filter_threads[i], nullptr, &filter, (void *) &thread_data[i]);
    }

    // Loop through the image file
    while (usrVideo.isOpened()) {

        // Get a frame from the video
        usrVideo >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            break;
        }

        // Wait for the image to be processed
        pthread_barrier_wait(&filter_barrier);

        // let the image display for a second
        waitKey(1);

        // Display the frame
        imshow(usrVideo_location, sobl_frame);
    }

    // waitKey technically displays the image
    waitKey(1);

    // set the done flag high to release the threads from the filters
    thread_lock = true;

    // Rejoin all threads
    pthread_barrier_wait(&filter_barrier);

    // Clean up threads
    for (auto &filter_thread: filter_threads) {
        pthread_join(filter_thread, nullptr);
    }

    // Clean up thread barriers
    pthread_barrier_destroy(&filter_barrier);

    // clean up video stuff
    usrVideo.release();
    destroyAllWindows();

    return 0;
}

/***** Project Functions *****/

/**
 * A grayscale filter for color images, applies the ITU-R (BT.709) grayscale algorithm. The function utilizes ARM neon
 * vectors to optimize the speed of the grayscale algorithm.
 * @param image An image
 * @param grayscale A grayscale image
 * Takes a grayscale image and applies the sobel operator to the given image.
 * @param threadArgs - A struct with variables for the sobel filter
 */
void *filter(void *threadArgs) {

    /***** Thread variables *****/
    auto *thread_data = (struct big_thread_data *) threadArgs;
    uchar *orig_frame_data;
    uchar *gray_frame_data;
    uchar *sobl_frame_data;
    uint8_t thread_id = thread_data->thread_id;

    /***** Filter variables *****/
    uint8x8x3_t BGR_values;
    uint16x8x8_t gray_pixels;
    uint16x8_t G_vect;
    int G;

    int orig_rows = thread_data->orig_frame->rows;
    int orig_cols = thread_data->orig_frame->cols;
    int orig_num_pixels = orig_rows * orig_cols;

    // Grayscale filter variables
    int gray_remainder_pixels = orig_num_pixels % (DATA_SIZE * NUM_THREADS);

    int gray_start = thread_id * (orig_num_pixels - gray_remainder_pixels) / NUM_THREADS;
    int gray_stop = (thread_id + 1) * (orig_num_pixels - gray_remainder_pixels) / NUM_THREADS - 1;

    // Sobel filter variables
    int row_count;

    int sobl_start = thread_id * orig_rows * orig_cols / NUM_THREADS;
    int sobl_stop = (thread_id + 1) * orig_rows * orig_cols / NUM_THREADS - 1;

    if (thread_id == NUM_THREADS - 1) {
        sobl_stop = (orig_rows * orig_cols - 1) - (2 * orig_cols) - 2;
    }

    // Wait main to load the first frame
    pthread_barrier_wait(&filter_barrier);

    /***** Begin filtering *****/

    // Lock the threads in the image processing loop until we've processed every frame of the video
    while (!thread_lock) {

        /***** Grayscale conversion *****/

        // Reset data address position for each loop
        orig_frame_data = thread_data->orig_frame->data + (gray_start * 3);
        gray_frame_data = thread_data->gray_frame->data + gray_start;
        sobl_frame_data = thread_data->sobl_frame->data;

        // Go through the image 8 pixels at a time
        for (int i = gray_start; i < gray_stop; i += 8, orig_frame_data += 24, gray_frame_data += 8) {

            // Load 8 pixels in a vector
            BGR_values = vld3_u8(orig_frame_data);

            // Multiply each pixel by the predefined grayscale ratios
            // Reusing G_vect to use fewer variables overall
            G_vect = vmull_u8(BGR_values.val[0], Blue_vect);

            G_vect = vmlal_u8(G_vect, BGR_values.val[1], Green_vect);
            G_vect = vmlal_u8(G_vect, BGR_values.val[2], Red_vect);

            // Write 8 pixels at a time to grayscale frame
            vst1_u8(gray_frame_data, vshrn_n_u16(G_vect, 8));
        }

        // If the end of the image has fewer than 8 pixels, process the remainder one pixel at a time
        // This only runs for a maximum of 7 pixels , so single pixel operation is fine
        for (int i = gray_stop; i < gray_stop + gray_remainder_pixels; i++, orig_frame_data += 3, gray_frame_data++) {
            *gray_frame_data = (uchar) (
                    ((*orig_frame_data * B_CONST) >> 8) +
                    ((*orig_frame_data * G_CONST) >> 8) +
                    ((*orig_frame_data * R_CONST) >> 8)
            );
        }

        gray_frame_data = thread_data->gray_frame->data;
        row_count = thread_id * orig_rows / NUM_THREADS;

        // Wait until all threads are done with grayscale
        pthread_barrier_wait(&filter_barrier);

        /***** Sobel conversion *****/

        for (int i = sobl_start; i < sobl_stop;) {

            // Verify that the index we are currently at has enough data to get 8 pixels
            if (i + DATA_SIZE < orig_cols * (row_count + 1)) {

                // 8 pixel processing
                // Load pixel sets into vectors
                gray_pixels.val[0] = vmovl_u8(vld1_u8(gray_frame_data + i));       // Pixel 1
                gray_pixels.val[1] = vmovl_u8(vld1_u8(gray_frame_data + i + 1));   // Pixel 2
                gray_pixels.val[2] = vmovl_u8(vld1_u8(gray_frame_data + i + 2));   // Pixel 3

                gray_pixels.val[3] = vmovl_u8(vld1_u8(gray_frame_data + i + orig_cols));       // Pixel 4
                gray_pixels.val[4] = vmovl_u8(vld1_u8(gray_frame_data + i + 2 + orig_cols));   // Pixel 6

                gray_pixels.val[5] = vmovl_u8(vld1_u8(gray_frame_data + i + (orig_cols << 1)));       // Pixel 7
                gray_pixels.val[6] = vmovl_u8(vld1_u8(gray_frame_data + i + 1 + (orig_cols << 1)));   // Pixel 8
                gray_pixels.val[7] = vmovl_u8(vld1_u8(gray_frame_data + i + 2 + (orig_cols << 1)));   // Pixel 9

                G_vect = (
                        vaddq_u16(
                                vabsq_s16(
                                        vaddq_s16(                                          // E =
                                                vsubq_s16(
                                                        vshlq_n_u16(gray_pixels.val[1], 1),
                                                        vshlq_n_u16(gray_pixels.val[6], 1)  // C = 2*P2 - 2*P8
                                                ),
                                                vsubq_s16(                                  // D = A - B
                                                        vaddq_u16(                          // A = P1 + P3
                                                                gray_pixels.val[0],
                                                                gray_pixels.val[2]
                                                        ),
                                                        vaddq_u16(                          // B = P7 + P9
                                                                gray_pixels.val[5],
                                                                gray_pixels.val[7]
                                                        )
                                                )
                                        )
                                ),
                                vabsq_s16(
                                        vaddq_s16(
                                                vsubq_s16(
                                                        vshlq_n_u16(gray_pixels.val[4], 1),
                                                        vshlq_n_u16(gray_pixels.val[3], 1)  // 2*P6 - 2*P4
                                                ),
                                                vsubq_s16(                                  // (P3+P9) - (P1+P7)
                                                        vaddq_u16(                          // P3 + P9
                                                                gray_pixels.val[2],
                                                                gray_pixels.val[7]
                                                        ),
                                                        vaddq_u16(                          // P1 + P7
                                                                gray_pixels.val[0],
                                                                gray_pixels.val[5]
                                                        )
                                                )
                                        )
                                )
                        )
                );

                vst1_u8(sobl_frame_data + i - (2 * row_count), vqmovn_u16(G_vect));
                i += DATA_SIZE;

            }

            // If the index is not at a position where we can grab 8 pixels, we filter them one by one
            else {

                // Reusing variables here
                G_vect = (uint16x8_t) {
                        gray_frame_data[i], gray_frame_data[i + 1], gray_frame_data[i + 2],
                        gray_frame_data[i + orig_cols], gray_frame_data[i + 2 + orig_cols],
                        gray_frame_data[i + (orig_cols * 2)], gray_frame_data[i + 1 + (orig_cols * 2)],
                        gray_frame_data[i + 2 + (orig_cols * 2)]
                };

                // Calculate G from the pixels, lots of conversions at once to avoid using more variables
                G = abs(vaddlvq_s16(vmulq_s16(Gx_kernel_small, G_vect))) +
                    abs(vaddlvq_s16(vmulq_s16(Gy_kernel_small, G_vect)));

                // Write the pixel to the sobel image
                sobl_frame_data[i - (2 * row_count)] = saturate_cast<uchar>(G);
                i++;
            }

            // Increase the row count every time we move to a new row
            if (i % orig_cols == 0) {
                row_count++;
            }
        }
        // Wait until the main loop moves on to the next image
        pthread_barrier_wait(&filter_barrier);
    }
    pthread_exit(nullptr);
}

