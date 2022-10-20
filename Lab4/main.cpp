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
* Revisions: V1.3
*
**********************************************************/

// Includes
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>
#include <pthread.h>

// Defines
#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
// #define PTHREAD_BARRIER_SERIAL_THREAD -1

// Namespaces
using namespace std;
using namespace cv;

// Structures 
typedef struct thread_data {
    Mat* input;
    Mat* output;
    int start;
    int step;
} in_out;

// typedef struct {
//     pthread_mutex_t mutex;
//     pthread_cond_t condition_variable;
//     int threads_required;
//     int threads_left;
//     unsigned int cycle;
// } pthread_barrier_t;

// Global Variables
pthread_barrier_t barrier;

// Re-implementation of pThread barrier functions
/*
int pthread_barrier_init(pthread_barrier_t* barrier, void* attr, int count) {
    barrier->threads_required = count;
    barrier->threads_left = count;
    barrier->cycle = 0;
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->condition_variable, NULL);
    return 0;
}

int pthread_barrier_wait(pthread_barrier_t* barrier) {
    pthread_mutex_lock(&barrier->mutex);

    if (--barrier->threads_left == 0) {
        barrier->cycle++;
        barrier->threads_left = barrier->threads_required;

        pthread_cond_broadcast(&barrier->condition_variable);
        pthread_mutex_unlock(&barrier->mutex);

        return PTHREAD_BARRIER_SERIAL_THREAD;
    }
    else {
        unsigned int cycle = barrier->cycle;

        while (cycle == barrier->cycle)
            pthread_cond_wait(&barrier->condition_variable, &barrier->mutex);

        pthread_mutex_unlock(&barrier->mutex);
        return 0;
    }
}

int pthread_barrier_destroy(pthread_barrier_t* barrier) {
    pthread_cond_destroy(&barrier->condition_variable);
    pthread_mutex_destroy(&barrier->mutex);
    return 0;
}
*/

// Project Functions 
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
    int start = thread_data->start;
    int step = thread_data->step;

    // init function varaibles 
    int numRows = thread_data->input->rows;
    int numCols = thread_data->input->cols;
    int16_t Gx, Gy, G;
    uchar* grayscale_data = thread_data->input->data;
    uchar* sobel_data = thread_data->output->data;

    // Loop through the rows and cols of the image and apply the sobel filter
    for (int row = start; row < numRows - 2; row += step) {
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
                sobel_data[(thread_data->output->cols * (row)+(col))] = 255;
            }
            else {
                sobel_data[(thread_data->output->cols * (row)+(col))] = (uchar)G;
            }
        }
    }

    // pThread return functions 
    pthread_barrier_wait(&barrier);
    pthread_exit(NULL);
}

void video_processor(String video_file, int num_threads) {

    // init pThreads
    pthread_t threads[num_threads];

    // init data used by threads
    struct thread_data in_out[num_threads];

    // Read the video
    VideoCapture usr_vid(video_file);
    int usr_vid_rows = usr_vid.get(4);
    int usr_vid_cols = usr_vid.get(3);

    // init image Mats 
    Mat frame;
    Mat gray_frame(usr_vid_rows, usr_vid_cols, CV_8UC1);
    Mat sobel_frame(usr_vid_rows - 2, usr_vid_cols - 2, CV_8UC1);

    for (int i = 0; i < num_threads; i++) {
        in_out[i].input = &gray_frame;
        in_out[i].output = &sobel_frame;
        in_out[i].step = num_threads;
        in_out[i].start = i;
    }

    // Loop through the image file
    while (usr_vid.isOpened()) {
        // Get a frame from the video
        usr_vid >> frame;

        // If we're all out of frames, the video is over
        if (frame.empty()) {
            break;
        }

        // Process the image
        grayscale_filter(&frame, &gray_frame);

        pthread_barrier_init(&barrier, NULL, num_threads);

        // Each thread processes half of the image
        for (int i = 0; i < num_threads; i++) {
            pthread_create(&threads[i], NULL, &threaded_sobel, (void*)&in_out[i]);
        }

        // Dislplay the frame
        imshow(video_file, sobel_frame);

        // Hold ESC to exit the video early
        if ((char)waitKey(25) == 27) break;
    }

    // Clean up
    usr_vid.release();
    destroyAllWindows();
    pthread_barrier_destroy(&barrier);
}

int main(int argc, char const* argv[]) {

    // Check for valid input
    if (2 > argc || argc > 3) {
        cout << "Invalid input, please try again\n";
        exit(-1);
    }

    // Get the file the user is requesting
    string usr_arg = argv[1];
    ifstream ifile;
    ifile.open(usr_arg);

    // Check to make sure the file exsists, quit if it does not
    if (!ifile) {
        cout << "The specified file does not exist";
        exit(-1);
    }
    else if (usr_arg.substr(usr_arg.size() - 4) != ".mp4") {
        cout << "This file is not supported";
        exit(-1);
    }

    // Set the number of threads 
    int num_threads = 4;

    // Check if the user gave a thread count value
    if (argc == 3) {
        num_threads = stoi(argv[2]);
    }
    else {
        cout << "Thread count set to default: 4 threads\n";
    }

    video_processor(usr_arg, num_threads);

    return 0;
}
