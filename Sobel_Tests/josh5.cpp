/*********************************************************
* File: lab4.cpp
*
* Description: program takes a video and puts it through a sobel filter using threads
*
* Author: Joshua Alderson
* 
* Revisions: 0
*
**********************************************************/

#include <opencv2/opencv.hpp>
#include "lib/pthread_barrier.h"
#include <pthread.h>
#include <arm_neon.h>

using namespace cv;
using namespace std;

//struct to hold all the variables for thread arguments
typedef struct thread_args {
    int start_gray{};
    int stop_gray{};
    int start_sobel{};
    int stop_sobel{};
    Mat frame;
    Mat gray;
    Mat sobel;
} thread_args;

//initialize two barriers. One for syncing image display and filtered
pthread_barrier_t display_barrier;
pthread_barrier_t sobel_barrier;

//global flag to determine when threads should stop
bool trim_threads = false;

//function prototype
void *thread_filter(void *args);


int main() {

    //create thread variables
    pthread_t quadrant[4];

    //make array to hold argument structs
    thread_args argument[4];

    //set up barriers for use
    pthread_barrier_init(&sobel_barrier, nullptr, 5);
    pthread_barrier_init(&display_barrier, nullptr, 5);

    VideoCapture video("../../Media/small_frog.mp4");
    //make Mat to hold each frame
    Mat vid_frame;
    //put the captured frame in the frame object
    video >> vid_frame;

    //make frame to hold the grayscale and filtered image
    Mat filtered_frame(vid_frame.rows - 2, vid_frame.cols - 2, CV_8UC1);
    Mat gray_frame(vid_frame.rows, vid_frame.cols, CV_8UC1);

    //get a quarter of all the pixels
    int data_chunk = ((vid_frame.rows * vid_frame.cols) / 4);

    //initialize arguments for the threads
    //divide values by 8 to account for vector operations
    argument[0] = {
            .start_gray = 0,
            .stop_gray = data_chunk / 8,
            .start_sobel = vid_frame.cols,
            .stop_sobel = data_chunk / 8,
            .frame = vid_frame,
            .gray = gray_frame,
            .sobel =  filtered_frame
    };

    argument[1] = {
            .start_gray = data_chunk / 8 - vid_frame.cols,
            .stop_gray = data_chunk / 4,
            .start_sobel = data_chunk / 8 - vid_frame.cols,
            .stop_sobel = data_chunk / 4,
            .frame = vid_frame,
            .gray = gray_frame,
            .sobel =  filtered_frame
    };

    argument[2] = {
            .start_gray = (data_chunk * 3) / 32,
            .stop_gray = data_chunk / 4 - vid_frame.cols,
            .start_sobel = (data_chunk * 3) / 32,
            .stop_sobel = data_chunk / 4 - vid_frame.cols,
            .frame = vid_frame,
            .gray = gray_frame,
            .sobel =  filtered_frame
    };

    argument[3] = {
            .start_gray = (data_chunk * 3) / 32 - vid_frame.cols,
            .stop_gray = data_chunk / 2,
            .start_sobel = (data_chunk * 3) / 32 - vid_frame.cols,
            .stop_sobel = data_chunk / 2 - vid_frame.cols,
            .frame = vid_frame,
            .gray = gray_frame,
            .sobel =  filtered_frame
    };

    //create the threads for computation
    for (int i = 0; i < 4; i++) {
        //creates a thread at each address of the quadrants, calls the thread_filter,
        //and passes the argument struct to the function.
        pthread_create(&quadrant[i], nullptr, thread_filter, (void *) &argument[i]);
    }

    //there will be 5 threads, 4 created and one for main
    while (true) {
        //wait for all threads and main to reach the loop
        pthread_barrier_wait(&sobel_barrier);

        //get new frame for processing
        video >> vid_frame;
        if (vid_frame.empty())
            break;


        //resize image to fit on 1920x1080 screen
//        namedWindow("vid_frame", WINDOW_NORMAL);
//        resizeWindow("vid_frame", 1920, 1080);

        //display the frame
        imshow("vid_frame", gray_frame);
//        imshow("vid_frame", filtered_frame);

        //press the escape key to close the player
        char c = (char) waitKey(1);
        if (c == 27)
            break;
        //wait for image to be shown and get a new frame
        pthread_barrier_wait(&display_barrier);
    }

    //turn on flag to stop thread loops
    trim_threads = true;
    //call wait one last time for loops to finish
    pthread_barrier_wait(&display_barrier);

    //clear the memory used for the video capture
    video.release();
    //close video player
    destroyAllWindows();
    //destroy barriers and resources they used
    pthread_barrier_destroy(&sobel_barrier);
    pthread_barrier_destroy(&display_barrier);

    return 0;
}

/*-----------------------------------------------------
* Function: thread_filter
* 
* Description: each thread will take in the frame of a video and processing 1/4 of it applying a sobel filter
* 
* param a: void pointer: start and stop values for image processing and the images to use
* 
* return: nullptr
*--------------------------------------------------------*/
void *thread_filter(void *args) {

    //get arguments and cast them
    auto *arguments = (thread_args *) (args);
    int start_gray = arguments->start_gray;
    int stop_gray = arguments->stop_gray;
    int start_sobel = arguments->start_sobel;
    int stop_sobel = arguments->stop_sobel;




    //create pointers to the picture data
    uchar *pixel;
    uchar *gray_data;
    uchar *filter_data;

    int gray_cols = arguments->gray.cols;

    //make variable to hold all the RGB values from the data.
    uint8x8x3_t colors;

    uint8x8_t r_num = vdup_n_u8(77);
    uint8x8_t b_num = vdup_n_u8(29);
    uint8x8_t g_num = vdup_n_u8(150);
    uint16x8_t min_comp_vect = vdupq_n_u16(255);

    //variable to hole the intermediate values (16 bit values)
    uint16x8_t holder_vect;

    //vector to hold the result of the gray pixel calculations
    uint8x8_t gray_vect;

    //vectors to hold kernel pixels
    uint8x8_t p1, p2, p3, p4, p6, p7, p8, p9;
    //vectors for intermediate calculations
    int16x8_t gx_holder_vect;
    int16x8_t gy_holder_vect;
    //full vector truncating values to 255
    //final vector to be stored for sobel
    uint16x8_t sobel_vect;

    while (!trim_threads) {

        pixel = arguments->frame.data + (start_gray * 3 * 8);
        gray_data = arguments->gray.data + (start_gray * 8);

        //for loop
        for (int i = start_gray; i < stop_gray; i++, pixel += (8 * 3), gray_data += 8) {

            //takes the RGB data and breaks in into 3 8x8 vectors each having one color
            //TODO: might need to change this to include an offset pixel + start?
            colors = vld3_u8(pixel);

            //multiply each vector lane by one of the constants
            //multiples the red pixels by the red number, then stores them in the holder vector
            holder_vect = vmull_u8(colors.val[0], b_num);

            //multiplies the green pixels by the number than adds it to the values in the  holder vector
            holder_vect = vmlal_u8(holder_vect, colors.val[1], g_num);

            //multiples the blue pixels by the blue number than adds it to the values in the holder vector
            holder_vect = vmlal_u8(holder_vect, colors.val[2], r_num);

            //bit shift the values from the holder vector to narrow them down from 16 to 8 bits
            gray_vect = vshrn_n_u16(holder_vect, 8);

            //store the values in the gray vector in the gray picture mat
            vst1_u8(gray_data, gray_vect);
        }

        gray_data = arguments->gray.data;
        filter_data = arguments->sobel.data;

        for (int i = start_sobel; i < stop_sobel; i++, gray_data += 8, filter_data += 8) {

            //load the kernel elements for sobel calculations and put them in vectors
            p1 = vld1_u8(gray_data);
            p2 = vld1_u8(gray_data + 1);
            p3 = vld1_u8(gray_data + 2);
            p4 = vld1_u8(gray_data + gray_cols);
            // no p5 - don't need it
            p6 = vld1_u8(gray_data + gray_cols + 2);
            p7 = vld1_u8(gray_data + 2 * gray_cols);
            p8 = vld1_u8(gray_data + 2 * gray_cols + 1);
            p9 = vld1_u8(gray_data + 2 * gray_cols + 2);


            //multiply and add correct kernel values
            /*****************gx calculations********************/
            gx_holder_vect =
                    vabsq_s16(
                            vaddq_u16(
                                    vsubq_u16(
                                            vshll_n_u8(p6, 1), vshll_n_u8(p4, 1)    //2P6 - 2P4
                                    ),
                                    vsubq_u16(
                                            vaddl_u8(p3, p9),                   //p3 + p9
                                            vaddl_u8(p1, p7)                    //p1 + p7
                                    )
                            )
                    );


            /*****************gy calculations********************/
            gy_holder_vect = vabsq_s16(
                    vaddq_u16(vsubq_u16(vaddl_s8(p1, p3), //P1 + P3
                                        vaddl_s8(p7, p9)), //P7 + P9
                              vsubq_u16(vshll_n_u8(p2, 1), vshll_n_u8(p8, 1)))); //2P2 - 2P8


            //add gx and gy
            sobel_vect = vaddq_s16(gx_holder_vect, gy_holder_vect);
            //get the min between G and the 255 vector
            sobel_vect = vminq_u16(sobel_vect, min_comp_vect);

            //narrow vector to 8 bits and store the values in memory
            vst1_u8(filter_data, vmovn_u16(sobel_vect));
        }

        //wait for all threads to finish processing the filtered image
        pthread_barrier_wait(&sobel_barrier);
        //wait until a new frame has been acquired
        pthread_barrier_wait(&display_barrier);
    }
    pthread_exit(nullptr);
}
