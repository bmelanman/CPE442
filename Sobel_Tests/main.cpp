#include <opencv2/opencv.hpp>
#include <string>
#include <arm_neon.h>

using namespace std;
using namespace cv;
using namespace std::this_thread;
using namespace std::chrono;

#define G_CONST 0.2126
#define B_CONST 0.7152
#define R_CONST 0.0722
#define NUM_THREADS 4

const uint8x8_t G_vect = vdup_n_u8(G_CONST * 256);
const uint8x8_t B_vect = vdup_n_u8(B_CONST * 256);
const uint8x8_t R_vect = vdup_n_u8(R_CONST * 256);

int loop_count = 0;

void grayscale_filter_vector(Mat *origImg, Mat *grayImg, int start, int stop, int remainder) {

    uchar *origImgData = origImg->data + (start * 3);
    uchar *grayImgData = grayImg->data + start;

    uint8x8x3_t BGR_values;
    uint16x8_t temp;

    for (int i = start; i < stop; i += 8, origImgData += 24, grayImgData += 8) {
        BGR_values = vld3_u8(origImgData);

        temp = vmull_u8(BGR_values.val[0], B_vect);

        temp = vmlal_u8(temp, BGR_values.val[1], G_vect);
        temp = vmlal_u8(temp, BGR_values.val[2], R_vect);

        vst1_u8(grayImgData, vshrn_n_u16(temp, 8));

        loop_count += 8;
    }

    for (int i = stop; i < stop + remainder; i++) {
        grayImgData[i] = (uchar) ((origImgData[3 * i + 0] * B_CONST) +
                                  (origImgData[3 * i + 1] * G_CONST) +
                                  (origImgData[3 * i + 2] * R_CONST));

        loop_count++;
    }
}

int main() {

    // Get the name of the file the user is requesting
    string usr_arg = "/Users/brycemelander/Documents/GitHub/CPE442/Media/valve_resize.PNG";

    // init image Mats
    Mat usr_img = imread(usr_arg);
    Mat gray_img(usr_img.rows, usr_img.cols, CV_8UC1);

    // Calculate starts and stops
    int img_size = usr_img.rows * usr_img.cols;
    int remainder_pixels = img_size % (8 * NUM_THREADS);
    int num_pixels = img_size - remainder_pixels;
    int remainder = 0;
    int start;
    int stop;

    // Image resolution
    cout << "Video resolution: " << usr_img.rows << "x" << usr_img.cols << endl;
    cout << "Total Pixels: " << usr_img.rows * usr_img.cols << endl;
    cout << "Thread-processed Pixels: " << num_pixels << endl;
    cout << "Remainder Pixels: " << remainder_pixels << endl;


    for (int i = 0; i < NUM_THREADS; i++) {
        start = i * num_pixels / NUM_THREADS;
        stop = (i + 1) * num_pixels / NUM_THREADS - 1;

        cout << "Thread " << i << ":  " << start << ", " << stop << endl;

        if (i == NUM_THREADS - 1) {
            remainder = remainder_pixels;
            cout << "Remainder: " << stop + 1 << ", " << stop + remainder << endl;
        }

        grayscale_filter_vector(&usr_img, &gray_img, start, stop, remainder);
    }
    cout << "Loop Counter: " << loop_count << endl;

    // Display the image
    imshow(usr_arg, gray_img);

    // Press key to close image
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();

    return 0;
}
