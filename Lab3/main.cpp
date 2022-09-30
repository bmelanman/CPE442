// Include Libraries
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

// Namespace nullifies the use of cv::function();
using namespace std;
using namespace cv;

Mat getNextMat(int row, int col, Mat image) {
    // might want to check if the Mat is big enough for this process?

    Mat mat = Mat(3, 3, CV_32S);

    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
            mat.at<int>(c, r) = image.at<int>(col + c, row + r);
        }
    }

    return mat;
}

Mat sobel_filter(Mat gray_image) {
    int row_size = gray_image.rows - 2;
    int col_size = gray_image.cols - 2;

    // Output image
    Mat sobel_img = Mat(row_size, col_size, CV_8U);

    // Constructing Filters
    int x[3][3] = {
        {1, 0, -1},
        {2, 0, -2},
        {1, 0, -1} };

    int y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1} };

    Mat Gx = Mat(3, 3, CV_32S);
    Mat Gy = Mat(3, 3, CV_32S);

    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Gx.at<int>(j, i) = x[i][j];
            Gy.at<int>(j, i) = y[i][j];
        }
    }

    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < col_size; j++) {
            Mat img_box = getNextMat(i, j, gray_image);

            sobel_img.at<int>(j, i) = abs(sum(img_box.mul(Gx))[0]) + abs(sum(img_box.mul(Gy))[0]);
        }
    }

    return sobel_img;
}

int main(int argc, char const* argv[]) {
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

    // If the file doesn't exist, quit
    if (!ifile) {
        cout << "The specified file does not exist";
        return 0;
    }

    // Read the image
    Mat usr_img = imread(usr_arg);

    // Split the image into three color channels
    Mat img_planes[3];
    split(usr_img, img_planes);

    // Implement the ITU-R (BT.709) algorithm
    Mat img_grayscale = (0.2126 * img_planes[2] + 0.7152 * img_planes[1] + 0.0722 * img_planes[0]);

    // Apply the sobel filter
    Mat img_sobel = sobel_filter(img_grayscale);

    imshow("sobel filter", img_sobel);

    // Wait for a keystroke
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();
}
