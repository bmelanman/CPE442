#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <math.h>

// Namespace nullifies the use of std::fucntion() and cv::function();
using namespace std;
using namespace cv;

Mat grayscale_img(Mat image) {

    Mat grayscale(image.rows, image.cols, CV_8UC1);

    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            grayscale.at<uchar>(i, j) = (
                0.0722 * image.at<Vec3b>(i, j)[0] +
                0.7152 * image.at<Vec3b>(i, j)[1] +
                0.2126 * image.at<Vec3b>(i, j)[2]
                );
        }
    }

    return grayscale;
}

Mat sobel_filter(Mat grayscale_image) {

    // Scanning with a 3x3 matrix which means we can stop at the 3rd last row and col
    int row_size = grayscale_image.rows - 2;
    int col_size = grayscale_image.cols - 2;

    // Output image
    Mat sobel_img(row_size, col_size, CV_8UC1);

    // Apply the sobel filer 
    for (int i = 0; i < row_size; i++) {
        for (int j = 0; j < col_size; j++) {
            // Gx Filter:
            // [-1, 0, 1]
            // [-2, 0, 2]
            // [-1, 0, 1]

            int8_t Gx = (
                - grayscale_image.at<uchar>(i    , j)
                + grayscale_image.at<uchar>(i    , j + 2)
                - grayscale_image.at<uchar>(i + 1, j) * 2
                + grayscale_image.at<uchar>(i + 1, j + 2) * 2
                - grayscale_image.at<uchar>(i + 2, j)
                + grayscale_image.at<uchar>(i + 2, j + 2)
            );

            // Gy Filter:
            // [-1, -2, -1]
            // [0, 0, 0]
            // [1, 2, 1]
            int8_t Gy = (
                - grayscale_image.at<uchar>(i    , j)
                - grayscale_image.at<uchar>(i    , j + 1) * 2
                - grayscale_image.at<uchar>(i    , j + 2)
                + grayscale_image.at<uchar>(i + 2, j)
                + grayscale_image.at<uchar>(i + 2, j + 1) * 2
                + grayscale_image.at<uchar>(i + 2, j + 2)                
            );

            // G = |Gx| + |Gy|
            sobel_img.at<uchar>(i, j) = abs(Gx) + abs(Gy);
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

    // Convert to grayscale
    Mat img_grayscale = grayscale_img(usr_img);

    // Apply the sobel filter
    Mat img_sobel = sobel_filter(img_grayscale);

    // Display the image
    imshow(usr_arg, usr_img);
    imshow("grayscale", img_grayscale);
    imshow("sobel filter", img_sobel);

    // Wait for a keystroke
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();
}
