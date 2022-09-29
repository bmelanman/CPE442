// Include Libraries
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <string>

// Namespace nullifies the use of cv::function();
using namespace std;
using namespace cv;

int main(int argc, char const *argv[])
{
    int x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}};

    int y[3][3] = {
        {1, 2, 1},
        {0, 0, 0},
        {-1, -2, -1}};

    Mat Gx(3, 3, CV_64U, x);

    Mat Gy(3, 3, CV_64S, y);

    cout << x;
    cout << Gx;

    Mat test = Gx * Gy;

    Scalar tester = sum(test);

    cout << tester;

    // Check for valid input
    if (argc != 2)
    {
        cout << "Invalid input, please try again\n";
        return 0;
    }

    // Get the name of the image the user is requesting
    string usr_arg = argv[1];

    // Check to make sure the file exsists
    ifstream ifile;
    ifile.open(usr_arg);

    // If the file doesn't exist, quit
    if (!ifile)
    {
        cout << "The specified file does not exist";
        exit(0);
    }

    // Read the image
    Mat usr_img = imread(usr_arg, IMREAD_UNCHANGED);

    // Split the image into three color channels
    Mat img_planes[3];
    split(usr_img, img_planes);

    // Implement the ITU-R (BT.709) algorithm
    Mat img_grayscale = (0.2126 * img_planes[2] + 0.7152 * img_planes[1] + 0.0722 * img_planes[0]);

    for (int i = 0; i < img_grayscale.cols; i++)
    {

        for (int j = 0; j < img_grayscale.rows; j++)
        {
        }
    }

    imshow("gray", img_grayscale);

    // Wait for a keystroke
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();
}