// Include Libraries
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <string>

// Namespace nullifies the use of cv::function();
using namespace std;
using namespace cv;

Mat getNextMat(int row, int col, Mat image)
{
    // might want to check if the Mat is big enough for this process

    Mat mat = Mat(3, 3, CV_8U);

    for (int r = 0; r < 3; r++)
    {
        for (int c = 0; c < 3; c++)
        {
            mat.at<int>(r, c) = image.at<int>(col + c, row + r);
        }
    }

    return mat;
}

int main(int argc, char const *argv[])
{
    // Constructing Filters
    // int x[3][3] = {
    //     {1, 0, -1},
    //     {2, 0, -2},
    //     {1, 0, -1}};

    // int y[3][3] = {
    //     {-1, -2, -1},
    //     {0, 0, 0},
    //     {1, 2, 1}};

    // Mat Gx = Mat(3, 3, CV_32S);
    // Mat Gy = Mat(3, 3, CV_32S);

    // for (int i = 0; i < 3; i++)
    // {
    //     for (int j = 0; j < 3; j++)
    //     {
    //         Gx.at<int>(j, i) = x[i][j];
    //         Gy.at<int>(j, i) = y[i][j];
    //     }
    // }

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
        return 0;
    }

    // Read the image
    Mat usr_img = imread(usr_arg);

    // Split the image into three color channels
    Mat img_planes[3];
    split(usr_img, img_planes);

    // Implement the ITU-R (BT.709) algorithm
    Mat img_grayscale = (0.2126 * img_planes[2] + 0.7152 * img_planes[1] + 0.0722 * img_planes[0]);

    Mat cropedImage = img_grayscale(Rect(100, 100, 5, 5));
    Mat test = getNextMat(1, 1, cropedImage);

    cout << "\n\n\n";
    cout << cropedImage;
    cout << "\n";

    cout << "\n";
    cout << test;
    cout << "\n\n";

    // imshow("gray", img_grayscale);

    // // Wait for a keystroke
    // waitKey(0);

    // // Destroy the window created
    // destroyAllWindows();
}
