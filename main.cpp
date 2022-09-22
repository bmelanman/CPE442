// Include Libraries
#include "opencv2/opencv.hpp"
#include <fstream>
#include <iostream>
#include <string>

int main(int argc, char const *argv[])
{
    // Check for valid input
    if (argc != 2)
    {
        std::cout << "Invalid input, please try again\n";
        return 0;
    }

    // Get the name of the image the user is requesting
    std::string usr_arg = argv[1];

    // Namespace nullifies the use of cv::function();
    using namespace std;
    using namespace cv;

    // Check to make sure the file exsists
    ifstream ifile;
    ifile.open(usr_arg);
    if (ifile)
    {
        // Read the image
        Mat usr_img = imread(usr_arg, IMREAD_UNCHANGED);

        // Display the image
        imshow(usr_arg, usr_img);

        // Wait for a keystroke
        waitKey(0);

        // Destroy the window created
        destroyAllWindows();
    }
    else
    {
        std::cout << "The specified file does not exist";
    }
}