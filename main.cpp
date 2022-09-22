//Include Libraries
#include <opencv2/opencv.hpp>
#include<iostream>
 
// Namespace nullifies the use of cv::function();
using namespace std;
using namespace cv;
 
// Read an image
Mat img_grayscale = imread("test.jpg", 0);

// Display the image.
imshow("grayscale image", img_grayscale);
 
// Wait for a keystroke.  
waitKey(0); 
 
// Destroys all the windows created                        
destroyAllWindows();

// Write the image in the same directory
imwrite("grayscale.jpg", img_grayscale);
