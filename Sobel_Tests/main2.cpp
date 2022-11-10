#include <pthread.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <utility>

#define NUM_THREADS 4

using namespace cv;
using namespace std;

pthread_t filter_threads[NUM_THREADS];

struct threadData {
    Mat input{};
    int startRow{};
    int endRow{};
    Mat Qn{};
};

Mat ReturnBW(Mat image);

Mat ReturnSob(Mat image);

void getImageBounds(const Mat &image, int &r1, int &r2, int &r3) {
    r1 = image.rows / NUM_THREADS;
    r2 = r1 + r1; //(image rows/2 .. etc.)
    r3 = r1 + r2;
}

Mat retImgChunk(Mat image, int startRow, int endRow) {
    Mat Qn = Mat(endRow - startRow, image.cols, CV_8UC3);
    for (int r = startRow; r < endRow; r++) {
        auto *ptr = image.ptr<Vec3b>(r);
        auto *q1 = Qn.ptr<Vec3b>(r - startRow);
        for (int c = 0; c < image.cols; c++) {
            q1[c] = ptr[c];
        }
    }
    return Qn;
}

Mat mergeChunks(const Mat &Q1, const Mat &Q2) {

    Mat stitched = Mat(Q1.rows - 1 + Q2.rows - 1, Q1.cols, CV_8UC1);

    int pixID = 0;

    for (; pixID < Q1.rows - 1; pixID++) {
        for (int c = 0; c < Q1.cols; c++) {
            stitched.data[pixID * Q1.cols + c] = Q1.data[pixID * Q1.cols + c];
        }

    }

    for (int r = 1; r < Q2.rows; r++) {
        for (int c = 0; c < Q2.cols; c++) {
            stitched.data[pixID * Q2.cols + c] = Q2.data[r * Q2.cols + c];
        }
        pixID++;
    }

    return stitched;
}

void *ProcessImage(void *arg) {
    auto *threadData = (struct threadData *) arg;
    Mat images = threadData->input;
    int startRow = threadData->startRow;
    int endRow = threadData->endRow;
    Mat res = ReturnSob(retImgChunk(images, startRow, endRow));
    threadData->Qn = res;

    pthread_exit(nullptr);
}

int main() {
    struct threadData arg1, arg2, arg3, arg4;
    //essentially a play video program (eventually...)
    Mat Qt = Mat(0, 0, CV_8UC1);
    int r1, r2, r3;

    VideoCapture cap("../../Media/small_frog.mp4");
    if (!cap.isOpened()) {
        cout << "ERROR!" << endl;
    }
    while (true) {
        Mat image;
        cap >> image;

        if (image.empty())
            break;
        getImageBounds(image, r1, r2, r3);

        //begin thread

        arg1 = {.input = image, .startRow = 0, .endRow = r1 + 1, .Qn = Qt};
        arg2 = {.input = image, .startRow = r1 - 1, .endRow = r2 + 1, .Qn = Qt};
        arg3 = {.input = image, .startRow = r2 - 1, .endRow = r3 + 1, .Qn = Qt};
        arg4 = {.input = image, .startRow = r3 - 1, .endRow = image.rows, .Qn = Qt};

        struct threadData args[NUM_THREADS]{arg1, arg2, arg3, arg4};

        for (int i = 0; i < NUM_THREADS; i++) {
            pthread_create(&filter_threads[i], nullptr, ProcessImage, (void *) &args[i]);
        }

        for (auto &filter_thread: filter_threads) {
            pthread_join(filter_thread, nullptr);
        }

        //eventually do with multiple threads too - could merge top half with one thread and bottom
        //half with second thread. third thread or main then merges the halves.
        Mat stitch = mergeChunks(arg1.Qn, arg2.Qn);
        stitch = mergeChunks(stitch, arg3.Qn);
        stitch = mergeChunks(stitch, arg4.Qn);


        imshow("N8than4U", stitch);
        char c = (char) waitKey(25);
        if (c == 27)
            break;
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

Mat ReturnBW(Mat image) {

    Mat BW = Mat(image.rows, image.cols, CV_8UC1);
    for (int r = 0; r < image.rows; r++) {
        auto *ptr = image.ptr<Vec3b>(r);
        for (int c = 0; c < image.cols; c++) {
            BW.data[r * image.cols + c] = (uchar) (
                    Vec3b(ptr[c][0])(0) * .0722 +
                    Vec3b(ptr[c][1])(0) * .7152 +
                    Vec3b(ptr[c][2])(0) * .2126
            );
        }
    }

    return BW;
}

Mat ReturnSob(Mat image) {
    int Mag;
    Mat Sob = ReturnBW(std::move(image));
    Mat Res = Mat(Sob.rows, Sob.cols, CV_8UC1);
    for (int r = 0; r < Sob.rows; r++) {
        for (int c = 0; c < Sob.cols; c++) {
            if (r != 0 && r != Sob.rows && c != 0 && c != Sob.cols) {
                Mag = abs(-Sob.data[(r - 1) * Sob.cols + c - 1] + Sob.data[(r - 1) * Sob.cols + c + 1] -
                          2 * Sob.data[r * Sob.cols + c - 1] + 2 * Sob.data[r * Sob.cols + c + 1] -
                          Sob.data[(r + 1) * Sob.cols + c - 1] + Sob.data[(r + 1) * Sob.cols + c + 1]) +
                      abs(Sob.data[(r - 1) * Sob.cols + c - 1] + 2 * Sob.data[(r - 1) * Sob.cols + c] +
                          Sob.data[(r - 1) * Sob.cols + c + 1] - Sob.data[(r + 1) * Sob.cols + c - 1] -
                          2 * Sob.data[(r + 1) * Sob.cols + c] - Sob.data[(r + 1) * Sob.cols + c + 1]);

                if (Mag > 255) {
                    Mag = 255;
                }
                Res.data[r * Res.cols + c] = Mag;
            } else
                Res.data[r * Res.cols + c] = 0;
        }
    }

    return Res;
}
