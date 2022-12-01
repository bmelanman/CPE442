#include <opencv2/opencv.hpp>
#include <string>
#include <arm_neon.h>
#include <OpenCL/opencl.h>

using namespace std;
using namespace chrono;
using namespace cv;

#define B_CONST 19
#define G_CONST 183
#define R_CONST 54
#define NUM_THREADS 4
#define DATA_SIZE 8

#define MAX_SOURCE_SIZE (0x100000)

typedef struct uint16x8x8_t {
    uint16x8_t val[8];
} uint16x8x8_t;

// These values will be divided by 256 by the grayscale filter to avoid float math
const uint8x8_t Blue_vect = vdup_n_u8(B_CONST);     // 0.0722 * 256, rounded up
const uint8x8_t Green_vect = vdup_n_u8(G_CONST);    // 0.7152 * 256
const uint8x8_t Red_vect = vdup_n_u8(R_CONST);      // 0.2126 * 256

// Kernels used for grayscale filtering
// Small kernel is used for single pixel processing
const int16x8_t Gx_kernel_small = {-1, 0, 1, -2, 2, -1, 0, 1};
const int16x8_t Gy_kernel_small = {1, 2, 1, 0, 0, -1, -2, -1};

void filter(Mat *origImg, Mat *grayImg, Mat *soblImg, uint8_t thread_id) {

    uchar *orig_frame_data;
    uchar *gray_frame_data;
    uchar *sobl_frame_data = soblImg->data;

    uint8x8x3_t BGR_values;
    uint16x8x8_t gray_pixels;
    uint16x8_t G_vect;
    int G;

    int orig_rows = origImg->rows;
    int orig_cols = origImg->cols;
    int orig_num_pixels = orig_rows * orig_cols;

    // Grayscale filter variables
    int gray_remainder_pixels = orig_num_pixels % (DATA_SIZE * NUM_THREADS);

    int gray_start = thread_id * (orig_num_pixels - gray_remainder_pixels) / NUM_THREADS;
    int gray_stop = (thread_id + 1) * (orig_num_pixels - gray_remainder_pixels) / NUM_THREADS - 1;

    // Sobel filter variables
    int row_count = thread_id * orig_rows / NUM_THREADS;

    int sobl_start = thread_id * orig_num_pixels / NUM_THREADS;
    int sobl_stop = (thread_id + 1) * orig_num_pixels / NUM_THREADS - 1;

    if (thread_id == NUM_THREADS - 1) {
        sobl_stop = (orig_num_pixels - 1) - (2 * orig_cols) - 2;
    }

    /***** Grayscale conversion *****/

    // Reset data address position for each loop
    orig_frame_data = origImg->data + (gray_start * 3);
    gray_frame_data = grayImg->data + gray_start;

    // Go through the image 8 pixels at a time
    for (int i = gray_start; i < gray_stop; i += DATA_SIZE, orig_frame_data += 24, gray_frame_data += DATA_SIZE) {

        // Load 8 pixels in a vector
        BGR_values = vld3_u8(orig_frame_data);

        // Multiply each pixel by the predefined grayscale ratios
        // Reusing G_vect to use fewer variables overall
        G_vect = vmull_u8(BGR_values.val[0], Blue_vect);

        G_vect = vmlal_u8(G_vect, BGR_values.val[1], Green_vect);
        G_vect = vmlal_u8(G_vect, BGR_values.val[2], Red_vect);

        // Write 8 pixels at a time to grayscale frame
        vst1_u8(gray_frame_data, vshrn_n_u16(G_vect, 8));
    }

    // If the end of the image has fewer than 8 pixels, process the remainder one pixel at a time
    // This only runs for a maximum of 7 pixels , so single pixel operation is fine
    for (int i = gray_stop; i < gray_stop + gray_remainder_pixels; i++, orig_frame_data += 3, gray_frame_data++) {
        *gray_frame_data = (uchar) (
                ((*orig_frame_data * B_CONST) >> 8) +
                ((*orig_frame_data * G_CONST) >> 8) +
                ((*orig_frame_data * R_CONST) >> 8)
        );
    }

    gray_frame_data = grayImg->data;

    /***** Sobel conversion *****/

    for (int i = sobl_start; i < sobl_stop;) {

        if (i + DATA_SIZE < orig_cols * (row_count + 1)) {

            // 8 pixel processing
            // Load pixel sets into vectors
            gray_pixels.val[0] = vmovl_u8(vld1_u8(gray_frame_data + i));       // Pixel 1
            gray_pixels.val[1] = vmovl_u8(vld1_u8(gray_frame_data + i + 1));   // Pixel 2
            gray_pixels.val[2] = vmovl_u8(vld1_u8(gray_frame_data + i + 2));   // Pixel 3

            gray_pixels.val[3] = vmovl_u8(vld1_u8(gray_frame_data + i + orig_cols));       // Pixel 4
            gray_pixels.val[4] = vmovl_u8(vld1_u8(gray_frame_data + i + 2 + orig_cols));   // Pixel 6

            gray_pixels.val[5] = vmovl_u8(vld1_u8(gray_frame_data + i + (orig_cols << 1)));       // Pixel 7
            gray_pixels.val[6] = vmovl_u8(vld1_u8(gray_frame_data + i + 1 + (orig_cols << 1)));   // Pixel 8
            gray_pixels.val[7] = vmovl_u8(vld1_u8(gray_frame_data + i + 2 + (orig_cols << 1)));   // Pixel 9

            G_vect = (
                    vaddq_u16(
                            vabsq_s16(
                                    vaddq_s16(                                          // E =
                                            vsubq_s16(
                                                    vshlq_n_u16(gray_pixels.val[1], 1),
                                                    vshlq_n_u16(gray_pixels.val[6], 1)  // C = 2*P2 - 2*P8
                                            ),
                                            vsubq_s16(                                  // D = A - B
                                                    vaddq_u16(                          // A = P1 + P3
                                                            gray_pixels.val[0],
                                                            gray_pixels.val[2]
                                                    ),
                                                    vaddq_u16(                          // B = P7 + P9
                                                            gray_pixels.val[5],
                                                            gray_pixels.val[7]
                                                    )
                                            )
                                    )
                            ),
                            vabsq_s16(
                                    vaddq_s16(
                                            vsubq_s16(
                                                    vshlq_n_u16(gray_pixels.val[4], 1),
                                                    vshlq_n_u16(gray_pixels.val[3], 1)  // 2*P6 - 2*P4
                                            ),
                                            vsubq_s16(                                  // (P3+P9) - (P1+P7)
                                                    vaddq_u16(                          // P3 + P9
                                                            gray_pixels.val[2],
                                                            gray_pixels.val[7]
                                                    ),
                                                    vaddq_u16(                          // P1 + P7
                                                            gray_pixels.val[0],
                                                            gray_pixels.val[5]
                                                    )
                                            )
                                    )
                            )
                    )
            );

            vst1_u8(sobl_frame_data + i - (2 * row_count), vqmovn_u16(G_vect));
            i += DATA_SIZE;

        } else {

            // If there's some remaining pixels, filter them one by one
            // Reusing variables here
            G_vect = (uint16x8_t) {
                    gray_frame_data[i], gray_frame_data[i + 1], gray_frame_data[i + 2],
                    gray_frame_data[i + orig_cols], gray_frame_data[i + 2 + orig_cols],
                    gray_frame_data[i + (orig_cols * 2)], gray_frame_data[i + 1 + (orig_cols * 2)],
                    gray_frame_data[i + 2 + (orig_cols * 2)]
            };

            // Calculate G from the pixels, lots of conversions at once to avoid using more variables
            G = abs(vaddlvq_s16(vmulq_s16(Gx_kernel_small, G_vect))) +
                abs(vaddlvq_s16(vmulq_s16(Gy_kernel_small, G_vect)));

            // Write the pixel to the sobel image
            sobl_frame_data[i - (2 * row_count)] = saturate_cast<uchar>(G);
            i++;
        }

        if (i % orig_cols == 0) {
            row_count++;
        }
    }
}

void openclTest() {

    // Create the two input vectors
    int i;
    const int LIST_SIZE = 1024;
    int *A = (int *) malloc(sizeof(int) * LIST_SIZE);
    int *B = (int *) malloc(sizeof(int) * LIST_SIZE);

    for (i = 0; i < LIST_SIZE; i++) {
        A[i] = i;
        B[i] = LIST_SIZE - i;
    }

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("../vector_add_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    // Create memory buffers on the device for each vector
    cl_mem a_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_SIZE * sizeof(int), nullptr, &ret);
    cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                      LIST_SIZE * sizeof(int), nullptr, &ret);
    cl_mem c_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                      LIST_SIZE * sizeof(int), nullptr, &ret);

    // Copy the lists A and B to their respective memory buffers
    ret = clEnqueueWriteBuffer(command_queue, a_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(int), A, 0, nullptr, nullptr);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0,
                               LIST_SIZE * sizeof(int), B, 0, nullptr, nullptr);

    // Create a program from the kernel source
    cl_program program = clCreateProgramWithSource(context, 1,
                                                   (const char **) &source_str, (const size_t *) &source_size, &ret);

    // Build the program
    ret = clBuildProgram(program, 1, &device_id, nullptr, nullptr, nullptr);

    // Create the OpenCL kernel
    cl_kernel kernel = clCreateKernel(program, "vector_add", &ret);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *) &a_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *) &b_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *) &c_mem_obj);

    // Execute the OpenCL kernel on the list
    size_t global_item_size = LIST_SIZE; // Process the entire lists
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr,
                                 &global_item_size, &local_item_size, 0, nullptr, nullptr);

    // Read the memory buffer C on the device to the local variable C
    int *C = (int *) malloc(sizeof(int) * LIST_SIZE);
    ret = clEnqueueReadBuffer(command_queue, c_mem_obj, CL_TRUE, 0,
                              LIST_SIZE * sizeof(int), C, 0, nullptr, nullptr);

    // Display the result to the screen
    for (i = 0; i < LIST_SIZE; i++)
        printf("%d + %d = %d\n", A[i], B[i], C[i]);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(a_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(c_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    free(A);
    free(B);
    free(C);
}

void runFilter(Mat *origImg, Mat *grayImg, Mat *soblImg) {

    // init and run grayscale filter
    for (int i = 0; i < NUM_THREADS; i++) {

        filter(origImg, grayImg, soblImg, i);
    }

    resize(*soblImg, *soblImg, Size(), 10, 10, INTER_LANCZOS4);
    imshow("Regular Filter", *soblImg);
    waitKey(0);

    // Destroy the window created
    destroyAllWindows();
}

void OpenCL_runFilter(Mat *origImg, Mat *grayImg, Mat *soblImg) {

    // Load the kernel source code into the array source_str
    FILE *fp;
    char *source_str;
    size_t source_size;

    fp = fopen("../vector_filter_kernel.cl", "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel.\n");
        exit(1);
    }
    source_str = (char *) malloc(MAX_SOURCE_SIZE);
    source_size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);

    // Get platform and device information
    cl_platform_id platform_id = nullptr;
    cl_device_id device_id = nullptr;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    // Create an OpenCL context
    cl_context context = clCreateContext(nullptr, 1, &device_id, nullptr, nullptr, &ret);

    // Create a command queue
    cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    cl_mem orig_img = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(*origImg), nullptr, &ret);



}

int main() {

    // Get the name of the file the user is requesting
    string usr_arg = "/Users/brycemelander/Documents/GitHub/CPE442/Media/valve_resize.PNG";

    // init image Mats
    Mat usr_img = imread(usr_arg);

    // Calculate starts and stops
    int num_rows = usr_img.rows;
    int num_cols = usr_img.cols;

    Mat gray_img(num_rows, num_cols, CV_8UC1);
    Mat sobl_img(num_rows - 2, num_cols - 2, CV_8UC1);

    // Test the original vectorized filter
    runFilter(&usr_img, &gray_img, &sobl_img);

    // Test OpenCL
    OpenCL_runFilter(&usr_img, &gray_img, &sobl_img);

    return 0;
}
