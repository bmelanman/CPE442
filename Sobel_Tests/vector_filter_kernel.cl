__kernel void vector_filter(__global const Mat *origImg, __global const Mat *grayImg,
                                __global const Mat *soblImg, __global const uint8_t thread_id){


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