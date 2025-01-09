#include "convolution.h"

// Convolve the input matrix with the kernel matrix element-wise
void convolution_top(hls::stream<axi_stream> &in_stream, hls::stream<axi_stream> &out_stream, hls::stream<axi_lite> &kernel_stream) {
    #pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE s_axilite port=kernel_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

    // Declare input matrix, kernel, and output matrix for variable channels
    static pixel_t input_matrix[MATRIX_SIZE][MATRIX_SIZE][NUM_CHANNELS]; // Input matrix
    #pragma HLS BIND_STORAGE variable=input_matrix type=RAM_1P impl=URAM

    static pixel_t kernel[KERNEL_SIZE][KERNEL_SIZE][NUM_CHANNELS];       // Kernel matrix
    static pixel_t output_matrix[MATRIX_SIZE][MATRIX_SIZE][NUM_CHANNELS]; // Output matrix
    #pragma HLS BIND_STORAGE variable=output_matrix type=RAM_1P impl=URAM

    // Step 1: Read input matrix from AXI-Stream interface into memory (input_matrix)
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                #pragma HLS PIPELINE II=1
                axi_stream input_data = in_stream.read();
                input_matrix[row][col][ch] = input_data.data;
            }
        }
    }

    // Step 2: Read kernel matrix from AXI-Stream interface into memory (kernel)
    for (int i = 0; i < KERNEL_SIZE; i++) {
        for (int j = 0; j < KERNEL_SIZE; j++) {
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                #pragma HLS PIPELINE II=1
                axi_lite kernel_data = kernel_stream.read();
                kernel[i][j][ch] = kernel_data.data;
            }
        }
    }


    // Step 3: Perform convolution (element-wise multiplication) on input_matrix with kernel
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                pixel_t result = 0;

                // Apply kernel to the input matrix
                for (int k_row = 0; k_row < KERNEL_SIZE; k_row++) {
                    for (int k_col = 0; k_col < KERNEL_SIZE; k_col++) {
                        int in_row = row + k_row - KERNEL_SIZE / 2;
                        int in_col = col + k_col - KERNEL_SIZE / 2;

                        // Handle boundary conditions (clamping to edge)
                        if (in_row >= 0 && in_row < MATRIX_SIZE && in_col >= 0 && in_col < MATRIX_SIZE) {
                            result += input_matrix[in_row][in_col][ch] * kernel[k_row][k_col][ch];
                        }
                    }
                }

                // Store the result in the output_matrix
                output_matrix[row][col][ch] = result;
            }
        }
    }

    // Step 4: Send the result (output_matrix) to the AXI-Stream interface
    for (int row = 0; row < MATRIX_SIZE; row++) {
        for (int col = 0; col < MATRIX_SIZE; col++) {
            for (int ch = 0; ch < NUM_CHANNELS; ch++) {
                axi_stream output_data;
                output_data.data = output_matrix[row][col][ch];
                output_data.last = (row == MATRIX_SIZE - 1 && col == MATRIX_SIZE - 1 && ch == NUM_CHANNELS - 1);
                out_stream.write(output_data);
            }
        }
    }
}
