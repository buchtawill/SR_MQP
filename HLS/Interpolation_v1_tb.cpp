#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>
#include <random>

void Interpolation_v1(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap, hls::stream<lite_data_t> &loadedInfo);

int main(){

	//create axi stream for image, featureMap, and loadedInfo
	hls::stream<ap_int<128>> image;
	hls::stream<ap_int<128>> featureMap;
	hls::stream<ap_int<32>> loadedInfo;

	//input image values
	unsigned char inputImageRed[28][28];
	unsigned char inputImageGreen[28][28];
	unsigned char inputImageBlue[28][28];
	//will need to be broken into 120 bit ints for axi-stream
	unsigned char inputImageAll[28*28*3];

	//output image values
	unsigned char outputImageRed[56][56];
	unsigned char outputImageGreen[56][56];
	unsigned char outputImageBlue[56][56];
	//will need to be reassembled from 120 bit stream outs
	unsigned char outputImageAll[56*56*3];


	//generate and store image pixel values
	for(int i = 0; i < 2352; i++){

		//generates a number between 0 and 255
		char temp = rand() % 256;
		inputImageAll[i] = temp;
		if(i % 3 == 0){
			inputImageRed[floor(i / 3)] = temp;
		}
		if(i % 3 == 1){
			inputImage[floor(i / 3)] = temp;
		}
		if(i % 3 == 2){
			inputImage[floor(i / 3)] = temp;
		}
	}


	//write data to input streams
	//5 pixels in per stream (120 bits), 157 streams required
	//only 4 pixels streamed on the last transfer
	for(int i = 0; i < 157; i++){
		ap_uint<128> valueIn;
	}

	//read data from output stream

	//verify result

}

void bilinear_interpolation(const std::vector<std::vector<int>>& input_image, std::vector<std::vector<int>>& output_image){

    // loop through output image - red
    for (int i = 0; i < 56; ++i) {
        for (int j = 0; j < 56; ++j) {

            // Find the corresponding position in the input image
        	//need to be floats in order to determine neighbors later
            float x = (float)(j) * (32 - 1) / (32 - 1);
            float y = (float)(i) * (32 - 1) / (32 - 1);

            //four nearest pixel values from input
            int x1 = std::floor(x);
            int y1 = std::floor(y);
            int x2 = std::min(x1 + 1, input_width - 1);
            int y2 = std::min(y1 + 1, input_height - 1);

            // Bilinear interpolation formula
            int top_left = input_image[y1][x1];
            int top_right = input_image[y1][x2];
            int bottom_left = input_image[y2][x1];
            int bottom_right = input_image[y2][x2];

            output_image[i][j] = (int)((top_left + top_right + bottom_left + bottom_right) /4 );
        }
    }


}
