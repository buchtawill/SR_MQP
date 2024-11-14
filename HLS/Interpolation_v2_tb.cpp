#include <hls_stream.h>
#include <ap_int.h>
#include <cmath>
#include <iostream>
#include <random>

void Interpolation_v1(hls::stream<stream_data_t> &image, hls::stream<stream_data_t> &featureMap, hls::stream<lite_data_t> &loadedInfo);

int main(){

	/*
	 * Initializations
	 */

	//create axi stream for image, featureMap, and loadedInfo
	hls::stream<ap_int<128>> image;
	hls::stream<ap_int<128>> featureMap;
	hls::stream<ap_int<32>> loadedInfo;

	//input image values - tb interpolation
	uint8_t inputImageRed[28][28];
	uint8_t inputImageGreen[28][28];
	uint8_t inputImageBlue[28][28];
	//will need to be broken into 120 bit ints for axi-stream
	uint8_t inputImageAll[28*28*3];

	//output image values - tb interpolation
	uint8_t outputImageRed[56][56];
	uint8_t outputImageGreen[56][56];
	uint8_t outputImageBlue[56][56];
	//will need to be reassembled from 120 bit stream outs
	uint8_t outputImageAll[56*56*3];


	/*
	 * Generate input image
	 */

	//counting up to 28*28*3 -> 2352
	int numValuesLoaded = 0;

	//generate and store image pixel values
	for(int i = 0; i < 2352; i++){

		//generates a number between 0 and 255
		uint8_t temp = rand() % 256;

		//used for block being tested
		inputImageAll[i] = temp;

		//used for block comparing against one being tested
		if(i % 3 == 0){
			inputImageRed[floor(i / 3)] = temp;
		}
		if(i % 3 == 1){
			inputImageGreen[floor(i / 3)] = temp;
		}
		if(i % 3 == 2){
			inputImageBlue[floor(i / 3)] = temp;
		}
	}



	/*
	 * Write input image to Interpolation block
	 */

	//16 color pixel values per transfer, 147 total transfers
	for(int i = 0; i < 147; i++){

		ap_uint<128> valueIn = 0;

            //loads in image from axi-stream to array of uint_8
            for(int j = 0; j < 16; j++){
            	valueIn = valueIn || (inputImageAll[i * 16 + j] << ((16 - 1 - j) * 8));
            }

		image.write(valueIn);
	}

	//run interpolation block
	Interpolation_v2(image, featureMap, loadedInfo);

	/*
	 * Read output from Interpolation block axi stream
	 */
	//9408 pixels values (56x56x3), 16 pixel values per read, 588 reads
	for(int i = 0; i < 588; i++){

		ap_uint<128> tempRead;
		unsigned char tempChar;

		tempRead = featureMap.read();

		//j < 16 because 16 chars in 128 bit transfer
		for(int j = 0; j < 16; j++){

			//get bottom 8 bits
			tempChar = tempRead & 0xF;
			//store in output image (getting in reverse order than we want them to be stored)
			outputImageAll[i*15 - j] = tempChar;
			//shift for next read
			tempRead = tempRead >> 8;
		}
	}

	//run bilinear interpolation from testbench
	bilinear_interpolation(inputImageRed, outputImageRed);
	bilinear_interpolation(inputImageGreen, outputImageGreen);
	bilinear_interpolation(inputImageBlue, outputImageBlue);

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
            //consider moving to rounding up on 5 instead of flooring it, success with Diyar's code
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
