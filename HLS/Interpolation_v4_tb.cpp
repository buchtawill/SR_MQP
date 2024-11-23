#include "Interpolation_v4.h"

void bilinear_interpolation(uint8_t (&inputImage1D)[784], uint8_t (&outputImage1D)[3136]);

// Function to simulate writing data to the stream
void write_data(hls::stream<bus_t>& imageIn, bus_t imageValues[147]) {
    bus_t data;

    for (int i = 0; i < 147; i++) {
        data = imageValues[i];
        imageIn.write(data);
        //std::cout << "Writing data: " << data << std::endl;
    }
}

// Function to simulate reading data from the output stream
void read_data(hls::stream<bus_t>& featureMapOut, bus_t featureMapValues[588]) {
    bus_t data;

    for (int i = 0; i < 588; i++) {

        if (!featureMapOut.empty()) {
            data = featureMapOut.read();
            featureMapValues[i] = data;
            //std::cout << "Feature map output: " << data << std::endl;
        } else {
            std::cout << "Error: No data available in featureMapOut" << std::endl;
        }
    }
}

int main() {
    // Create two streams to simulate the HLS streams
    hls::stream<bus_t> imageIn;
    hls::stream<bus_t> featureMapOut;

    bus_t imageValuesTransfer[147];
    bus_t featureMapValuesTransfer[588];

    //might be able to remove imageValues and just assemble transfers from Blue, Green, Red arrays
    uint8_t imageValues[2352];

    uint8_t imageValuesBlue[784];
    uint8_t imageValuesGreen[784];
    uint8_t imageValuesRed[784];

    uint8_t featureMapValues[9408];

    uint8_t featureMapValuesBlue[3136];
    uint8_t featureMapValuesGreen[3136];
    uint8_t featureMapValuesRed[3136];

    //generate three values per loop (1 B, 1 G, 1 R)
    for(int i = 0; i < 748; i++){

    	uint8_t randBlue = rand() % 256;
    	uint8_t randGreen = rand() % 256;
    	uint8_t randRed = rand() % 256;

    	imageValues[i*3] = randBlue;
    	imageValues[i*3 + 1] = randGreen;
    	imageValues[i*3 + 2] = randRed;

    	imageValuesBlue[i] = randBlue;
    	imageValuesGreen[i] = randGreen;
    	imageValuesRed[i] = randRed;
    }

    //assemble image transfer values
    for(int i = 0; i < 147; i++){

    	bus_t temp = 0;

    	//create 128 bit bus value by combining 16, 8 bit values
    	for(int j = 0; j < 16; j++){
    		//std::cout << "Random Value: " << (int)randomVal << " ";
    		uint8_t currentValue = imageValues[i*3 + j];
			temp = temp << 8;
			temp = temp | currentValue;
    	}

    	imageValuesTransfer[i] = temp;
    	//std::cout << "\nCombined 128-bit value: " << temp << std::endl;
    	//std::cout << "\n";
    }

    /*
    for(int i = 0; i < 147; i++){

    	bus_t temp = 0;

    	//generates a random 128 bit number by combining 16, 8 bit nums
    	for(int j = 0; j < 16; j++){
			//gets random 8 bit num and adds it to temp value
    		uint8_t randomVal = rand() % 256;
    		//std::cout << "Random Value: " << (int)randomVal << " ";
			temp = temp << 8;
			temp = temp | randomVal;
    	}

    	imageValues[i] = temp;
    	//std::cout << "\nCombined 128-bit value: " << temp << std::endl;
    	//std::cout << "\n";
    }
    */

    // Simulate writing data to the imageIn stream
    write_data(imageIn, imageValuesTransfer);

    // Call the Interpolation_v4 function to read from imageIn and process the data
    Interpolation_v4(imageIn, featureMapOut);

    // Now read and print the data from the featureMapOut stream
    read_data(featureMapOut, featureMapValuesTransfer);

    /*
    for(int i = 0; i < 588; i++){
    	std::cout << "Feature map output: " << featureMapValuesTransfer[i] << std::endl;
    }
    */

    uint8_t bitMask = 0xFF;

    //disassemble feature map transfer values
    for(int i = 0; i < 588; i++){
    	//split each 128 bit bus into its 8 bit pieces
        for (int j = 0; j < 16; j++) {
            featureMapValues[i*16 + j] = (featureMapValuesTransfer[i] >> (120 - j * 8)) & bitMask;
        }
    }

    //run bilinear_interpolation in test bench
    bilinear_interpolation(imageValuesBlue, featureMapValuesBlue);
    bilinear_interpolation(imageValuesGreen, featureMapValuesGreen);
    bilinear_interpolation(imageValuesRed, featureMapValuesRed);

    //compare module values to ones from test bench
    int errors = 0;
    for(int i = 0; i < 3136; i++){

    	if(featureMapValues[i*3] != featureMapValuesBlue[i]){
    		errors++;
    		std::cout << "ERROR! Module Value: " << featureMapValues[i*3] << " TB value B: " << featureMapValuesBlue[i]<< std::endl;
    	}
    	if(featureMapValues[i*3 + 1] != featureMapValuesGreen[i]){
    		errors++;
    		std::cout << "ERROR! Module Value: " << featureMapValues[i*3 + 1] << " TB value G: " << featureMapValuesGreen[i]<< std::endl;
    	}
    	if(featureMapValues[i*3 + 2] != featureMapValuesRed[i]){
    		errors++;
    		std::cout << "ERROR! Module Value: " << featureMapValues[i*3 + 2] << " TB value R: " << featureMapValuesRed[i]<< std::endl;
    	}

    	if(errors >= 36){
    		break;
    	}
    }

    if(errors == 0){
    	std::cout << "Completed without errors!" << std::endl;
    }

    return 0;
}

void bilinear_interpolation(uint8_t (&inputImage1D)[784], uint8_t (&outputImage1D)[3136]){

	//printf("Making it to bilin interp function");

	uint8_t input_image[28][28];
	uint8_t output_image[56][56];

    // split input image into 2D array
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            input_image[i][j] = inputImage1D[i * 28 + j];
        }
    }

    // loop through output image - red
    for (int i = 0; i < 56; ++i) {
        for (int j = 0; j < 56; ++j) {

            // Find the corresponding position in the input image
        	//need to be floats in order to determine neighbors later
            float x = (float)(j) * (28 - 1) / (56 - 1);
            float y = (float)(i) * (28 - 1) / (56 - 1);

            //four nearest pixel values from input
            //consider moving to rounding up on 5 instead of flooring it, success with Diyar's code
            int x1 = std::floor(x);
            int y1 = std::floor(y);
            int x2 = std::min(x1 + 1, 28 - 1);
            int y2 = std::min(y1 + 1, 28 - 1);

            // Bilinear interpolation formula
            int top_left = input_image[y1][x1];
            int top_right = input_image[y1][x2];
            int bottom_left = input_image[y2][x1];
            int bottom_right = input_image[y2][x2];


			float interpolated_value = (1 - (x - x1)) * (1 - (y - y1)) * top_left + (x - x1) * (1 - (y - y1)) * top_right +
									   (1 - (x - x1)) * (y - y1) * bottom_left + (x - x1) * (y - y1) * bottom_right;

			// clamp result
			output_image[i][j] = std::min(std::max(0, (int)std::round(interpolated_value)), 255);
        }
    }

    // assemble 1D array from 2D array
    for (int i = 0; i < 56; i++) {
        for (int j = 0; j < 56; j++) {
            outputImage1D[i * 56 + j] = output_image[i][j];
        }
    }


}

