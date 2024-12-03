#include "Interpolation_v5.h"

std::vector<uint8_t> bilinearInterpolation(const std::vector<uint8_t>& image, int width, int height, int channels, float scale);

// Function to simulate writing data to the stream
void write_data(hls::stream<bus_t>& imageIn, bus_t imageValues[numOfTransfersIn]) {
    bus_t data;

    for (int i = 0; i < numOfTransfersIn; i++) {
        data = imageValues[i];
        imageIn.write(data);
        //std::cout << "Writing data: " << data << std::endl;
    }
}

// Function to simulate reading data from the output stream
void read_data(hls::stream<bus_t>& featureMapOut, bus_t featureMapValues[numOfTransfersOut]) {
    bus_t data;

    for (int i = 0; i < numOfTransfersOut; i++) {

        if (!featureMapOut.empty()) {
            data = featureMapOut.read();
            featureMapValues[i] = data;
            //std::cout << "Feature map output: " << data << std::endl;
        } else {
            std::cout << "Error: No data available in featureMapOut" << std::endl;
        }
    }
}



int main(){

	//Create two axi-streams to get values to and from module
    hls::stream<bus_t> imageIn;
    hls::stream<bus_t> featureMapOut;

    //Store 32-bit values passed in to and out of module
    bus_t imageValuesTransfer[numOfTransfersIn];
    bus_t featureMapValuesTransfer[numOfTransfersOut];

    //Hardcoded interpolation values for test
	uint8_t input_test1[] =  {
		// Row 0
		10, 20, 30,    // (0,0)
		40, 60, 80,    // (1,0)
		// Row 1
		70, 90, 110,   // (0,1)
		130, 150, 170  // (1,1)
	};


	uint8_t output_test1[] = {
		// Row 0
		10, 20, 30,      // (0,0)
		25, 40, 55,      // (1,0)
		40, 60, 80,      // (2,0)
		40, 60, 80,      // (3,0)

		// Row 1
		40, 55, 70,      // (0,1)
		63, 80, 98,      // (1,1)
		85, 105, 125,    // (2,1)
		85, 105, 125,    // (3,1)

		// Row 2
		70, 90, 110,     // (0,2)
		100, 120, 140,   // (1,2)
		130, 150, 170,   // (2,2)
		130, 150, 170,   // (3,2)

		// Row 3
		70, 90, 110,     // (0,3)
		100, 120, 140,   // (1,3)
		130, 150, 170,   // (2,3)
		130, 150, 170    // (3,3)
	};

    //Assemble 32 bit image transfer values
    for(int i = 0; i < numOfTransfersIn; i++){

    	bus_t temp = 0;

    	//create 32 bit bus value by combining 4, 8 bit values
    	for(int j = 0; j < bytesPerTransfer; j++){
    		uint8_t currentValue = input_test1[i*bytesPerTransfer + j];
			temp = temp << 8;
			temp = temp | currentValue;
    	}

    	imageValuesTransfer[i] = temp;
    }
    //printf("making it through assembly\n");

    // Simulate writing data to the imageIn stream
    write_data(imageIn, imageValuesTransfer);
    //printf("making it through write \n");

    // Call the Interpolation_v4 function to read from imageIn and process the data
    Interpolation_v5(imageIn, featureMapOut);
    //printf("making it through call \n");

    // Now read and print the data from the featureMapOut stream
    read_data(featureMapOut, featureMapValuesTransfer);
    //printf("making it through read \n");

    uint8_t bitMask = 0xFF;
    uint8_t featureMapValues[featureMapSize];


    //disassemble feature map transfer values
    for(int i = 0; i < numOfTransfersOut; i++){
    	//split each 32 bit bus into its 8 bit pieces
        for (int j = 0; j < bytesPerTransfer; j++) {
            featureMapValues[i*bytesPerTransfer + j] = (featureMapValuesTransfer[i] >> ((BUS_WIDTH - 8) - j * 8)) & bitMask;
        }
    }
    //printf("making it through disassemble of feature map \n");

    int errors = 0;

    float scale = 2;
    //std::vector<uint8_t> tbBiLinOutput = bilinearInterpolation(input_test1_vector, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_OF_CHANNELS, scale);


    for(int i = 0; i < featureMapSize; i++){
    	printf("i value: %d, Expected Value: %d, Calculated Value: %d\n", i, output_test1[i], featureMapValues[i]);
    	if(output_test1[i] != featureMapValues[i]){
    		errors++;
    	}
    }

    if(errors){
    	printf("ERROR!\n");
    }
    else{
    	printf("SUCCESS!\n");
    }

}

std::vector<uint8_t> bilinearInterpolation(
    const std::vector<uint8_t>& image,
    int width,
    int height,
    int channels,
    float scale)
{
    // Calculate new dimensions
    int newWidth = static_cast<int>(width * scale);
    int newHeight = static_cast<int>(height * scale);

    // Initialize the output image
    std::vector<uint8_t> outputImage(newWidth * newHeight * channels);

    // For each pixel in the output image
    for (int y_out = 0; y_out < newHeight; ++y_out)
    {
        for (int x_out = 0; x_out < newWidth; ++x_out)
        {
            // Map the pixel to the input image
            float x_in = x_out / scale;
            float y_in = y_out / scale;

            // Find the coordinates of the four neighboring pixels
            int x0 = static_cast<int>(std::floor(x_in));
            int x1 = std::min(x0 + 1, width - 1);
            int y0 = static_cast<int>(std::floor(y_in));
            int y1 = std::min(y0 + 1, height - 1);

            // Calculate the distances between the neighboring pixels
            float dx = x_in - x0;
            float dy = y_in - y0;

            // Compute interpolation weights
            float w00 = (1 - dx) * (1 - dy);
            float w10 = dx * (1 - dy);
            float w01 = (1 - dx) * dy;
            float w11 = dx * dy;

            // For each color channel
            for (int c = 0; c < channels; ++c)
            {
                // Get the values of the four neighboring pixels
                uint8_t p00 = image[(y0 * width + x0) * channels + c];
                uint8_t p10 = image[(y0 * width + x1) * channels + c];
                uint8_t p01 = image[(y1 * width + x0) * channels + c];
                uint8_t p11 = image[(y1 * width + x1) * channels + c];

                // Compute the interpolated pixel value
                float value = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;

                // Clamp the value between 0 and 255
                value = std::min(std::max(value, 0.0f), 255.0f);

                // Round the value to the nearest integer
                uint8_t interpolatedValue = static_cast<uint8_t>(std::round(value));

                // Set the pixel value in the output image
                outputImage[(y_out * newWidth + x_out) * channels + c] = interpolatedValue;
            }
        }
    }
    return outputImage;
}
