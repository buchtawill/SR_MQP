#include "Interpolation_v4.h"

//std::vector<uint8_t> bilinearInterpolation(uint8_t (&image)[imageSize], int width, int height, int channels, float scale);
std::vector<uint8_t> bilinearInterpolation(const std::vector<uint8_t>& image, int width, int height, int channels, float scale);

// Function to simulate writing data to the stream
//imageValues is an array length of how many transfers needed (num of pixels per image (W*H*3) / (bus width/8))
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

int main() {
	//printf("start of simulation \n");
    // Create two streams to simulate the HLS streams
    hls::stream<bus_t> imageIn;
    hls::stream<bus_t> featureMapOut;

    bus_t imageValuesTransfer[numOfTransfersIn];
    bus_t featureMapValuesTransfer[numOfTransfersOut];


    //uint8_t imageValues[imageSize];
    uint8_t featureMapValues[featureMapSize];

    /*
    uint8_t imageValuesArray[] =  {
        10, 10, 10,
        20, 20, 20,
        30, 30, 30,
        40, 40, 40
        };*/

    uint8_t imageValuesArray[] =  {
            10, 20, 30, 40
            };

    std::vector<uint8_t> imageValuesVector =  {
        255, 0, 0,
        0, 255, 0,
        0, 0, 255,
        255, 255, 255
        };

/*
    uint8_t output_test1[] = {
        // Row 0
        10, 10, 10,        // (0,0)
        12, 12, 12,      // (1,0)
        17, 17, 17,        // (2,0)
        20, 20, 20,        // (3,0)

        // Row 1
        15, 15, 15,      // (0,1)
        17, 17, 17,    // (1,1)
        22, 22, 22,    // (2,1)
        25, 25, 25,    // (3,1)

        // Row 2
        25, 25, 25,        // (0,2)
        27, 27, 27,    // (1,2)
        32, 32, 32,    // (2,2)
        35, 35, 35,    // (3,2)

        // Row 3
        30, 30, 30,        // (0,3)
        32, 32, 32,    // (1,3)
        37, 37, 37,    // (2,3)
        40, 40, 40     // (3,3)
    }; */


    uint8_t output_test1[] = {
        // Row 0
        10, 12, 17, 20,

        // Row 1
        15, 17, 22, 25,

        // Row 2
        25, 27, 32, 35,

        // Row 3
        30, 32, 37, 40
    };

    uint8_t input_test2[] = {
        // Row 0
        10, 20, 30,    // (0,0)
        40, 60, 80,    // (1,0)
        // Row 1
        70, 90, 110,   // (0,1)
        130, 150, 170  // (1,1)
    };

    uint8_t output_test2[] = {
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

    uint8_t input_test3[] = {
        // Row 0
        200, 50, 25,     // (0,0)
        100, 150, 75,    // (1,0)

        // Row 1
        50, 200, 125,    // (0,1)
        25, 100, 175     // (1,1)
    };

    uint8_t output_test3[] = {
        // Row 0
        200, 50, 25,     // (0,0)
        150, 100, 50,    // (1,0)
        100, 150, 75,    // (2,0)
        100, 150, 75,    // (3,0)

        // Row 1
        125, 125, 75,    // (0,1)
        94, 125, 100,    // (1,1)
        63, 125, 125,    // (2,1)
        63, 125, 125,    // (3,1)

        // Row 2
        50, 200, 125,    // (0,2)
        38, 150, 150,    // (1,2)
        25, 100, 175,    // (2,2)
        25, 100, 175,    // (3,2)

        // Row 3
        50, 200, 125,    // (0,3)
        38, 150, 150,    // (1,3)
        25, 100, 175,    // (2,3)
        25, 100, 175     // (3,3)
    };

    uint8_t input_test4[] = {
        // Row 0
        15, 45, 75,    // (0,0)
        85, 115, 145,  // (1,0)

        // Row 1
        155, 185, 215, // (0,1)
        225, 255, 35   // (1,1)
    };

    uint8_t output_test4[] = {
        // Row 0
        15, 45, 75,      // (0,0)
        50, 80, 110,     // (1,0)
        85, 115, 145,    // (2,0)
        85, 115, 145,    // (3,0)

        // Row 1
        85, 115, 145,    // (0,1)
        120, 150, 118,   // (1,1)
        155, 185, 90,    // (2,1)
        155, 185, 90,    // (3,1)

        // Row 2
        155, 185, 215,   // (0,2)
        190, 220, 125,   // (1,2)
        225, 255, 35,    // (2,2)
        225, 255, 35,    // (3,2)

        // Row 3
        155, 185, 215,   // (0,3)
        190, 220, 125,   // (1,3)
        225, 255, 35,    // (2,3)
        225, 255, 35     // (3,3)
    };


    uint8_t input_test5[] = {
        // Row 0
        10, 20, 30,    // (0,0) - Top-left pixel
        40, 50, 60,    // (1,0) - Top-right pixel

        // Row 1
        70, 80, 90,    // (0,1) - Bottom-left pixel
        100, 110, 120  // (1,1) - Bottom-right pixel
    };

    uint8_t output_test5[] = {
        // Row 0
        10, 20, 30,    // (0,0)
        25, 35, 45,    // (1,0)
        40, 50, 60,    // (2,0)
        40, 50, 60,    // (3,0)

        // Row 1
        40, 50, 60,    // (0,1)
        55, 65, 75,    // (1,1)
        70, 80, 90,    // (2,1)
        70, 80, 90,    // (3,1)

        // Row 2
        70, 80, 90,    // (0,2)
        85, 95, 105,   // (1,2)
        100, 110, 120, // (2,2)
        100, 110, 120, // (3,2)

        // Row 3
        70, 80, 90,    // (0,3)
        85, 95, 105,   // (1,3)
        100, 110, 120, // (2,3)
        100, 110, 120  // (3,3)
    };

    /*
    for(int i = 0; i < featureMapSize; i++){
    	uint8_t randValue = rand() % 256;
    	imageValues[i] = randValue;
    } */


    //assemble 16 bit image transfer values
    for(int i = 0; i < numOfTransfersIn; i++){

    	bus_t temp = 0;

    	//create 16 bit bus value by combining 2, 8 bit values
    	for(int j = 0; j < bytesPerTransfer; j++){
    		//std::cout << "Random Value: " << (int)randomVal << " ";
    		uint8_t currentValue = imageValuesArray[i*bytesPerTransfer + j];
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
    Interpolation_v4(imageIn, featureMapOut);
    //printf("making it through call \n");

    // Now read and print the data from the featureMapOut stream
    read_data(featureMapOut, featureMapValuesTransfer);
    //printf("making it through read \n");

    uint8_t bitMask = 0xFF;

    //disassemble feature map transfer values
    for(int i = 0; i < numOfTransfersOut; i++){
    	//split each 128 bit bus into its 8 bit pieces
        for (int j = 0; j < bytesPerTransfer; j++) {
            featureMapValues[i*bytesPerTransfer + j] = (featureMapValuesTransfer[i] >> ((BUS_WIDTH - 8) - j * 8)) & bitMask;
        }
    }
    printf("making it through disassemble of feature map \n");



    float scale = 2;
    std::vector<uint8_t> tbBiLinOutput = bilinearInterpolation(imageValuesVector, IMAGE_WIDTH, IMAGE_HEIGHT, NUM_OF_CHANNELS, scale);

    uint8_t timesFourValues[48];


    /*
    for(int i = 0; i < 12; i++){
    	for(int j = 0; j < 4; j++){
    		timesFourValues[i*4 + j] = imageValues[i];
    	}
    } */

    //compare module values to ones from test bench
    int errors = 0;
    for(int i = 0; i < featureMapSize; i++){

    	printf("Iteration number %d, Module value: %d, TB value: %d\n", i, featureMapValues[i], output_test1[i]);

    	if(featureMapValues[i] != output_test1[i]){
    		errors++;
    		//printf("ERROR! Module value: %d, TB value: %d\n", featureMapValues[i], tbBiLinOutput[i]);
    		//std::cout << "ERROR!"<< std::endl;
    	}

    	if(errors >= 36){
    		break;
    	}
    }

    if(errors == 0){
    	std::cout << "Completed without errors!" << std::endl;
    }
    else{
    	std::cout << "Has errors" << std::endl;
    }

    return 0;
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

/*
std::vector<uint8_t> bilinearInterpolation(
	uint8_t (&image)[imageSize],
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
*/

