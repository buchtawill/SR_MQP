#include "Interpolation_v5.h"

void Interpolation_v5(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut){
	#pragma HLS interface mode=axis port=imageIn
	#pragma HLS interface mode=axis port=featureMapOut


	//Store transfers in 8-bit sections
	uint8_t imageInStoredBytes[imageSize];
	uint8_t imageOutStoredBytes[featureMapSize];

	//Read each transfer in
    for (int i = 0; i < numOfTransfersIn ; i++) {

        // wait until there is data to read
        while (imageIn.empty()) {
            // TO DO: add wait or stall
        }

        bus_t temp = imageIn.read();

        //split read into bytes
        for (int j = 0; j < bytesPerTransfer; j++) {
            uint8_t bitMask = 0xFF;
            imageInStoredBytes[i * bytesPerTransfer + j] = (temp >> ((BUS_WIDTH - 8) - j * 8)) & bitMask;
        }

    }

    //INTERPOLATION BLOCK GOES HERE

    // For each pixel in the output image
    for (int y_out = 0; y_out < featureMapHeight; ++y_out)
    {
        for (int x_out = 0; x_out < featureMapWidth; ++x_out)
        {
            // Map the pixel to the input image
            float x_in = x_out / (static_cast<float>(SCALING_FACTOR));
            float y_in = y_out / (static_cast<float>(SCALING_FACTOR));

            // Find the coordinates of the four neighboring pixels
			int x0 = static_cast<int>(std::floor(x_in));
			int x1 = (std::ceil(x_in) < IMAGE_HEIGHT - 1) ? std::ceil(x_in) : (IMAGE_HEIGHT - 1);
			int y0 = static_cast<int>(std::floor(y_in));
			int y1 = (std::ceil(y_in) < IMAGE_WIDTH - 1) ? std::ceil(y_in) : (IMAGE_WIDTH - 1);

            // Calculate the distances between the neighboring pixels
            float dx = x_in - x0;
            float dy = y_in - y0;

            // Compute interpolation weights
            float w00 = (1 - dx) * (1 - dy);
            float w10 = dx * (1 - dy);
            float w01 = (1 - dx) * dy;
            float w11 = dx * dy;

            // For each color channel
            for (int c = 0; c < NUM_OF_CHANNELS; ++c)
            {
                // Get the values of the four neighboring pixels
                uint8_t p00 = imageInStoredBytes[(y0 * IMAGE_WIDTH + x0) * NUM_OF_CHANNELS + c];
                uint8_t p10 = imageInStoredBytes[(y0 * IMAGE_WIDTH + x1) * NUM_OF_CHANNELS + c];
                uint8_t p01 = imageInStoredBytes[(y1 * IMAGE_WIDTH + x0) * NUM_OF_CHANNELS + c];
                uint8_t p11 = imageInStoredBytes[(y1 * IMAGE_WIDTH + x1) * NUM_OF_CHANNELS + c];

                // Compute the interpolated pixel value
                float value = w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11;

                // Clamp the value between 0 and 255
                value = std::min(std::max(value, 0.0f), 255.0f);

                // Round the value to the nearest integer
                uint8_t interpolatedValue = static_cast<uint8_t>(std::round(value));

                // Set the pixel value in the output image
                imageOutStoredBytes[(y_out * featureMapWidth + x_out) * NUM_OF_CHANNELS + c] = interpolatedValue;
            }
        }
    }
    /*
    for(int i = 0; i < imageSize; i++){
    	for(int j = 0; j < 4; j++){
    		imageOutStoredBytes[i*4 + j] = imageInStoredBytes[i];
    	}
    }*/


    //Write each transfer out
    for (int i = 0; i < numOfTransfersOut; i++){

    	bus_t temp = 0;

    	//values per bus transfer
    	for(int j = 0; j < bytesPerTransfer; j++){
    		uint8_t currentValue = imageOutStoredBytes[i*bytesPerTransfer + j];
			temp = temp << 8;
			temp = temp | currentValue;
    	}

    	//write stored data to featureMap
        featureMapOut.write(temp);
    }


}
