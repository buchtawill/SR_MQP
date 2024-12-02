#include "Interpolation_v4.h"

void Interpolation_v4(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut){
	#pragma HLS interface mode=axis port=imageIn
	#pragma HLS interface mode=axis port=featureMapOut


	//store transfers in 8-bit sections
	//uint8_t imageInStoredBytes[IMAGE_WIDTH * IMAGE_WIDTH * NUM_OF_CHANNELS];
	uint8_t imageInStoredBytes[imageSize]; //2x2x3
	uint8_t imageOutStoredBytes[featureMapSize]; //4x4x3

	//read each transfer in width*height*channels*bitsperpixel / buswidth
    for (int i = 0; i < numOfTransfersIn ; i++) {

        // wait until there is data to read
        while (imageIn.empty()) {
            // TO DO: add wait or stall
        }

        bus_t temp = imageIn.read();

        //bytes per transfer
        for (int j = 0; j < bytesPerTransfer; j++) {
            uint8_t bitMask = 0xFF;
            imageInStoredBytes[i * bytesPerTransfer + j] = (temp >> ((BUS_WIDTH - 8) - j * 8)) & bitMask;
        }

    }


    //INTERPOLATION BLOCK WILL GO HERE

	// Calculate new dimensions
	int newWidth = static_cast<int>(IMAGE_WIDTH * SCALING_FACTOR);
	int newHeight = static_cast<int>(IMAGE_HEIGHT * SCALING_FACTOR);

	// Initialize the output image
	//std::vector<uint8_t> outputImage(newWidth * newHeight * NUM_OF_CHANNELS);

	// For each pixel in the output image
	for (int y_out = 0; y_out < newHeight; ++y_out)
	{
		for (int x_out = 0; x_out < newWidth; ++x_out)
		{
			// Map the pixel to the input image
			float x_in = x_out / SCALING_FACTOR;
			float y_in = y_out / SCALING_FACTOR;

			// Find the coordinates of the four neighboring pixels
			int x0 = static_cast<int>(std::floor(x_in));
			int x1 = std::min(x0 + 1, IMAGE_WIDTH - 1);
			int y0 = static_cast<int>(std::floor(y_in));
			int y1 = std::min(y0 + 1, IMAGE_HEIGHT - 1);

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
				imageOutStoredBytes[(y_out * newWidth + x_out) * NUM_OF_CHANNELS + c] = interpolatedValue;
			}
		}
	}

    /*
    // For each pixel in feature map image
	for (int fmY = 0; fmY < featureMapWidth; ++fmY){
		for (int fmX = 0; fmX < featureMapWidth; ++fmX)
		{
			// Map the pixel to the input image
			int x_in = static_cast<int>(std::round(fmX / SCALING_FACTOR));
			int y_in = static_cast<int>(std::round(fmY / SCALING_FACTOR));
			//float x_in = fmX / scalingFactor;
			//float y_in = fmY / scalingFactor;

			//location in input array of nearest pixels to one being interpolated
			int x0 = static_cast<int>(std::floor(x_in));
			int x1 = std::min(x0 + 1, IMAGE_WIDTH - 1);
			int y0 = static_cast<int>(std::floor(y_in));
			int y1 = std::min(y0 + 1, IMAGE_WIDTH - 1);

			// Calculate the distances between the neighboring pixels -> should always be 1
			int dx = static_cast<int>(std::round(x_in - x0));
			int dy = static_cast<int>(std::round(y_in - y0));
			//float dx = x_in - x0;
			//float dy = y_in - y0;

			// Compute interpolation weights
			int w00 = static_cast<int>(std::round((1 - dx) * (1 - dy)));
			int w10 = static_cast<int>(std::round(dx * (1 - dy)));
			int w01 = static_cast<int>(std::round((1 - dx) * dy));
			int w11 = static_cast<int>(std::round(dx * dy));
			//float w00 = (1 - dx) * (1 - dy);
			//float w10 = dx * (1 - dy);
			//float w01 = (1 - dx) * dy;
			//float w11 = dx * dy;


			// per color per pixel
			for (int c = 0; c < NUM_OF_CHANNELS; ++c)
			{
				// Get the values of the four neighboring pixels
				uint8_t p00 = imageInStoredBytes[(y0 * IMAGE_WIDTH + x0) * NUM_OF_CHANNELS + c];
				uint8_t p10 = imageInStoredBytes[(y0 * IMAGE_WIDTH + x1) * NUM_OF_CHANNELS + c];
				uint8_t p01 = imageInStoredBytes[(y1 * IMAGE_WIDTH + x0) * NUM_OF_CHANNELS + c];
				uint8_t p11 = imageInStoredBytes[(y1 * IMAGE_WIDTH + x1) * NUM_OF_CHANNELS + c];

                // Compute the interpolated pixel value
                int value = static_cast<int>(w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11);

                // Clamp the value between 0 and 255
                //value = std::min(std::max(value, 0.0f), 255.0f);

                // Round the value to the nearest integer
                uint8_t interpolatedValue = static_cast<uint8_t>(std::round(value));

                // Set the pixel value in the output image
                imageOutStoredBytes[(fmY * featureMapWidth + fmX) * NUM_OF_CHANNELS + c] = interpolatedValue;

				// Compute the interpolated pixel value
				//uint8_t interpolatedValue = static_cast<uint8_t>(std::round(w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11));

				// Set the pixel value in the output image
				//imageOutStoredBytes[(fmY * featureMapWidth + fmX) * 3 + c] = interpolatedValue;
			} //end per channel

		}
	} //end bilinear interp calcs*/

    /*
    //num of bytes in input
    for(int i = 0; i < 12; i++){
    	for(int j = 0; j < 4; j++){
    		imageOutStoredBytes[i*4 + j] = imageInStoredBytes[i];
    	}
    } */

    //num of transfers
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
