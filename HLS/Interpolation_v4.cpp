#include "Interpolation_v4.h"

void Interpolation_v4(hls::stream<bus_t> &imageIn, hls::stream<bus_t> &featureMapOut){
	#pragma HLS interface mode=axis port=imageIn
	#pragma HLS interface mode=axis port=featureMapOut

	const int inputWidth = 28;
	const int numOfChannels = 3;
	const int numOfTransfers = inputWidth * inputWidth * numOfChannels * 8 / BUS_WIDTH;
	const int scalingFactor = 2;
	const int featureMapWidth = inputWidth * scalingFactor;


	//store 128-bit transfers
	bus_t allImageInStored[147];
	bus_t allImageOutStored[588];

	//store in 8-bit sections
	uint8_t imageInStoredBytes[2352];
	uint8_t imageOutStoredBytes[9408];

    // 147 transfers are needed to get the full image
    for (int i = 0; i < 147; i++) {

        // wait until there is data to read
        while (imageIn.empty()) {
            // TO DO: add wait or stall
        }

        // read from stream
        //allImageInStored[i] = imageIn.read();

        bus_t temp = imageIn.read();

        for (int j = 0; j < 16; j++) {
            uint8_t bitMask = 0xFF;
            imageInStoredBytes[i * 16 + j] = (temp >> (120 - j * 8)) & bitMask;
        }

        /*
        //16 bytes per transfer
        for(int j = 0; j < 16; j++){
        	uint8_t bitMask = 0xFF;
        	bus_t shifted = temp >> (120 - j*8);
        	shifted = shifted & bitMask;
        	imageInStoredBytes[i*16 + j] = shifted;
        }
        */
    }

    //INTERPOLATION BLOCK WILL GO HERE
    /*
    for(int i = 0; i < 9408; i++){
    	imageOutStoredBytes[i] = imageInStoredBytes[static_cast<int>(floor(i / 4))];
    }
    */
    // For each pixel in feature map image
	for (int fmY = 0; fmY < featureMapWidth; ++fmY){
		for (int fmX = 0; fmX < featureMapWidth; ++fmX)
		{
			// Map the pixel to the input image
			int x_in = static_cast<int>(std::round(fmX / scalingFactor));
			int y_in = static_cast<int>(std::round(fmY / scalingFactor));
			//float x_in = j / upscalingFactor;
			//float y_in = i / upscalingFactor;

			//location in input array of nearest pixels to one being interpolated
			int x0 = static_cast<int>(std::floor(x_in));
			int x1 = std::min(x0 + 1, inputWidth - 1);
			int y0 = static_cast<int>(std::floor(y_in));
			int y1 = std::min(y0 + 1, inputWidth - 1);

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


			// per color (3 colors) per pixel
			for (int c = 0; c < 3; ++c)
			{
				// Get the values of the four neighboring pixels
				uint8_t p00 = imageInStoredBytes[(y0 * inputWidth + x0) * 3 + c];
				uint8_t p10 = imageInStoredBytes[(y0 * inputWidth + x1) * 3 + c];
				uint8_t p01 = imageInStoredBytes[(y1 * inputWidth + x0) * 3 + c];
				uint8_t p11 = imageInStoredBytes[(y1 * inputWidth + x1) * 3 + c];

				// Compute the interpolated pixel value
				uint8_t interpolatedValue = static_cast<uint8_t>(std::round(w00 * p00 + w10 * p10 + w01 * p01 + w11 * p11));

				// Set the pixel value in the output image
				imageOutStoredBytes[(fmY * featureMapWidth + fmX) * 3 + c] = interpolatedValue;
			} //end per channel


			/*
			// Get the values of the four neighboring pixels for the Red channel (c = 0)
			uint8_t p00_r = imageInStoredBytes[(y0 * inputWidth + x0) * 3 + 0];
			uint8_t p10_r = imageInStoredBytes[(y0 * inputWidth + x1) * 3 + 0];
			uint8_t p01_r = imageInStoredBytes[(y1 * inputWidth + x0) * 3 + 0];
			uint8_t p11_r = imageInStoredBytes[(y1 * inputWidth + x1) * 3 + 0];

			// Compute the interpolated Red pixel value
			uint8_t interpolatedValue_r = static_cast<uint8_t>(std::round(w00 * p00_r + w10 * p10_r + w01 * p01_r + w11 * p11_r));

			// Set the Red channel pixel value in the output image
			imageOutStoredBytes[(fmY * featureMapWidth + fmX) * 3 + 0] = interpolatedValue_r;

			// Get the values of the four neighboring pixels for the Green channel (c = 1)
			uint8_t p00_g = imageInStoredBytes[(y0 * inputWidth + x0) * 3 + 1];
			uint8_t p10_g = imageInStoredBytes[(y0 * inputWidth + x1) * 3 + 1];
			uint8_t p01_g = imageInStoredBytes[(y1 * inputWidth + x0) * 3 + 1];
			uint8_t p11_g = imageInStoredBytes[(y1 * inputWidth + x1) * 3 + 1];

			// Compute the interpolated Green pixel value
			uint8_t interpolatedValue_g = static_cast<uint8_t>(std::round(w00 * p00_g + w10 * p10_g + w01 * p01_g + w11 * p11_g));

			// Set the Green channel pixel value in the output image
			imageOutStoredBytes[(fmY * featureMapWidth + fmX) * 3 + 1] = interpolatedValue_g;

			// Get the values of the four neighboring pixels for the Blue channel (c = 2)
			uint8_t p00_b = imageInStoredBytes[(y0 * inputWidth + x0) * 3 + 2];
			uint8_t p10_b = imageInStoredBytes[(y0 * inputWidth + x1) * 3 + 2];
			uint8_t p01_b = imageInStoredBytes[(y1 * inputWidth + x0) * 3 + 2];
			uint8_t p11_b = imageInStoredBytes[(y1 * inputWidth + x1) * 3 + 2];

			// Compute the interpolated Blue pixel value
			uint8_t interpolatedValue_b = static_cast<uint8_t>(std::round(w00 * p00_b + w10 * p10_b + w01 * p01_b + w11 * p11_b));

			// Set the Blue channel pixel value in the output image
			imageOutStoredBytes[(fmY * featureMapWidth + fmX) * 3 + 2] = interpolatedValue_b;
			*/
		}
	} //end bilinear interp calcs


    for (int i = 0; i < 588; i++){

    	bus_t temp = 0;

    	//combines 8 bytes from input image for featureMap output
    	for(int j = 0; j < 16; j++){
			//gets random 8 bit num and adds it to temp value
    		uint8_t pixelValue = imageOutStoredBytes[i*16 + j];
    		//std::cout << "Random Value: " << (int)randomVal << " ";
			temp = temp << 8;
			temp = temp | pixelValue;
    	}

    	//write stored data to featureMap
        featureMapOut.write(temp);
    }

}
