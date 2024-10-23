void conv_top(int image[], int kernel[], int featureMap[]){
    #pragma HLS INTERFACE s_axilite port=image
    #pragma HLS INTERFACE s_axilite port=feature
    #pragma HLS INTERFACE s_axilite port=featureMap
    
    //assuming kernel is 5x5 and image is 32x32

    //parameterize these later 
    //index of kernel middles
    const int kernelMiddleRow = 2;
    const int kernelMiddleColumn = 2;

    //iterate through image rows
    for(int i = 0; i < 32; i++){

        //iterate through image columns
        for(int j = 0; j < 32; j++){
            
            //create variable to track the pixel value being calculated
            int tempPixel = 0;

            //iterate through kernel rows
            for(int m = 0; m < 5; m++){
                
                //iterates through kernel columns backwards to account for kernel flipping
                for(int n = 4; n >= 0; n--){

                    int kernelIndex = m*5 + n;
                    int imageIndexRow = i + (m - kernelMiddleRow);
                    int imageIndexColumn = j + (n - kernelMiddleColumn);
                    int imageIndex = imageIndexRow*32 + imageIndexColumn;

                    //check to see if current multiplication is in bounds
                    if(imageIndexRow >= 0 && imageIndexRow < 32 && imageIndexColumn >= 0 && imageIndexColumn < 32){
                        tempPixel += image[imageIndex] * kernel[kernelIndex];
                    } 
                } //kernel column iteration
            } //kernel row iteration

            //set feature map value
            int featureMapIndex = i*32 + j;
            featureMap[featureMapIndex] = tempPixel;

        } //image column iteration
    } //image row iteration
} 
