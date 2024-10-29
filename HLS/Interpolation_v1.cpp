#include <iostream>
#include <cmath>

//based off of this https://gist.github.com/folkertdev/6b930c7a7856e36dcad0a72a03e66716
void interp_top(int image[], int featureMap[]){
    const int imageWidth = 28;
    const int imageHeight = 28;
    const int featureMapWidth = 56;
    const int featureMapHeight = 56;

    float x_ratio = (imageWidth - 1) / (featureMapWidth - 1);
    float y_ratio = (imageHeight - 1) / (featureMapHeight - 1);

    //create output values
    //iterate through each of the output rows
    for(int i = 0; i < featureMapHeight; i++){
        //iterate through each of the output columns
        for(int j = 0; j < featureMapWidth; j++){

                        float x_l = floor(x_ratio * (float)j);
            float x_h = ceil(x_ratio * (float)j);

            float y_l = floor(x_ratio * (float)i);
            float y_h = ceil(x_ratio * (flat)i);

            float x_weight = (x_ratio * (float)j) - x_l;
            float y_weight = (y_ratio * (float)i) - y_l;

            float a = image[(int)y_l * imageWidth + int(x_l)];
            float b = image[(int)y_l * imageWidth + int(x_h)];
            float c = image[(int)y_h * imageWidth + int(x_l)];
            float d = image[(int)y_h * imageWidth + int(x_h)];

            float pixel = (a + b + c + d)/4;

            featureMap[i * featureMapWidth + j] = pixel;

        }
    }



    
}