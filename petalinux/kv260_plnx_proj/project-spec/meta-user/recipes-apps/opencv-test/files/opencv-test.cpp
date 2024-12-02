/*
* Placeholder PetaLinux user application.
*
* Replace this with your application code

* Copyright (C) 2013-2022  Xilinx, Inc.  All rights reserved.
* Copyright (c) 2022-2023 Advanced Micro Devices, Inc. All Rights Reserved.
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in this
* Software without prior written authorization from Xilinx.
*
*/

#include <iostream>

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cstdlib> // For rand()

int main() {
    // Create a 256x256 image with 3 channels (BGR)
    cv::Mat randomImage(256, 256, CV_8UC3);

    // Fill the image with random colors
    for (int i = 0; i < randomImage.rows; ++i) {
        for (int j = 0; j < randomImage.cols; ++j) {
            randomImage.at<cv::Vec3b>(i, j) = cv::Vec3b(rand() % 256, rand() % 256, rand() % 256);
        }
    }

    // Save the image to a file
    std::string filename = "random_image.png";
    if (cv::imwrite(filename, randomImage)) {
        std::cout << "Image saved successfully as " << filename << std::endl;
    } else {
        std::cerr << "Error: Could not save the image!" << std::endl;
        return -1;
    }

    return 0;
}


