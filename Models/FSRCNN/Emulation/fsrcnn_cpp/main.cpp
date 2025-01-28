
#include "adaptive_upscaler.hpp"
#include <iostream>
#include <string>
#include <filesystem>

namespace fs = std::filesystem;

/**
 * @brief Main program for image upscaling using FSRCNN
 * @param argc Number of command line arguments
 * @param argv Array of command line arguments
 * @return 0 on success, 1 on error
 * 
 * Command line options:
 * --image <path>: Specify input image path (default: img.png)
 * --quantize: Enable INT8 quantization
 * --save-model [path]: Save quantized model to specified path (default: fsrcnn_quantized.pth)
 * 
 * Program flow:
 * 1. Parse command line arguments
 * 2. Initialize upscaler with specified options
 * 3. Save quantized model if enabled
 * 4. Process and upscale image
 * 5. Display and save results
 */
int main(int argc, char** argv) {
    try {
        // Default parameters
        std::string image_path = "img.png";
        QuantizationType quant_type = QuantizationType::NONE;
        std::string output_weights = "quantized_weights";
        std::string model_output = "fsrcnn_quantized.pth";
        bool save_model = false;
        UpscaleMethod upscale_method = UpscaleMethod::FSRCNN;
        
        // Parse command line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc) {
                image_path = argv[++i];
            } else if (arg == "--quantize") {
                quant_type = QuantizationType::INT8;
            } else if (arg == "--save-model") {
                save_model = true;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    model_output = argv[++i];
                }
            } else if (arg == "--method") {
                if (i + 1 < argc) {
                    std::string method = argv[++i];
                    if (method == "bilinear") {
                        upscale_method = UpscaleMethod::BILINEAR;
                    } else if (method == "fsrcnn") {
                        upscale_method = UpscaleMethod::FSRCNN;
                    } else {
                        std::cerr << "Invalid upscaling method. Using default (FSRCNN)" << std::endl;
                    }
                }
            }
        }

        // Validate input image existence
        if (!fs::exists(image_path)) {
            std::cerr << "Error: Image file not found: " << image_path << std::endl;
            return 1;
        }

        // Load and display original image
        cv::Mat original = cv::imread(image_path);
        if (original.empty()) {
            std::cerr << "Error: Could not read image: " << image_path << std::endl;
            return 1;
        }

        // Create a named window for the original image
        cv::namedWindow("Original Image", cv::WINDOW_NORMAL);
        cv::imshow("Original Image", original);

        std::cout << "Processing image: " << image_path << std::endl;

        // Initialize upscaling processor with selected method
        AdaptiveUpscaleProcessor processor(2, 28, "fsrcnn_model.pt", quant_type, upscale_method);

        try {
            // Process image
            auto result = processor.crop_image(image_path);
            auto& tiles = result.first;
            auto& target_size = result.second;
            
            if (tiles.empty()) {
                std::cerr << "Error: No tiles were created from the image" << std::endl;
                return 1;
            }
            
            std::cout << "Created " << tiles.size() << " tiles" << std::endl;
            std::cout << "Target size: " << target_size.width << "x" << target_size.height << std::endl;

            // Process tiles and combine into final image
            cv::Mat upscaled_image = processor.stitch_tiles(tiles, target_size);
            
            if (upscaled_image.empty()) {
                std::cerr << "Error: Failed to create upscaled image" << std::endl;
                return 1;
            }

            // Save upscaled image
            std::string output_path = "final_upscaled.png";
            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);
            
            cv::imwrite(output_path, upscaled_image, compression_params);
            std::cout << "Saved upscaled image to: " << output_path << std::endl;

            // Create a named window for the upscaled image
            cv::namedWindow("Upscaled Image", cv::WINDOW_NORMAL);
            cv::imshow("Upscaled Image", upscaled_image);

            // Resize windows to fit screen
            cv::resizeWindow("Original Image", original.cols/2, original.rows/2);
            cv::resizeWindow("Upscaled Image", upscaled_image.cols/2, upscaled_image.rows/2);

            // Move windows to non-overlapping positions
            cv::moveWindow("Original Image", 50, 50);
            cv::moveWindow("Upscaled Image", 50 + original.cols/2 + 30, 50);

            // Wait for a key press before closing windows
            std::cout << "Press any key to close the images..." << std::endl;
            cv::waitKey(0);

            // Cleanup and close windows
            cv::destroyAllWindows();
            processor.cleanup();

        } catch (const std::exception& e) {
            std::cerr << "Error during image processing: " << e.what() << std::endl;
            return 1;
        }

        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    }
}