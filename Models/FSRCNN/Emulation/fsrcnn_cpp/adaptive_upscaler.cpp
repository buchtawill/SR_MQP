#include "adaptive_upscaler.hpp"
#include <filesystem>
#include <fstream>
#include <thread>
#include <numeric>
#include <torch/cuda.h>  

namespace fs = std::filesystem;

namespace {
    void printMatStats(const cv::Mat& mat, const std::string& name) {
        std::vector<cv::Mat> channels;
        cv::split(mat, channels);
        
        for (size_t i = 0; i < channels.size(); i++) {
            double minVal, maxVal;
            cv::minMaxLoc(channels[i], &minVal, &maxVal);
            std::cout << name << " channel " << i << " range - Min: " << minVal 
                     << ", Max: " << maxVal << std::endl;
        }
        
        cv::Scalar mean = cv::mean(mat);
        std::cout << name << " mean values: [";
        for (int i = 0; i < mat.channels(); i++) {
            if (i > 0) std::cout << ", ";
            std::cout << mean[i];
        }
        std::cout << "]" << std::endl;
    }
}

AdaptiveUpscaleProcessor::AdaptiveUpscaleProcessor(int upscale, int tile,
                                                 const std::string& model_path,
                                                 QuantizationType quantization,
                                                 UpscaleMethod method)
    : device(torch::kCPU),
      upscale_factor(upscale),
      tile_size(tile),
      quant_type(quantization),
      upscale_method(method) {
    
    if (method == UpscaleMethod::FSRCNN) {
        if (torch::cuda::is_available()) {
            device = torch::kCUDA;
            std::cout << "Using CUDA device" << std::endl;
        } else {
            std::cout << "Using CPU device" << std::endl;
        }
        
        try {
            fsrcnn_model = torch::jit::load(model_path);
            fsrcnn_model.to(device);
            fsrcnn_model.eval();
            
            std::cout << "Model loaded successfully from: " << model_path << std::endl;
            
            torch::Tensor test_input = torch::rand({1, 3, tile_size, tile_size}).to(device);
            std::vector<torch::jit::IValue> test_inputs;
            test_inputs.push_back(test_input);
            
            auto test_output = fsrcnn_model.forward(test_inputs).toTensor();
            std::cout << "Model test successful. Output shape: " 
                      << test_output.sizes() << std::endl;
        } catch (const c10::Error& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            throw;
        }
    }
    
    max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    std::cout << "Using " << max_threads << " threads" << std::endl;
    std::cout << "Upscaling method: " << (method == UpscaleMethod::FSRCNN ? "FSRCNN" : "Bilinear") << std::endl;
}

torch::Tensor AdaptiveUpscaleProcessor::matToTensor(const cv::Mat& image) {
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    
    cv::Mat float_img;
    rgb_image.convertTo(float_img, CV_32F, 1.0/255.0);
    
    auto tensor = torch::from_blob(
        float_img.data,
        {1, float_img.rows, float_img.cols, 3},
        torch::kFloat32
    ).clone();
    
    tensor = tensor.permute({0, 3, 1, 2});
    tensor = tensor.to(device);
    
    return tensor;
}

cv::Mat AdaptiveUpscaleProcessor::tensorToMat(const torch::Tensor& tensor) {
    auto processed_tensor = tensor.detach().cpu();
    processed_tensor = processed_tensor.squeeze(0);
    processed_tensor = processed_tensor.permute({1, 2, 0});
    processed_tensor = processed_tensor.contiguous();
    
    processed_tensor = processed_tensor.clamp(0, 1);
    processed_tensor = processed_tensor.mul(255.0);
    
    cv::Mat output(
        processed_tensor.size(0),
        processed_tensor.size(1),
        CV_32FC3,
        processed_tensor.data_ptr<float>()
    );
    
    cv::Mat bgr_output;
    cv::cvtColor(output, bgr_output, cv::COLOR_RGB2BGR);
    
    cv::Mat final_output;
    bgr_output.convertTo(final_output, CV_8UC3);
    
    return final_output;
}

cv::Mat AdaptiveUpscaleProcessor::upscaleWithFSRCNN(const cv::Mat& input) {
    torch::NoGradGuard no_grad;
    
    try {
        // Ensure input is the correct size
        cv::Mat resized_input;
        if (input.rows != tile_size || input.cols != tile_size) {
            cv::resize(input, resized_input, cv::Size(tile_size, tile_size));
        } else {
            resized_input = input;
        }
        
        auto input_tensor = matToTensor(resized_input);
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        
        auto output_tensor = fsrcnn_model.forward(inputs).toTensor();
        
        cv::Mat output = tensorToMat(output_tensor);
        
        // Resize back to the expected size if needed
        cv::Mat final_output;
        if (input.rows != tile_size || input.cols != tile_size) {
            cv::resize(output, final_output, cv::Size(input.cols * upscale_factor, 
                                                    input.rows * upscale_factor));
        } else {
            final_output = output;
        }
        
        return final_output;
    } catch (const c10::Error& e) {
        std::cerr << "FSRCNN processing error: " << e.what() << std::endl;
        throw;
    }
}

cv::Mat AdaptiveUpscaleProcessor::upscaleWithBilinear(const cv::Mat& input) {
    cv::Mat output;
    cv::Size target_size(input.cols * upscale_factor, input.rows * upscale_factor);
    cv::resize(input, output, target_size, 0, 0, cv::INTER_LINEAR);
    return output;
}

std::pair<std::vector<AdaptiveUpscaleProcessor::TileInfo>, cv::Size> 
AdaptiveUpscaleProcessor::crop_image(const std::string& image_path) {
    cv::Mat image = cv::imread(image_path);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;
    printMatStats(image, "Original image");

    fs::create_directories("debug_crops");
    fs::create_directories("debug_crops/original");
    fs::create_directories("debug_crops/upscaled");

    int width = image.cols;
    int height = image.rows;
    std::vector<TileInfo> tiles;
    tiles.reserve((height / tile_size + 1) * (width / tile_size + 1));

    for (int i = 0; i < height; i += tile_size) {
        for (int j = 0; j < width; j += tile_size) {
            int actual_width = std::min(tile_size, width - j);
            int actual_height = std::min(tile_size, height - i);

            cv::Rect roi(j, i, actual_width, actual_height);
            cv::Mat tile = image(roi).clone();
            
            // Pad if needed
            if (actual_width < tile_size || actual_height < tile_size) {
                cv::Mat padded;
                cv::copyMakeBorder(tile, padded, 0, tile_size - actual_height, 0, 
                                 tile_size - actual_width, cv::BORDER_REFLECT);
                tile = padded;
            }
            
            TileInfo tile_info;
            tile_info.tile = tile;
            tile_info.position = roi;
            
            std::stringstream ss_debug;
            ss_debug << "debug_crops/original/crop_" 
                    << i/tile_size << "_" 
                    << j/tile_size;
            
            tile_info.debug_filename = ss_debug.str();

            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);

            {
                std::lock_guard<std::mutex> lock(mtx);
                cv::imwrite(tile_info.debug_filename + ".png", tile_info.tile, compression_params);
                saveTileInfo(tile_info.debug_filename, tile_info);
                tiles.push_back(tile_info);
            }
        }
    }

    std::cout << "Dividing into " << (width + tile_size - 1)/tile_size << "x" 
              << (height + tile_size - 1)/tile_size << " tiles" << std::endl;

    return std::make_pair(tiles, cv::Size(width * upscale_factor, height * upscale_factor));
}

void AdaptiveUpscaleProcessor::process_tile_batch(std::vector<TileInfo>& tiles, size_t start, size_t end) {
    for (size_t i = start; i < end && i < tiles.size(); ++i) {
        try {
            // Get original tile dimensions before upscaling
            int orig_width = tiles[i].position.width;
            int orig_height = tiles[i].position.height;
            
            cv::Mat upscaled;
            if (upscale_method == UpscaleMethod::FSRCNN) {
                upscaled = upscaleWithFSRCNN(tiles[i].tile);
                
                // Ensure the upscaled result matches the expected dimensions
                if (upscaled.cols != orig_width * upscale_factor || 
                    upscaled.rows != orig_height * upscale_factor) {
                    cv::resize(upscaled, upscaled, 
                             cv::Size(orig_width * upscale_factor, 
                                    orig_height * upscale_factor));
                }
            } else {
                upscaled = upscaleWithBilinear(tiles[i].tile);
            }
            
            std::string debug_path = "debug_crops/upscaled/tile_" + 
                                   std::to_string(tiles[i].position.y/tile_size) + "_" + 
                                   std::to_string(tiles[i].position.x/tile_size) + ".png";
            cv::imwrite(debug_path, upscaled);
            
            tiles[i].upscaled_result = upscaled;
        } catch (const std::exception& e) {
            std::cerr << "Error processing tile " << i << ": " << e.what() << std::endl;
        }
    }
}

cv::Mat AdaptiveUpscaleProcessor::stitch_tiles(std::vector<TileInfo>& tiles, const cv::Size& target_size) {
    cv::Mat stitched_image(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
    int total_tiles = tiles.size();

    timer = std::make_unique<ProcessingTimer>(total_tiles);

    std::vector<std::thread> threads;
    size_t batch_size = (tiles.size() + max_threads - 1) / max_threads;

    // Process tiles in batches
    for (size_t i = 0; i < tiles.size(); i += batch_size) {
        size_t end = std::min(i + batch_size, tiles.size());
        threads.emplace_back(&AdaptiveUpscaleProcessor::process_tile_batch, this, std::ref(tiles), i, end);
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Stitch tiles together
    for (size_t i = 0; i < tiles.size(); ++i) {
        cv::Rect upscaled_position(
            tiles[i].position.x * upscale_factor,
            tiles[i].position.y * upscale_factor,
            tiles[i].position.width * upscale_factor,
            tiles[i].position.height * upscale_factor
        );

        // Ensure we don't exceed the target image boundaries
        upscaled_position.width = std::min(upscaled_position.width, 
                                         target_size.width - upscaled_position.x);
        upscaled_position.height = std::min(upscaled_position.height, 
                                          target_size.height - upscaled_position.y);

        // Create ROI and copy tile
        cv::Mat roi = stitched_image(upscaled_position);
        cv::Mat upscaled_roi = tiles[i].upscaled_result(cv::Rect(0, 0, roi.cols, roi.rows));
        upscaled_roi.copyTo(roi);

        timer->update(i + 1);
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "\r" << timer->getProgress() << std::flush;
        }
    }
    std::cout << std::endl;
    printMatStats(stitched_image, "Final stitched image");

    return stitched_image;
}

void AdaptiveUpscaleProcessor::saveTileInfo(const std::string& base_path, const TileInfo& tile) {
    std::string info_path = base_path + ".txt";
    std::ofstream info_file(info_path);
    
    if (info_file.is_open()) {
        info_file << "Position: (" << tile.position.x << "," << tile.position.y << ")\n"
                 << "Size: " << tile.position.width << "x" << tile.position.height;
        info_file.close();
    } else {
        std::cerr << "Warning: Could not save tile info to " << info_path << std::endl;
    }
}

void AdaptiveUpscaleProcessor::cleanup() {
    if (fs::exists("debug_crops")) {
        fs::remove_all("debug_crops");
    }
}

void AdaptiveUpscaleProcessor::save_quantized_weights(const std::string& output_path) {
    if (quant_type != QuantizationType::INT8) {
        throw std::runtime_error("Cannot save quantized weights: model is not quantized");
    }
    QuantizedProcessor::save_quantized_weights(fsrcnn_model, output_path);
}

void AdaptiveUpscaleProcessor::save_model(const std::string& output_path) {
    if (quant_type != QuantizationType::INT8) {
        throw std::runtime_error("Cannot save model: model is not quantized");
    }
    QuantizedProcessor::save_model_to_pth(fsrcnn_model, output_path);
}