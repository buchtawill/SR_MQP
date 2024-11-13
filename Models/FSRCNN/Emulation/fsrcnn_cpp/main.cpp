#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <utility>
#include <iostream>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <sstream>
#include <thread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include <ratio>
#include <numeric>

namespace fs = std::filesystem;

enum class QuantizationType {
    NONE,
    INT8
};

struct QuantizationParams {
    float scale;
    int64_t zero_point;
};



class QuantizedProcessor {
public:  

/**
 * @brief Calculates optimal quantization parameters for a tensor
 * 
 * @param tensor Input tensor to be quantized
 * @return QuantizationParams Containing scale and zero_point for quantization
 * 
 * This function:
 * 1. Finds min and max values in tensor
 * 2. Calculates optimal scale factor for INT8 range (-127 to 127)
 * 3. Sets zero point to 0 for symmetric quantization
 */
    static QuantizationParams calculate_quantization_params(const torch::Tensor& tensor) {
        float min_val = tensor.min().item<float>();
        float max_val = tensor.max().item<float>();
        
        float abs_max = std::max(std::abs(min_val), std::abs(max_val));
        float scale = abs_max / 127.0f;
        int64_t zero_point = 0;

        return {scale, zero_point};
    }

    /**
 * @brief Quantizes a floating-point tensor to INT8
 * 
 * @param tensor Input tensor (float32)
 * @param params Quantization parameters (scale and zero_point)
 * @return torch::Tensor Quantized tensor (int8)
 * 
 * Process:
 * 1. Scales values using provided scale factor
 * 2. Clamps values to INT8 range (-127 to 127)
 * 3. Rounds to nearest integer
 * 4. Converts to INT8 data type
 */
    static torch::Tensor quantize_tensor(const torch::Tensor& tensor, const QuantizationParams& params) {
        auto scaled = tensor / params.scale;
        auto clamped = scaled.clamp(-127, 127);
        return clamped.round().to(torch::kInt8);
    }

    /**
 * @brief Dequantizes an INT8 tensor back to floating-point
 * 
 * @param qtensor Quantized tensor (int8)
 * @param params Quantization parameters used during quantization
 * @return torch::Tensor Dequantized tensor (float32)
 * 
 * Process:
 * 1. Converts INT8 values to float32
 * 2. Multiplies by scale factor to restore original range
 */

    //static torch::Tensor dequantize_tensor(const torch::Tensor& qtensor, const QuantizationParams& params) {
    //    return qtensor.to(torch::kFloat32) * params.scale;
    //}


/**
 * @brief Quantizes all parameters in a PyTorch model
 * 
 * @param model Input model to be quantized
 * @return torch::jit::Module Quantized model
 * 
 * Process:
 * 1. Sets model to evaluation mode
 * 2. Iterates through all model parameters
 * 3. Quantizes each parameter tensor
 * 4. Immediately dequantizes to maintain float32 interface
 * 5. Updates model parameters with quantized-then-dequantized values
 */
    static torch::jit::Module quantize_model(torch::jit::Module& model) {
        model.eval();
        
        for (const auto& param : model.named_parameters()) {
            auto tensor = param.value;
            auto qparams = calculate_quantization_params(tensor);
            auto quantized = quantize_tensor(tensor, qparams);
            //auto dequantized = dequantize_tensor(quantized, qparams);
            //param.value.set_data(dequantized);
        }
        
        return model;
    }
};


/**
 * @class ProcessingTimer
 * @brief Tracks and displays processing progress
 * 
 * Provides functionality for:
 * - Tracking elapsed time
 * - Estimating remaining time
 * - Calculating processing speed
 * - Formatting progress messages
 */

class ProcessingTimer {
public:
    std::chrono::steady_clock::time_point start_time;

private:
    size_t total_items;
    size_t processed_items;

    /**
     * @brief Formats duration into human-readable string
     * @param duration Duration in milliseconds
     * @return std::string Formatted time string (e.g., "1h 30m 45.123s")
     */

    std::string formatDuration(std::chrono::milliseconds duration) const {
        auto total_ms = duration.count();
        
        auto hours = total_ms / (1000 * 60 * 60);
        total_ms %= (1000 * 60 * 60);
        
        auto minutes = total_ms / (1000 * 60);
        total_ms %= (1000 * 60);
        
        auto seconds = total_ms / 1000;
        auto ms = total_ms % 1000;

        std::stringstream ss;
        if (hours > 0) {
            ss << hours << "h ";
        }
        if (minutes > 0 || hours > 0) {
            ss << minutes << "m ";
        }
        ss << seconds << "." << std::setfill('0') << std::setw(3) << ms << "s";
        return ss.str();
    }

public:
    ProcessingTimer(size_t total) 
        : start_time(std::chrono::steady_clock::now()),
          total_items(total),
          processed_items(0) {}

    void update(size_t current_items) {
        processed_items = current_items;
    }


    /**
     * @brief Generates progress status string
     * @return std::string Formatted progress message
     * 
     * Includes:
     * - Percentage complete
     * - Items processed
     * - Processing speed
     * - Elapsed time
     * - Estimated remaining time
     */

    std::string getProgress() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        
        double progress = static_cast<double>(processed_items) / total_items;
        double items_per_second = static_cast<double>(processed_items * 1000.0) / elapsed.count();
        
        double remaining_items = total_items - processed_items;
        auto estimated_remaining_ms = static_cast<long long>(remaining_items * elapsed.count() / processed_items);
        std::chrono::milliseconds remaining(estimated_remaining_ms);

        std::stringstream ss;
        ss << std::fixed << std::setprecision(1)
           << "Progress: " << (progress * 100) << "% "
           << "(" << processed_items << "/" << total_items << " tiles) | "
           << "Speed: " << items_per_second << " tiles/sec | "
           << "Elapsed: " << formatDuration(elapsed) << " | "
           << "Remaining: " << formatDuration(remaining);
        return ss.str();
    }
};

class AdaptiveUpscaleProcessor {
private:
    struct TileInfo {
        cv::Mat tile;
        cv::Rect position;
        std::string debug_filename;
        cv::Mat upscaled_result;
    };

    int upscale_factor;
    int tile_size;
    std::mutex mtx;
    std::condition_variable cv;
    int max_threads;
    torch::jit::script::Module fsrcnn_model;
    torch::Device device;
    std::unique_ptr<ProcessingTimer> timer;
    QuantizationType quant_type;
    std::vector<QuantizationParams> activation_qparams;


/**
 * @brief Quantizes activation tensors during model inference
 * 
 * @param tensor Input activation tensor to be quantized
 * 
 * Process:
 * 1. Checks if quantization is enabled
 * 2. Calculates quantization parameters
 * 3. Stores parameters for later analysis
 * 4. Applies quantization and dequantization (dequantization was removed)
 * 5. Updates tensor in-place with quantized values
 */

    void quantize_activations(torch::Tensor& tensor) {
        if (quant_type == QuantizationType::INT8) {
            auto qparams = QuantizedProcessor::calculate_quantization_params(tensor);
            activation_qparams.push_back(qparams);
            
            //tensor = QuantizedProcessor::dequantize_tensor(
            //    QuantizedProcessor::quantize_tensor(tensor, qparams),
            //    qparams
            //);
        }
    }

/**
 * @brief Converts OpenCV Mat to PyTorch tensor with optional quantization
 * 
 * @param image Input image in OpenCV Mat format (BGR)
 * @return torch::Tensor Output tensor in NCHW format
 * 
 * Process:
 * 1. Converts image to float32 and normalizes to [0,1]
 * 2. Splits into channels
 * 3. Converts BGR to RGB order
 * 4. Creates tensor on appropriate device (CPU/CUDA)
 * 5. Adds batch dimension
 * 6. Applies quantization if enabled
 */

    torch::Tensor matToTensor(const cv::Mat& image) {
        cv::Mat float_img;
        image.convertTo(float_img, CV_32F, 1.0 / 255.0);
        
        cv::Mat channels[3];
        cv::split(float_img, channels);
        
        auto options = torch::TensorOptions()
            .dtype(torch::kFloat32)
            .device(device);
        
        std::vector<torch::Tensor> tensor_channels;
        for (int i = 2; i >= 0; --i) {
            torch::Tensor channel = torch::from_blob(channels[i].data, 
                {1, channels[i].rows, channels[i].cols}, 
                options).clone();
            tensor_channels.push_back(channel);
        }
        
        auto tensor = torch::cat(tensor_channels, 0);
        auto batched = tensor.unsqueeze(0);
        
        if (quant_type == QuantizationType::INT8) {
            quantize_activations(batched);
        }
        
        return batched;
    }

     /**
     * @brief Converts PyTorch tensor to OpenCV Mat
     * @param tensor Input tensor
     * @return cv::Mat OpenCV image
     * 
     * Converts RGB tensor to BGR image with optional dequantization
     */
    cv::Mat tensorToMat(const torch::Tensor& tensor) {
        auto processed_tensor = tensor;
        if (quant_type == QuantizationType::INT8) {
            quantize_activations(processed_tensor);
        }
        
        auto cpu_tensor = processed_tensor.squeeze().cpu();
        auto accessor = cpu_tensor.accessor<float, 3>();
        
        std::vector<cv::Mat> channels;
        for (int i = 0; i < 3; ++i) {
            cv::Mat channel(accessor.size(1), accessor.size(2), CV_32F);
            for (int h = 0; h < channel.rows; ++h) {
                for (int w = 0; w < channel.cols; ++w) {
                    channel.at<float>(h, w) = accessor[2-i][h][w];
                }
            }
            channels.push_back(channel);
        }
        
        cv::Mat output;
        cv::merge(channels, output);
        output.convertTo(output, CV_8UC3, 255.0);
        return output;
    }

    /**
     * @brief Applies FSRCNN model to upscale image
     * @param input Input image tile
     * @return cv::Mat Upscaled image
     * 
     * Handles tensor conversion and model inference
     */

    cv::Mat upscaleWithFSRCNN(const cv::Mat& input) {
        torch::NoGradGuard no_grad;
        
        try {
            auto input_tensor = matToTensor(input);
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input_tensor);
            auto output_tensor = fsrcnn_model.forward(inputs).toTensor();
            return tensorToMat(output_tensor);
        } catch (const c10::Error& e) {
            throw std::runtime_error("FSRCNN processing error: " + std::string(e.what()));
        }
    }

    /**
     * @brief Saves debug information for a tile
     * @param base_path Base path for output
     * @param tile Tile information
     */

    void saveTileInfo(const std::string& base_path, const TileInfo& tile) {
        std::string info_path = base_path + ".txt";
        std::ofstream info_file(info_path);
        info_file << "Method: FSRCNN\n"
                 << "Position: (" << tile.position.x << "," << tile.position.y << ")\n"
                 << "Size: " << tile.position.width << "x" << tile.position.height;
        info_file.close();
    }


    /**
     * @brief Processes a batch of tiles in parallel
     * @param tiles Vector of tiles
     * @param start Start index
     * @param end End index
     */

    void process_tile_batch(std::vector<TileInfo>& tiles, size_t start, size_t end) {
        for (size_t i = start; i < end && i < tiles.size(); ++i) {
            cv::Mat upscaled = upscaleWithFSRCNN(tiles[i].tile);

            std::stringstream ss_upscaled;
            ss_upscaled << "debug_crops/upscaled/crop_" 
                       << tiles[i].position.y/tile_size << "_" 
                       << tiles[i].position.x/tile_size 
                       << "_upscaled";

            std::vector<int> compression_params;
            compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
            compression_params.push_back(9);

            {
                std::lock_guard<std::mutex> lock(mtx);
                cv::imwrite(ss_upscaled.str() + ".png", upscaled, compression_params);
                saveTileInfo(ss_upscaled.str(), tiles[i]);
            }

            tiles[i].upscaled_result = upscaled;
        }
    }

public:

    /**
     * @brief Constructor
     * @param upscale Upscaling factor
     * @param tile Tile size
     * @param model_path Path to FSRCNN model
     * @param quantization Quantization mode
     * 
     * Initializes processor and loads/quantizes model
     */

    AdaptiveUpscaleProcessor(int upscale = 2, int tile = 28,
                            const std::string& model_path = "fsrcnn_model.pt",
                            QuantizationType quantization = QuantizationType::NONE)
        : upscale_factor(upscale),
          tile_size(tile),
          quant_type(quantization),          // Matches order of declaration,
          device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
        
        max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
        
        try {
            fsrcnn_model = torch::jit::load(model_path);
            fsrcnn_model.to(device);
            fsrcnn_model.eval();
            
            if (quant_type == QuantizationType::INT8) {
                fsrcnn_model = QuantizedProcessor::quantize_model(fsrcnn_model);
                std::cout << "Model quantized to INT8" << std::endl;
            }
            
        } catch (const c10::Error& e) {
            throw std::runtime_error("Error loading/quantizing the FSRCNN model: " + std::string(e.what()));
        }

        std::cout << "Initializing FSRCNN Processor with " 
                  << upscale << "x upscaling, " 
                  << tile << "px tiles\n"
                  << "Quantization: " << (quant_type == QuantizationType::NONE ? "None" : "INT8") << "\n"
                  << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n"
                  << "Using " << max_threads << " threads" << std::endl;
    }

    /**
     * @brief Splits image into tiles for processing
     * @param image_path Path to input image
     * @return pair<vector<TileInfo>, Size> Tiles and target size
     */
    std::pair<std::vector<TileInfo>, cv::Size> crop_image(const std::string& image_path) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;

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
                
                TileInfo tile_info;
                tile_info.tile = image(roi).clone();
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

        return std::make_pair(tiles, cv::Size(width * upscale_factor, height * upscale_factor));
    }


    /**
     * @brief Combines processed tiles into final image
     * @param tiles Vector of processed tiles
     * @param target_size Size of output image
     * @return cv::Mat Final upscaled image
     */

    cv::Mat stitch_tiles(std::vector<TileInfo>& tiles, const cv::Size& target_size) {
        cv::Mat stitched_image(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
        int total_tiles = tiles.size();

        timer = std::make_unique<ProcessingTimer>(total_tiles);

        std::vector<std::thread> threads;
        size_t batch_size = (tiles.size() + max_threads - 1) / max_threads;

        for (size_t i = 0; i < tiles.size(); i += batch_size) {
            size_t end = std::min(i + batch_size, tiles.size());
            threads.emplace_back(&AdaptiveUpscaleProcessor::process_tile_batch, this, std::ref(tiles), i, end);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        for (size_t i = 0; i < tiles.size(); ++i) {
            cv::Rect upscaled_position(
                tiles[i].position.x * upscale_factor,
                tiles[i].position.y * upscale_factor,
                tiles[i].position.width * upscale_factor,
                tiles[i].position.height * upscale_factor
            );

            tiles[i].upscaled_result.copyTo(stitched_image(upscaled_position));

            timer->update(i + 1);
            {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "\r" << timer->getProgress() << std::flush;
            }
        }
        std::cout << std::endl;

        auto processing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - timer->start_time).count();
        double processing_time_sec = processing_time_ms / 1000.0;
        double tiles_per_second = static_cast<double>(total_tiles) / processing_time_sec;

        std::ofstream summary_file("debug_crops/processing_summary.txt");
        summary_file << "Processing Summary\n"
                    << "=================\n"
                    << "Total tiles: " << total_tiles << "\n"
                    << "Processing time: " << processing_time_ms << " ms (" 
                    << std::fixed << std::setprecision(3) << processing_time_sec << " seconds)\n"
                    << "Processing speed: " << std::fixed << std::setprecision(2) 
                    << tiles_per_second << " tiles/second\n"
                    << "Tile size: " << tile_size << "x" << tile_size << " pixels\n"
                    << "Upscale factor: " << upscale_factor << "x\n"
                    << "Original size: " << target_size.width/upscale_factor << "x" << target_size.height/upscale_factor << "\n"
                    << "Final size: " << target_size.width << "x" << target_size.height << "\n"
                    << "Processing device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n"
                    << "Quantization: " << (quant_type == QuantizationType::NONE ? "None" : "INT8") << "\n";
        
        if (quant_type == QuantizationType::INT8) {
            summary_file << "\nQuantization Statistics:\n"
                        << "Number of activation quantizations: " << activation_qparams.size() << "\n"
                        << "Average activation scale: " << std::accumulate(activation_qparams.begin(), 
                                                                         activation_qparams.end(), 0.0f,
                                                                         [](float acc, const QuantizationParams& qp) {
                                                                             return acc + qp.scale;
                                                                         }) / activation_qparams.size() << "\n";
        }
        
        summary_file.close();

        std::cout << "\nProcessing summary:\n"
                  << "Total tiles: " << total_tiles << "\n"
                  << "Total processing time: " << processing_time_ms << " ms (" 
                  << std::fixed << std::setprecision(3) << processing_time_sec << " seconds)\n"
                  << "Average processing speed: " << std::fixed << std::setprecision(2) 
                  << tiles_per_second << " tiles/second\n"
                  << "Quantization: " << (quant_type == QuantizationType::NONE ? "None" : "INT8")
                  << std::endl;

        return stitched_image;
    }

    void cleanup() {
        if (fs::exists("debug_crops")) {
            fs::remove_all("debug_crops");
        }
    }
};

/**
 * @brief Main program entry point
 * @param argc Argument count
 * @param argv Command line arguments
 * @return int Exit code
 * 
 * Handles:
 * - Command line parsing
 * - Processor initialization
 * - Image processing
 * - Output generation
 * - Error handling
 */

int main(int argc, char** argv) {
    try {
        std::string image_path = "img.png";
        QuantizationType quant_type = QuantizationType::NONE;
        
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--image" && i + 1 < argc) {
                image_path = argv[++i];
            } else if (arg == "--quantize") {
                quant_type = QuantizationType::INT8;
            }
        }

        std::cout << "Processing image: " << image_path << std::endl;

        // Create processor instance with quantization option
        AdaptiveUpscaleProcessor processor(2, 28, "fsrcnn_model.pt", quant_type);

        auto result = processor.crop_image(image_path);
        auto& tiles = result.first;
        auto& target_size = result.second;
        
        std::cout << "Created " << tiles.size() << " tiles" << std::endl;
        std::cout << "Target size: " << target_size.width << "x" << target_size.height << std::endl;

        cv::Mat upscaled_image = processor.stitch_tiles(tiles, target_size);
        
        std::string output_path = "final_upscaled.png";
        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9);
        
        cv::imwrite(output_path, upscaled_image, compression_params);
        std::cout << "Saved upscaled image to: " << output_path << std::endl;

        cv::imshow("Original Image", cv::imread(image_path));
        cv::imshow("Upscaled Image", upscaled_image);
        cv::waitKey(0);

        processor.cleanup();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}