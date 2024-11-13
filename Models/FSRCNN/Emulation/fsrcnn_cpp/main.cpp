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
#include <pthread>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <iomanip>
#include <fstream>
#include <chrono>
#include <ctime>
#include <ratio>

namespace fs = std::filesystem;

/**
 * @brief Timer class for tracking processing progress and estimating remaining time
 * 
 * Provides functionality to:
 * - Track total and processed items
 * - Calculate processing speed
 * - Estimate remaining time
 * - Format durations in human-readable format (HH:MM:SS.mmm)
 */
class ProcessingTimer {
public:
    std::chrono::steady_clock::time_point start_time;

private:
    size_t total_items;
    size_t processed_items;

    // Helper function for formatting milliseconds
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

    std::string getProgress() const {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - start_time);
        
        double progress = static_cast<double>(processed_items) / total_items;
        double items_per_second = static_cast<double>(processed_items * 1000.0) / elapsed.count();
        
        // Calculate estimated time remaining
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

/**
 * @brief Main processor class for adaptive image upscaling using FSRCNN and KNN
 * 
 * This class implements a hybrid upscaling approach that:
 * - Splits input image into tiles
 * - Analyzes each tile's complexity using variance
 * - Applies FSRCNN to complex tiles and KNN to simple tiles
 * - Processes tiles in parallel using multiple threads
 * - Provides progress tracking and timing information
 */

class AdaptiveUpscaleProcessor {
private:
    // Struct definition
    struct TileInfo {
        cv::Mat tile;
        cv::Rect position;
        std::string debug_filename;
        cv::Mat upscaled_result;
        double variance;
        bool needs_fsrcnn;
    };

    // Member variables
    int upscale_factor;
    int tile_size;
    double variance_threshold;
    std::mutex mtx;
    std::condition_variable cv;
    int max_threads;
    torch::jit::script::Module fsrcnn_model;
    torch::Device device;
    std::unique_ptr<ProcessingTimer> timer;

    // Private member function declarations
    torch::Tensor matToTensor(const cv::Mat& image);
    cv::Mat tensorToMat(const torch::Tensor& tensor);
    cv::Mat upscaleWithFSRCNN(const cv::Mat& input);
    double calculateVariance(const cv::Mat& tile);
    void saveTileInfo(const std::string& base_path, const TileInfo& tile);
    cv::Mat upscaleWithKNN(const cv::Mat& input);
    void process_tile_batch(std::vector<TileInfo>& tiles, size_t start, size_t end);
    void drawMethodIndicator(cv::Mat& image, const cv::Rect& position, bool isFSRCNN);

public:
    // Constructor
    AdaptiveUpscaleProcessor(int upscale = 2, int tile = 32, double var_threshold = 100.0, 
                            const std::string& model_path = "fsrcnn_model.pt");

    // Public member function declarations
    std::pair<std::vector<TileInfo>, cv::Size> crop_image(const std::string& image_path);
    cv::Mat stitch_tiles(std::vector<TileInfo>& tiles, const cv::Size& target_size);
    void cleanup();
};

AdaptiveUpscaleProcessor::AdaptiveUpscaleProcessor(int upscale, int tile, double var_threshold, 
                                                  const std::string& model_path) 
    : upscale_factor(upscale),
      tile_size(tile), 
      variance_threshold(var_threshold),
      device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU) {
    
    max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
    
    try {
        fsrcnn_model = torch::jit::load(model_path);
        fsrcnn_model.to(device);
        fsrcnn_model.eval();
    } catch (const c10::Error& e) {
        throw std::runtime_error("Error loading the FSRCNN model: " + std::string(e.what()));
    }

    std::cout << "Initializing Adaptive Processor with " 
              << upscale << "x upscaling, " 
              << tile << "px tiles, "
              << "variance threshold: " << var_threshold << "\n"
              << "Using device: " << (device.is_cuda() ? "CUDA" : "CPU") << "\n"
              << "Using " << max_threads << " threads" << std::endl;
}


/**
 * @brief Converts an OpenCV Mat to a PyTorch tensor
 * 
 * @param image Input image in BGR format
 * @return torch::Tensor Tensor in NCHW format, normalized to [0,1]
 * 
 * Process:
 * 1. Converts image to float32 and normalizes to [0,1]
 * 2. Splits into BGR channels
 * 3. Reverses channel order (BGR->RGB)
 * 4. Creates tensor on appropriate device (CPU/CUDA)
 * 5. Adds batch dimension
 */
torch::Tensor AdaptiveUpscaleProcessor::matToTensor(const cv::Mat& image) {
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F, 1.0 / 255.0);
    
    cv::Mat channels[3];
    cv::split(float_img, channels);
    
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    
    std::vector<torch::Tensor> tensor_channels;
    for (int i = 2; i >= 0; --i) {
        torch::Tensor channel = torch::from_blob(channels[i].data, 
            {1, channels[i].rows, channels[i].cols}, 
            options).clone();
        tensor_channels.push_back(channel);
    }
    
    auto tensor = torch::cat(tensor_channels, 0);
    return tensor.unsqueeze(0);
}

/**
 * @brief Converts a PyTorch tensor back to OpenCV Mat
 * 
 * @param tensor Input tensor in NCHW format with values in [0,1]
 * @return cv::Mat Output image in BGR format with uint8 values
 * 
 * Process:
 * 1. Moves tensor to CPU and removes batch dimension
 * 2. Reverses channel order (RGB->BGR)
 * 3. Scales values to [0,255] and converts to uint8
 */
cv::Mat AdaptiveUpscaleProcessor::tensorToMat(const torch::Tensor& tensor) {
    auto cpu_tensor = tensor.squeeze().cpu();
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
 * @brief Upscales an image tile using the FSRCNN model
 * 
 * @param input Input tile image
 * @return cv::Mat Upscaled image
 * 
 * Process:
 * 1. Converts input to tensor
 * 2. Applies FSRCNN model
 * 3. Converts result back to Mat
 * @throws runtime_error If FSRCNN processing fails
 */
cv::Mat AdaptiveUpscaleProcessor::upscaleWithFSRCNN(const cv::Mat& input) {
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
 * @brief Upscales an image tile using bicubic interpolation
 * 
 * @param input Input tile image
 * @return cv::Mat Upscaled image
 * 
 * Used for low-complexity regions where FSRCNN isn't necessary.
 * Faster than FSRCNN but lower quality for complex details.
 */

cv::Mat AdaptiveUpscaleProcessor::upscaleWithKNN(const cv::Mat& input) {
    cv::Mat output;
    cv::resize(input, output, cv::Size(), upscale_factor, upscale_factor, cv::INTER_CUBIC);
    return output;
}

/**
 * @brief Calculates the variance of a tile to determine its complexity
 * 
 * @param tile Input tile image
 * @return double Variance value
 * 
 * Process:
 * 1. Converts to grayscale
 * 2. Calculates standard deviation
 * 3. Returns variance (stddev²)
 * 
 * Higher variance indicates more complex detail in the tile.
 */
double AdaptiveUpscaleProcessor::calculateVariance(const cv::Mat& tile) {
    cv::Mat gray;
    cv::cvtColor(tile, gray, cv::COLOR_BGR2GRAY);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(gray, mean, stddev);
    
    return stddev[0] * stddev[0];
}

/**
 * @brief Saves debug information about a processed tile
 * 
 * @param base_path Base path for the output file
 * @param tile Tile information structure
 * 
 * Saves:
 * - Processing method used (FSRCNN/KNN)
 * - Variance value
 * - Tile position and size
 * - Threshold used for decision
 */
void AdaptiveUpscaleProcessor::saveTileInfo(const std::string& base_path, const TileInfo& tile) {
    std::string info_path = base_path + ".txt";
    std::ofstream info_file(info_path);
    info_file << "Method: " << (tile.needs_fsrcnn ? "FSRCNN" : "KNN") << "\n"
             << "Variance: " << std::fixed << std::setprecision(2) << tile.variance << "\n"
             << "Position: (" << tile.position.x << "," << tile.position.y << ")\n"
             << "Size: " << tile.position.width << "x" << tile.position.height << "\n"
             << "Threshold: " << variance_threshold;
    info_file.close();
}

/**
 * @brief Adds a visual indicator showing which upscaling method was used
 * 
 * @param image Output image to draw on
 * @param position Position of the tile
 * @param isFSRCNN Whether FSRCNN was used
 * 
 * Draws:
 * - Green dot for FSRCNN-processed tiles
 * - Red dot for KNN-processed tiles
 * Dot size scales with tile size
 */

void AdaptiveUpscaleProcessor::drawMethodIndicator(cv::Mat& image, const cv::Rect& position, bool isFSRCNN) {
    int dot_radius = std::max(3, position.width / 32);
    cv::Point center(
        position.x + dot_radius * 2,
        position.y + position.height - dot_radius * 2
    );
    
    cv::Scalar color = isFSRCNN ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
    cv::circle(image, center, dot_radius, color, -1);
}

/**
 * @brief Splits input image into tiles and analyzes each tile
 * 
 * @param image_path Path to input image
 * @return pair<vector<TileInfo>, Size> Tiles and target size
 * 
 * Process:
 * 1. Loads and validates input image
 * 2. Creates tile grid
 * 3. Analyzes each tile's complexity
 * 4. Saves debug information
 * @throws runtime_error If image loading fails
 */

std::pair<std::vector<AdaptiveUpscaleProcessor::TileInfo>, cv::Size> 
AdaptiveUpscaleProcessor::crop_image(const std::string& image_path) {
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
            tile_info.variance = calculateVariance(tile_info.tile);
            tile_info.needs_fsrcnn = tile_info.variance >= variance_threshold;
            
            std::stringstream ss_debug;
            ss_debug << "debug_crops/original/crop_" << i/tile_size << "_" << j/tile_size 
                    << "_" << (tile_info.needs_fsrcnn ? "FSRCNN" : "KNN") 
                    << "_var" << std::fixed << std::setprecision(2) << tile_info.variance;
            
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
 * @brief Processes a batch of tiles in parallel
 * 
 * @param tiles Vector of all tiles
 * @param start Start index of batch
 * @param end End index of batch
 * 
 * For each tile:
 * 1. Chooses upscaling method based on complexity
 * 2. Applies chosen method
 * 3. Saves debug output
 * 4. Updates tile with result
 */

void AdaptiveUpscaleProcessor::process_tile_batch(std::vector<TileInfo>& tiles, size_t start, size_t end) {
    for (size_t i = start; i < end && i < tiles.size(); ++i) {
        cv::Mat upscaled;
        
        if (tiles[i].needs_fsrcnn) {
            {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "Upscaling detailed tile with FSRCNN (LibTorch)" << std::endl;
            }
            upscaled = upscaleWithFSRCNN(tiles[i].tile);
        } else {
            {
                std::lock_guard<std::mutex> lock(mtx);
                std::cout << "Upscaling low-detail tile with KNN" << std::endl;
            }
            upscaled = upscaleWithKNN(tiles[i].tile);
        }

        std::stringstream ss_upscaled;
        ss_upscaled << "debug_crops/upscaled/crop_" 
                   << tiles[i].position.y/tile_size << "_" 
                   << tiles[i].position.x/tile_size 
                   << "_" << (tiles[i].needs_fsrcnn ? "FSRCNN" : "KNN") 
                   << "_var" << std::fixed << std::setprecision(2) << tiles[i].variance 
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

/**
 * @brief Combines processed tiles into final upscaled image
 * 
 * @param tiles Vector of processed tiles
 * @param target_size Size of output image
 * @return cv::Mat Final upscaled image
 * 
 * Process:
 * 1. Creates empty output image
 * 2. Processes tiles in parallel
 * 3. Copies tiles to correct positions
 * 4. Adds method indicators
 * 5. Generates processing summary
 */
cv::Mat AdaptiveUpscaleProcessor::stitch_tiles(std::vector<AdaptiveUpscaleProcessor::TileInfo>& tiles,
                                             const cv::Size& target_size) {
    cv::Mat stitched_image(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
    int total_tiles = tiles.size();
    int fsrcnn_tiles = 0;
    int knn_tiles = 0;

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
        drawMethodIndicator(stitched_image, upscaled_position, tiles[i].needs_fsrcnn);
        
        if (tiles[i].needs_fsrcnn) fsrcnn_tiles++;
        else knn_tiles++;

        timer->update(i + 1);
        {
            std::lock_guard<std::mutex> lock(mtx);
            std::cout << "\r" << timer->getProgress() << std::flush;
        }
    }
    std::cout << std::endl;  // New line after progress bar

    // Add processing speed to the summary
    auto processing_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - timer->start_time).count();
    double processing_time_sec = processing_time_ms / 1000.0;
    double tiles_per_second = static_cast<double>(total_tiles) / processing_time_sec;

    std::ofstream summary_file("debug_crops/processing_summary.txt");
    summary_file << "Processing Summary\n"
                << "=================\n"
                << "Total tiles: " << total_tiles << "\n"
                << "FSRCNN processed: " << fsrcnn_tiles << " tiles (marked with green dots)\n"
                << "KNN processed: " << knn_tiles << " tiles (marked with red dots)\n"
                << "Processing time: " << processing_time_ms << " ms (" 
                << std::fixed << std::setprecision(3) << processing_time_sec << " seconds)\n"
                << "Processing speed: " << std::fixed << std::setprecision(2) 
                << tiles_per_second << " tiles/second\n"
                << "Variance threshold: " << variance_threshold << "\n"
                << "Tile size: " << tile_size << "x" << tile_size << " pixels\n"
                << "Upscale factor: " << upscale_factor << "x\n"
                << "Original size: " << target_size.width/upscale_factor << "x" << target_size.height/upscale_factor << "\n"
                << "Final size: " << target_size.width << "x" << target_size.height << "\n"
                << "Processing device: " << (device.is_cuda() ? "CUDA" : "CPU");
    summary_file.close();

    std::cout << "\nProcessing summary:\n"
              << "Total tiles: " << total_tiles << "\n"
              << "FSRCNN processed: " << fsrcnn_tiles << " tiles (marked with green dots)\n"
              << "KNN processed: " << knn_tiles << " tiles (marked with red dots)\n"
              << "Total processing time: " << processing_time_ms << " ms (" 
              << std::fixed << std::setprecision(3) << processing_time_sec << " seconds)\n"
              << "Average processing speed: " << std::fixed << std::setprecision(2) 
              << tiles_per_second << " tiles/second" 
              << std::endl;

    return stitched_image;
}

/**
 * @brief Cleans up temporary files and directories
 * 
 * Removes:
 * - Debug crop directory
 * - Original tile images
 * - Upscaled tile images
 * - Debug information files
 */

void AdaptiveUpscaleProcessor::cleanup() {
    if (fs::exists("debug_crops")) {
        fs::remove_all("debug_crops");
    }
}

/**
 * @brief Main program entry point
 * 
 * @param argc Argument count
 * @param argv Argument values
 * @return int 0 on success, 1 on error
 * 
 * Process:
 * 1. Parses command line arguments
 * 2. Initializes processor
 * 3. Processes image
 * 4. Saves result
 * 5. Shows output
 * 6. Cleans up
 */

int main(int argc, char** argv) {
    try {
        std::string image_path = "img.png";
        if (argc > 1) {
            image_path = argv[1];
        }

        std::cout << "Processing image: " << image_path << std::endl;

        // Create processor instance
        AdaptiveUpscaleProcessor processor(2, 28, 1000.0, "fsrcnn_model.pt");

        // Process the image
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