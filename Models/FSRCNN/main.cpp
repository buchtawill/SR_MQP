#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
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

namespace fs = std::filesystem;

class AdaptiveUpscaleProcessor {
private:
    int upscale_factor;
    int tile_size;
    double variance_threshold;
    const std::string UPSCALED_FILE = "upscaled_image.png";
    std::mutex mtx;
    std::condition_variable cv;
    int max_threads;
    
public:

    /**
     * @brief Constructor initializing the processor with specific parameters
     * @param upscale Upscaling factor (default: 2)
     * @param tile Tile size in pixels (default: 32)
     * @param var_threshold Variance threshold for FSRCNN vs KNN decision (default: 100.0)
     */

    AdaptiveUpscaleProcessor(int upscale = 2, int tile = 32, double var_threshold = 100.0) 
        : upscale_factor(upscale), tile_size(tile), variance_threshold(var_threshold) {
        max_threads = std::max(1, static_cast<int>(std::thread::hardware_concurrency()) - 1);
        std::cout << "Initializing Adaptive Processor with " 
                  << upscale << "x upscaling, " 
                  << tile_size << "px tiles, "
                  << "variance threshold: " << variance_threshold
                  << " using " << max_threads << " threads" << std::endl;
    }


    /**
     * @struct TileInfo
     * @brief Contains all information about a single image tile
     * 
     * Stores both the original and processed data for each tile,
     * including position information and processing decisions
     */

    struct TileInfo {
        cv::Mat tile;
        cv::Rect position;
        std::string filename;
        std::string debug_filename;
        cv::Mat upscaled_result;
        double variance;
        bool needs_fsrcnn;
    };


    /**
     * @brief Calculates the variance of a tile to determine its complexity
     * @param tile Input image tile
     * @return Variance value indicating complexity
     * 
     * Converts to grayscale and calculates statistical variance
     * Higher variance indicates more detail/complexity
     */


    double calculateVariance(const cv::Mat& tile) {
        cv::Mat gray;
        cv::cvtColor(tile, gray, cv::COLOR_BGR2GRAY);
        
        cv::Scalar mean, stddev;
        cv::meanStdDev(gray, mean, stddev);
        
        return stddev[0] * stddev[0];
    }


    /**
     * @brief Saves processing information for a tile to a text file
     * @param base_path Base path for the info file
     * @param tile TileInfo containing the tile's data
     * 
     * Creates a detailed record of processing decisions and parameters
     */

    void saveTileInfo(const std::string& base_path, const TileInfo& tile) {
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
     * @brief Performs nearest-neighbor upscaling
     * @param input Input image
     * @return Upscaled image
     * 
     * Used for low-complexity tiles where FSRCNN would be overkill
     */

    cv::Mat upscaleWithKNN(const cv::Mat& input) {
        cv::Mat output;
        cv::resize(input, output, cv::Size(), upscale_factor, upscale_factor, cv::INTER_NEAREST);
        return output;
    }

    /**
     * @brief Splits an image into tiles and analyzes each one
     * @param image_path Path to input image
     * @return Pair of vector of TileInfo and target size
     * 
     * Creates directory structure for debug output
     * Processes each tile in parallel using OpenMP
     * Saves debug information for each tile
     */
    
    std::pair<std::vector<TileInfo>, cv::Size> crop_image(const std::string& image_path) {
        cv::Mat image = cv::imread(image_path);
        if (image.empty()) {
            throw std::runtime_error("Failed to load image: " + image_path);
        }

        std::cout << "Original image size: " << image.cols << "x" << image.rows << std::endl;

        // Create directories
        fs::create_directories("temp_crops");
        fs::create_directories("debug_crops");
        fs::create_directories("debug_crops/original");
        fs::create_directories("debug_crops/upscaled");

        int width = image.cols;
        int height = image.rows;
        std::vector<TileInfo> tiles;
        tiles.reserve((height / tile_size + 1) * (width / tile_size + 1));

        #pragma omp parallel for collapse(2) schedule(dynamic)
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

                // Generate filenames with method and variance in name
                std::stringstream ss_proc, ss_debug;
                ss_proc << "temp_crops/crop_" << i/tile_size << "_" << j/tile_size << ".png";
                
                ss_debug << "debug_crops/original/crop_" << i/tile_size << "_" << j/tile_size 
                        << "_" << (tile_info.needs_fsrcnn ? "FSRCNN" : "KNN") 
                        << "_var" << std::fixed << std::setprecision(2) << tile_info.variance;
                
                tile_info.filename = ss_proc.str();
                tile_info.debug_filename = ss_debug.str();

                std::vector<int> compression_params;
                compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
                compression_params.push_back(9);

                {
                    std::lock_guard<std::mutex> lock(mtx);
                    // Save the original tile
                    cv::imwrite(tile_info.debug_filename + ".png", tile_info.tile, compression_params);
                    
                    // Save the tile information
                    saveTileInfo(tile_info.debug_filename, tile_info);
                    
                    // Save processing version only if needed for FSRCNN
                    if (tile_info.needs_fsrcnn) {
                        cv::imwrite(tile_info.filename, tile_info.tile, compression_params);
                    }
                    
                    tiles.push_back(tile_info);
                }
            }
        }

        return std::make_pair(tiles, cv::Size(width * upscale_factor, height * upscale_factor));
    }


    /**
     * @brief Processes a batch of tiles using either FSRCNN or KNN
     * @param tiles Vector of tiles to process
     * @param start Starting index
     * @param end Ending index
     * 
     * Handles the actual upscaling of tiles
     * Calls external Python FSRCNN script when needed
     * Saves debug information and upscaled results
     */

    void process_tile_batch(std::vector<TileInfo>& tiles, size_t start, size_t end) {
        for (size_t i = start; i < end && i < tiles.size(); ++i) {
            cv::Mat upscaled;
            
            if (tiles[i].needs_fsrcnn) {
                if (fs::exists(UPSCALED_FILE)) {
                    fs::remove(UPSCALED_FILE);
                }

                std::string command = "python3 FSRCNN.py \"" + tiles[i].filename + "\"";
                
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    std::cout << "Upscaling detailed tile with FSRCNN: " << tiles[i].filename << std::endl;
                }
                
                int status = system(command.c_str());
                if (status != 0) {
                    throw std::runtime_error("Failed to upscale tile: " + tiles[i].filename);
                }

                upscaled = cv::imread(UPSCALED_FILE);
                if (upscaled.empty()) {
                    throw std::runtime_error("Failed to load upscaled tile: " + UPSCALED_FILE);
                }

                fs::remove(tiles[i].filename);
                fs::remove(UPSCALED_FILE);
            } else {
                {
                    std::lock_guard<std::mutex> lock(mtx);
                    std::cout << "Upscaling low-detail tile with KNN" << std::endl;
                }
                upscaled = upscaleWithKNN(tiles[i].tile);
            }

            // Save upscaled version
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
     * @brief Draws an indicator showing which method was used for each tile
     * @param image Output image
     * @param position Tile position
     * @param isFSRCNN Whether FSRCNN was used
     * 
     * Adds visual debugging information to the output
     * Green dot = FSRCNN, Red dot = KNN
     */

    void drawMethodIndicator(cv::Mat& image, const cv::Rect& position, bool isFSRCNN) {
        // Calculate dot position (bottom left of tile)
        int dot_radius = std::max(3, position.width / 32);  // Scale dot size with tile
        cv::Point center(
            position.x + dot_radius * 2,  // Offset from left edge
            position.y + position.height - dot_radius * 2  // Offset from bottom edge
        );
        
        // Draw filled circle with appropriate color
        cv::Scalar color = isFSRCNN ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);  // Green for FSRCNN, Red for KNN
        cv::circle(image, center, dot_radius, color, -1);  // -1 means filled circle
    }



    /**
     * @brief Combines processed tiles into final image
     * @param tiles Vector of processed tiles
     * @param target_size Size of output image
     * @return Final stitched image
     * 
     * Uses multiple threads for processing
     * Adds visual indicators for processing method
     * Creates detailed processing summary
     */

    cv::Mat stitch_tiles(std::vector<TileInfo>& tiles, const cv::Size& target_size) {
        cv::Mat stitched_image(target_size, CV_8UC3, cv::Scalar(0, 0, 0));
        int total_tiles = tiles.size();
        int fsrcnn_tiles = 0;
        int knn_tiles = 0;

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
            
            // Draw the method indicator after copying the tile
            drawMethodIndicator(stitched_image, upscaled_position, tiles[i].needs_fsrcnn);
            
            if (tiles[i].needs_fsrcnn) fsrcnn_tiles++;
            else knn_tiles++;

            std::cout << "Progress: " << (i + 1) << "/" << total_tiles 
                      << " tiles processed (" 
                      << ((i + 1) * 100 / total_tiles) << "%)" << std::endl;
        }

        // Update summary information to include indicator information
        std::ofstream summary_file("debug_crops/processing_summary.txt");
        summary_file << "Processing Summary\n"
                    << "=================\n"
                    << "Total tiles: " << total_tiles << "\n"
                    << "FSRCNN processed: " << fsrcnn_tiles << " tiles (marked with green dots)\n"
                    << "KNN processed: " << knn_tiles << " tiles (marked with red dots)\n"
                    << "Variance threshold: " << variance_threshold << "\n"
                    << "Tile size: " << tile_size << "x" << tile_size << " pixels\n"
                    << "Upscale factor: " << upscale_factor << "x\n"
                    << "Original size: " << target_size.width/upscale_factor << "x" << target_size.height/upscale_factor << "\n"
                    << "Final size: " << target_size.width << "x" << target_size.height;
        summary_file.close();

        std::cout << "Processing summary:\n"
                  << "Total tiles: " << total_tiles << "\n"
                  << "FSRCNN processed: " << fsrcnn_tiles << " tiles (marked with green dots)\n"
                  << "KNN processed: " << knn_tiles << " tiles (marked with red dots)" << std::endl;

        return stitched_image;
    }

    /**
     * @brief Cleans up temporary files and directories
     * 
     * Removes temporary processing files and directories
     * Should be called after processing is complete
     */

    void cleanup() {
        if (fs::exists("temp_crops")) {
            fs::remove_all("temp_crops");
        }
        if (fs::exists(UPSCALED_FILE)) {
            fs::remove(UPSCALED_FILE);
        }
    }
};


/**
 * @brief Main function handling program execution
 * @param argc Argument count
 * @param argv Argument values
 * @return 0 on success, 1 on error
 * 
 * Sets up the processor with specified parameters
 * Processes the input image
 * Displays and saves results
 * Handles error conditions
 */


int main(int argc, char** argv) {
    try {
        std::string image_path = "testImage/img.png";
        if (argc > 1) {
            image_path = argv[1];
        }

        std::cout << "Processing image: " << image_path << std::endl;

        // Change this depending on upscale, tile size, variance
        AdaptiveUpscaleProcessor processor(2, 28, 1000.0);

        auto [tiles, target_size] = processor.crop_image(image_path);
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