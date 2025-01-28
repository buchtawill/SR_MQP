#ifndef ADAPTIVE_UPSCALER_HPP
#define ADAPTIVE_UPSCALER_HPP

#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <memory>
#include "quantization.hpp"
#include "processing_timer.hpp"

enum class UpscaleMethod {
    FSRCNN,
    BILINEAR
};

class AdaptiveUpscaleProcessor {
public:
    struct TileInfo {
        cv::Mat tile;
        cv::Rect position;
        std::string debug_filename;
        cv::Mat upscaled_result;
        bool processed = false;  // Flag to track if tile has been processed
    };

    AdaptiveUpscaleProcessor(int upscale = 2, int tile = 28,
                            const std::string& model_path = "fsrcnn_model.pt",
                            QuantizationType quantization = QuantizationType::NONE,
                            UpscaleMethod method = UpscaleMethod::FSRCNN);

    // Core processing methods
    std::pair<std::vector<TileInfo>, cv::Size> crop_image(const std::string& image_path);
    cv::Mat stitch_tiles(std::vector<TileInfo>& tiles, const cv::Size& target_size);
    
    // Cleanup and model saving methods
    void cleanup();
    void save_quantized_weights(const std::string& output_path);
    void save_model(const std::string& output_path);

private:
    torch::Device device;
    int upscale_factor;
    int tile_size;
    int output_tile_size;  // New member to store output tile size
    QuantizationType quant_type;
    UpscaleMethod upscale_method;
    std::mutex mtx;
    std::condition_variable cv;
    int max_threads;
    torch::jit::script::Module fsrcnn_model;
    std::unique_ptr<ProcessingTimer> timer;
    std::vector<QuantizationParams> activation_qparams;

    // Helper methods
    void quantize_activations(torch::Tensor& tensor);
    torch::Tensor matToTensor(const cv::Mat& image);
    cv::Mat tensorToMat(const torch::Tensor& tensor);
    cv::Mat upscaleWithFSRCNN(const cv::Mat& input);
    cv::Mat upscaleWithBilinear(const cv::Mat& input);
    void saveTileInfo(const std::string& base_path, const TileInfo& tile);
    void process_tile_batch(std::vector<TileInfo>& tiles, size_t start, size_t end);
    
    // New helper methods
    bool verify_tile_size(const cv::Mat& tile) const;
    void pad_tile_if_needed(cv::Mat& tile) const;
    void prepare_tile_for_model(TileInfo& tile);
    cv::Mat remove_padding(const cv::Mat& tile, const cv::Size& original_size) const;
};

#endif // ADAPTIVE_UPSCALER_HPP