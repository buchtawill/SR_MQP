#ifndef QUANTIZATION_HPP
#define QUANTIZATION_HPP

#include <torch/script.h>
#include <string>

struct QuantizationParams {
    float scale;
    int64_t zero_point;
};

enum class QuantizationType {
    NONE,
    INT8
};

class QuantizedProcessor {
public:
    // Core quantization methods
    static QuantizationParams calculate_quantization_params(const torch::Tensor& tensor);
    static torch::Tensor quantize_tensor(const torch::Tensor& tensor, const QuantizationParams& params);
    static torch::jit::Module quantize_model(torch::jit::Module& model);
    
    // Weight saving methods
    static void save_quantized_weights(const torch::jit::Module& model, const std::string& output_dir);
    
    // New method for saving complete model
    static void save_model_to_pth(const torch::jit::Module& model, const std::string& output_path);

private:
    static void save_tensor_to_file(const std::string& filename, 
                                  const torch::Tensor& tensor, 
                                  const QuantizationParams& params);
    static std::string get_safe_filename(const std::string& parameter_name);
};

#endif // QUANTIZATION_HPP