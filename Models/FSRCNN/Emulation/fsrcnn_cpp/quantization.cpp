#include "quantization.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <iomanip>
#include <torch/nn.h>

namespace fs = std::filesystem;


QuantizationParams QuantizedProcessor::calculate_quantization_params(const torch::Tensor& tensor) {
    auto tensor_no_grad = tensor.detach();
    tensor_no_grad.set_requires_grad(false);
    
    float min_val = tensor_no_grad.min().item<float>();
    float max_val = tensor_no_grad.max().item<float>();
    
    float abs_max = std::max(std::abs(min_val), std::abs(max_val));
    float scale = abs_max / 127.0f;
    int64_t zero_point = 0;

    return {scale, zero_point};
}

torch::Tensor QuantizedProcessor::quantize_tensor(const torch::Tensor& tensor, const QuantizationParams& params) {
    auto tensor_no_grad = tensor.detach();
    tensor_no_grad.set_requires_grad(false);
    
    auto scaled = tensor_no_grad / params.scale;
    auto clamped = scaled.clamp(-127, 127);
    return clamped.round().to(torch::kInt8);
}

torch::jit::Module QuantizedProcessor::quantize_model(torch::jit::Module& model) {
    model.eval();

    // Quantize weights
    const auto named_parameters = model.named_parameters();
    for (const auto& param : named_parameters) {
        auto tensor = param.value.detach();
        tensor.set_requires_grad(false);

        auto qparams = calculate_quantization_params(tensor);
        auto quantized = quantize_tensor(tensor, qparams);

        // Update the parameter with quantized tensor
        param.value.set_data(quantized.to(torch::kFloat32));
    }

    // Note: Directly accessing children modules
    for (const auto& child : model.named_children()) {
        auto child_type = child.value.type();
        if (child_type->findMethod("forward")) {
        // If "forward" exists, process this child module
        }
    }


    return model;
}

void QuantizedProcessor::save_model_to_pth(const torch::jit::Module& model, const std::string& output_path) {
    try {
        // Save the model
        model.save(output_path);
        
        // Analyze model parameters
        size_t total_fp32_bytes = 0;
        size_t total_int8_bytes = 0;
        size_t num_parameters = 0;
        
        std::cout << "\nParameter Size Analysis:" << std::endl;
        std::cout << std::setw(50) << std::left << "Layer Name" 
                 << std::setw(15) << "Parameters"
                 << std::setw(15) << "FP32 Size"
                 << std::setw(15) << "INT8 Size" << std::endl;
        std::cout << std::string(95, '-') << std::endl;
        
        for (const auto& param : model.named_parameters()) {
            if (!param.value.defined()) continue;
            
            const auto& tensor = param.value;
            size_t params = tensor.numel();
            size_t fp32_bytes = params * 4;  // 4 bytes per FP32
            size_t int8_bytes = params * 1;  // 1 byte per INT8
            
            num_parameters += params;
            total_fp32_bytes += fp32_bytes;
            total_int8_bytes += int8_bytes;
            
            std::cout << std::setw(50) << std::left << param.name
                     << std::setw(15) << params
                     << std::setw(15) << (fp32_bytes / 1024.0) << "KB"
                     << std::setw(15) << (int8_bytes / 1024.0) << "KB" << std::endl;
        }
        
        std::cout << std::string(95, '-') << std::endl;
        std::cout << "Total parameters: " << num_parameters << std::endl;
        std::cout << "Theoretical FP32 size: " << (total_fp32_bytes / 1024.0) << " KB" << std::endl;
        std::cout << "Theoretical INT8 size: " << (total_int8_bytes / 1024.0) << " KB" << std::endl;
        
        // Get actual file size
        std::ifstream file(output_path, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            auto file_size = static_cast<size_t>(file.tellg());  // Convert to size_t
            std::cout << "Actual file size: " << (file_size / 1024.0) << " KB" << std::endl;
            std::cout << "Storage overhead: " << ((file_size - total_int8_bytes) / 1024.0) << " KB" << std::endl;
        }
        
    } catch (const c10::Error& e) {
        throw std::runtime_error("PyTorch error while saving model: " + std::string(e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error("Error saving model: " + std::string(e.what()));
    }
}

void QuantizedProcessor::save_quantized_weights(const torch::jit::Module& model, const std::string& output_dir) {
    fs::create_directories(output_dir);
    
    const auto named_parameters = model.named_parameters();
    for (const auto& param : named_parameters) {
        auto tensor = param.value.detach();
        tensor.set_requires_grad(false);
        
        auto qparams = calculate_quantization_params(tensor);
        auto quantized = quantize_tensor(tensor, qparams);
        
        std::string safe_name = get_safe_filename(param.name);
        std::string full_path = (fs::path(output_dir) / safe_name).string();
        
        std::cout << "Saving weight: " << param.name << " to " << full_path << std::endl;
        save_tensor_to_file(full_path, quantized, qparams);
    }
}

void QuantizedProcessor::save_tensor_to_file(const std::string& filename,
                                             const torch::Tensor& tensor,
                                             const QuantizationParams& params) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    
    file.write(reinterpret_cast<const char*>(&params.scale), sizeof(float));
    file.write(reinterpret_cast<const char*>(&params.zero_point), sizeof(int64_t));
    
    auto sizes = tensor.sizes();
    int64_t num_dims = sizes.size();
    file.write(reinterpret_cast<const char*>(&num_dims), sizeof(int64_t));
    for (int64_t i = 0; i < num_dims; i++) {
        int64_t dim_size = sizes[i];
        file.write(reinterpret_cast<const char*>(&dim_size), sizeof(int64_t));
    }
    
    auto data_ptr = tensor.data_ptr<int8_t>();
    size_t num_elements = tensor.numel();
    file.write(reinterpret_cast<const char*>(data_ptr), num_elements * sizeof(int8_t));
}

std::string QuantizedProcessor::get_safe_filename(const std::string& parameter_name) {
    std::string filename = parameter_name;
    std::replace_if(filename.begin(), filename.end(),
                   [](char c) { return !std::isalnum(c) && c != '_' && c != '-'; }, '_');
    return filename + ".bin";
}