#include "processing_timer.hpp"

ProcessingTimer::ProcessingTimer(size_t total)
    : total_items(total),
      processed_items(0),
      start_time(std::chrono::steady_clock::now()) {
}

void ProcessingTimer::update(size_t current_items) {
    processed_items = current_items;
}

std::string ProcessingTimer::formatDuration(std::chrono::milliseconds duration) const {
    auto total_seconds = duration.count() / 1000;
    auto hours = total_seconds / 3600;
    auto minutes = (total_seconds % 3600) / 60;
    auto seconds = total_seconds % 60;
    
    std::stringstream ss;
    if (hours > 0) {
        ss << hours << "h ";
    }
    if (minutes > 0 || hours > 0) {
        ss << minutes << "m ";
    }
    ss << seconds << "s";
    return ss.str();
}

std::string ProcessingTimer::getProgress() const {
    auto current_time = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(current_time - start_time);

    double progress = static_cast<double>(processed_items) / total_items;
    double items_per_second = static_cast<double>(processed_items * 1000.0) / elapsed.count();
    
    double remaining_items = total_items - processed_items;
    auto estimated_remaining_ms = static_cast<long long>(remaining_items * elapsed.count() / processed_items);
    auto estimated_remaining = std::chrono::milliseconds(estimated_remaining_ms);
    
    std::stringstream ss;
    ss << std::fixed << std::setprecision(1)
       << "[" << std::string(static_cast<size_t>(progress * 40), '=') << std::string(40 - static_cast<size_t>(progress * 40), ' ') << "] "
       << "(" << processed_items << "/" << total_items << " tiles) | "
       << items_per_second << " tiles/s | "
       << "ETA: " << formatDuration(estimated_remaining);
    
    return ss.str();
}