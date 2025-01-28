// processing_timer.hpp (no changes needed, but included for completeness)
#ifndef PROCESSING_TIMER_HPP
#define PROCESSING_TIMER_HPP

#include <chrono>
#include <string>
#include <sstream>
#include <iomanip>

class ProcessingTimer {
public:
    explicit ProcessingTimer(size_t total);
    void update(size_t current_items);
    std::string getProgress() const;

private:
    size_t total_items;
    size_t processed_items;
    std::chrono::steady_clock::time_point start_time;
    
    std::string formatDuration(std::chrono::milliseconds duration) const;
};

#endif // PROCESSING_TIMER_HPP