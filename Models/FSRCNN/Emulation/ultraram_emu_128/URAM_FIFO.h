// There's probably a better way to do this by abstracing
// FIFO to instantiate either block rams or ultra rams

#ifndef URAM_FIFO_H_
#define URAM_FIFO_H_

#include "URAM.h"
#include "stdint.h"

#define MAX_URAMS_WIDE 32
#define MAX_URAMS_DEEP 32

class URAM_FIFO {

    public:
    URAM_FIFO(uint32_t num_deep, uint32_t num_wide);
    ~URAM_FIFO();


    // TODO: @PLavering - Implement these methods
    void enqueue(uint8_t byte);
    uint8_t dequeue(uint8_t byte);

    private:
    uint32_t num_deep = 1;
    uint32_t num_wide = 1;
    uint32_t total_num_urams = 0;
    URAM *mem_arr[MAX_URAMS_DEEP][MAX_URAMS_WIDE];

    uint32_t write_pointer_byte = 0;
    uint32_t read_pointer_byte = 0;
    uint32_t size_bytes = 0;

};


#endif // URAM_FIFO_H