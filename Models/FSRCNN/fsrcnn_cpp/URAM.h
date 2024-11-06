// Include Guards
#ifndef URAM_H_
#define URAM_H_

#include <stdint.h>

#define URAM_DEPTH          4096    // 4096 addresses
#define URAM_WIDTH_BITS     72      // Each address is 72 bits wide
#define URAM_WIDTH_BYTES    9       // Each address has 9 bytes
#define NUM_BYTES_IN_URAM   (URAM_DEPTH * URAM_WIDTH_BYTES)

// Start by assuming only 1 URAM block
class URAM {

    public:

    URAM();
    ~URAM();

    // Read the address of uram into buf (buf must be allocated to URAM_WIDTH_BYTES)
    // 0 <= row < URAM_DEPTH
    void read_uram(uint32_t row, uint8_t* buf);

    // Return the byte at the row and column of URAM
    // 0 <= row < URAM_DEPTH
    // 0 <= col < URAM_WIDTH_BYTES
    uint8_t read_uram(uint32_t row, uint8_t col);

    // Put val into the uram at col index of row
    // 0 <= row < URAM_DEPTH
    // 0 <= col < URAM_WIDTH_BYTES
    void write_uram(uint32_t row, uint8_t col, uint8_t val);

    // Copy the contents of buf into the specified row of URAM
    // buf MUST be URAM_WIDTH_BYTES
    // 0 <= row < URAM_DEPTH
    void write_uram(uint32_t row, uint8_t *buf);

    private:
        uint8_t mem[URAM_DEPTH][URAM_WIDTH_BYTES];

        
};

#endif // URAM_H_