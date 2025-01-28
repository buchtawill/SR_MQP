#include "URAM.h"
#include <stdint.h>

// Constructor
URAM::URAM(){}

// Destructor
URAM::~URAM(){}

void URAM::read_uram(uint32_t row, uint8_t *buf){
    if(row >= URAM_DEPTH) return;

    for(int i = 0; i < URAM_WIDTH_BYTES; i++){
        buf[i] = this->mem[row][i];
    }
}

void URAM::write_uram(uint32_t row, uint8_t *buf){
    if(row >= URAM_DEPTH) return;

    else { 
        for(int i = 0; i < URAM_WIDTH_BYTES; i++){
            this->mem[row][i] = buf[i];
        }
    }
}

uint8_t URAM::read_uram(uint32_t row, uint8_t col){
    if((row >= URAM_DEPTH) || (col > URAM_WIDTH_BYTES)) return 0;

    else return this->mem[row][col];
}

void URAM::write_uram(uint32_t row, uint8_t col, uint8_t val){
    if((row >= URAM_DEPTH) || (col > URAM_WIDTH_BYTES)) return;

    else this->mem[row][col] = val;
}

