#include "URAM_FIFO.h"
#include "URAM.h"
#include "stdint.h"
#include <stdio.h>

URAM_FIFO::URAM_FIFO(uint32_t num_deep, uint32_t num_wide){


    this->num_deep = num_deep;
    this->num_wide = num_wide;

    if(this->num_deep > MAX_URAMS_DEEP){
        this->num_deep = MAX_URAMS_DEEP;
        printf("WARNING [URAM_FIFO.cpp::URAM_FIFO()] Num deep truncated to %d (was %d)",\
        MAX_URAMS_DEEP, num_deep);
    }

    if(this->num_wide > MAX_URAMS_WIDE){
        this->num_wide = MAX_URAMS_WIDE;
        printf("WARNING [URAM_FIFO.cpp::URAM_FIFO()] Num wide truncated to %d (was %d)",\
        MAX_URAMS_WIDE, num_wide);
    }

    for (int i = 0; i < this->num_deep; i++){
        for(int j = 0; j < this->num_wide; j++){
            mem_arr[i][j] = new URAM();
        }
    }
}

URAM_FIFO::~URAM_FIFO(){

    for (int i = 0; i < this->num_deep; i++){
        for(int j = 0; j < this->num_wide; j++){
            delete mem_arr[i][j];
        }
    }
}