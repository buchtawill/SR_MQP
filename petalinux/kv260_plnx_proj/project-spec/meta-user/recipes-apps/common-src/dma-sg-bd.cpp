
#include "axi-dma.h"
#include "dma-sg-bd.h"
#include "bits.h"

#ifdef DMA_SG_MODE
uint32_t mm2s_bd_idx_to_addr(uint32_t index){
    return DMA_BDR_MM2S_BASE + (index * sizeof(DMA_SG_BD));
}

uint32_t s2mm_bd_idx_to_addr(uint32_t index){
    return DMA_BDR_S2MM_BASE + (index * sizeof(DMA_SG_BD));
}

void set_buffer_length(volatile DMA_SG_BD *bd, uint32_t len){
    // Set the buffer length in the control register by RMW'ing it

    uint32_t temp = bd->control;
    temp &= ~BD_CONTROL_BUF_LEN_MASK;
    temp |= len & BD_CONTROL_BUF_LEN_MASK;
    bd->control = temp;
}

void set_txsof_bit(volatile DMA_SG_BD *bd, uint32_t val){
    // Bit 27 of control register
    uint32_t temp = bd->control;
    temp &= ~BIT27;
    temp |= val << 27;
    bd->control = temp;
}

void set_txeof_bit(volatile DMA_SG_BD *bd, uint32_t val){
    // Bit 26 of control register
    uint32_t temp = bd->control;
    temp &= ~BIT26;
    temp |= val << 26;
    bd->control = temp;
}

uint32_t get_bd_cmplt_bit(volatile DMA_SG_BD *bd){
    uint32_t temp = bd->status;
    temp &= BIT31;
    if(temp == BIT31) return 1;
    else              return 0;
}

uint32_t get_transferred_bytes(volatile DMA_SG_BD *bd){

    // Lower 26 bits of register
    return bd->status & 0x03FFFFFF;
}

#endif // DMA_SG_MODE