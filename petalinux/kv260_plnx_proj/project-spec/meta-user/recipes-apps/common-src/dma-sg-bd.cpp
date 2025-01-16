
#include "axi-dma.h"
#include "dma-sg-bd.h"
#include "bits.h"

#include <stdio.h>

#ifdef DMA_SG_MODE
uint32_t mm2s_bd_idx_to_addr(uint32_t index){
    return DMA_BDR_MM2S_BASE + (index * sizeof(DMA_SG_BD));
}

uint32_t s2mm_bd_idx_to_addr(uint32_t index){
    return DMA_BDR_S2MM_BASE + (index * sizeof(DMA_SG_BD));
}

void set_buffer_length(BD_PTR bd, uint32_t len){
    // Set the buffer length in the control register by RMW'ing it

    uint32_t temp = bd->control;
    temp &= ~BD_CONTROL_BUF_LEN_MASK;
    temp |= len & BD_CONTROL_BUF_LEN_MASK;
    bd->control = temp;
}

void set_sof_bit(BD_PTR bd, uint32_t val){
    // Bit 27 of control register
    uint32_t temp = bd->control;
    temp &= ~BIT27;
    temp |= val << 27;
    bd->control = temp;
}

void set_eof_bit(BD_PTR bd, uint32_t val){
    // Bit 26 of control register
    uint32_t temp = bd->control;
    temp &= ~BIT26;
    temp |= val << 26;
    bd->control = temp;
}

uint32_t get_bd_cmplt_bit(BD_PTR bd){
    uint32_t temp = bd->status;
    temp &= BIT31;
    if(temp == BIT31) return 1;
    else              return 0;
}

uint32_t get_transferred_bytes(BD_PTR bd){

    // Lower 26 bits of register
    return bd->status & 0x03FFFFFF;
}

void clear_cmplt_bit(BD_PTR bd){
    // Clear the cmplt bit in the status register
    uint32_t temp = bd->status;
    temp &= ~BIT31;
    bd->status = temp;
}

void print_bd_info(BD_PTR bd){
    // printf("INFO [dma-sg-bd] BD @ 0x%08X\n", bd->get_phys_address());
    printf("INFO [dma-sg-bd] Control:        0x%08X\n", bd->control);
    printf("INFO [dma-sg-bd] Status:         0x%08X\n", bd->status);
    printf("INFO [dma-sg-bd] Buffer address: 0x%08X\n", bd->buffer_address);
    printf("INFO [dma-sg-bd] Next desc ptr:  0x%08X\n", bd->next_desc_ptr);
    
}

#endif // DMA_SG_MODE