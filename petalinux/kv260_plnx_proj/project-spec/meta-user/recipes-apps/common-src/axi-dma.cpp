#include "axi-dma.h"
#include "dma-sg-bd.h"
#include <stdint.h>
#include <stdio.h>
#include <sys/mman.h> //for mmap
#include <errno.h>
#include <string.h> //for strerror
#include <unistd.h> // for usleep
#include <stdlib.h> // for rand

#include "PhysMem.h"
#include "phys-mman.h"
#include <time.h> // for srand

AXIDMA::AXIDMA(uint32_t base_addr, int dev_mem_fd){
    this->base_address = base_addr;
    this->mem_fd = dev_mem_fd;
}

AXIDMA::~AXIDMA(){
    // printf("INFO [AXIDMA::~AXIDMA] Deleting instance with base address 0x%08X\n", this->base_address);
    munmap((void*)dma_phys_addr, DMA_ADDRESS_SPACE_SIZE);
}

int AXIDMA::initialize(){
    this->dma_phys_addr = (volatile uint32_t *)mmap(
        NULL,                   // Let the kernel decide the virtual address
        DMA_ADDRESS_SPACE_SIZE, // Size of memory to map 
        PROT_READ | PROT_WRITE, // Permissions: read and write
        MAP_SHARED,             // Changes are shared with other mappings
        this->mem_fd,           // File descriptor for /dev/mem
        this->base_address      // Physical address of the reserved memory
    );
	if(this->dma_phys_addr == MAP_FAILED){
		printf("ERROR [AXIDMA::initialize()] Failed to map DMA AXI Lite register block: %s\n", strerror(errno));
		return -1;
	}

    this->reset_dma();

    return 0;
}

void AXIDMA::write_dma(uint32_t reg, uint32_t val){
    this->dma_phys_addr[reg >> 2] = val;
}

void AXIDMA::rmw_dma(uint32_t reg, uint32_t mask, uint32_t val){
    uint32_t temp = read_dma(reg);
    temp &= ~mask;
    temp |= val;
    write_dma(reg, temp);
}

uint32_t AXIDMA::read_dma(uint32_t reg){
    return this->dma_phys_addr[reg >> 2];
}

#ifndef DMA_SG_MODE
void AXIDMA::set_mm2s_len(uint16_t transfer_length){
    write_dma(MM2S_TRANSFER_LENGTH, transfer_length);
}

void AXIDMA::set_s2mm_len(uint16_t transfer_length){
    write_dma(S2MM_TRANSFER_LENGTH, transfer_length);
}

void AXIDMA::set_mm2s_src(uint32_t source){
    write_dma(MM2S_SA_LSB32, source);
}

void AXIDMA::set_s2mm_dest(uint32_t destination){
    write_dma(S2MM_DA_LSB32, destination);
}
#endif // DMA_SG_MODE

void AXIDMA::start_mm2s(){
    write_dma(MM2S_DMACR, RUN_DMA);
}

void AXIDMA::start_s2mm(){
    write_dma(S2MM_DMACR, RUN_DMA);
}

void AXIDMA::reset_mm2s(){
    write_dma(MM2S_DMACR, RESET_DMA);
}

void AXIDMA::reset_s2mm(){
    write_dma(S2MM_DMACR, RESET_DMA);
}

void AXIDMA::halt_mm2s(){
    write_dma(MM2S_DMACR, HALT_DMA);
}

void AXIDMA::halt_s2mm(){
    write_dma(S2MM_DMACR, HALT_DMA);
}

void AXIDMA::print_status(uint32_t reg){
    unsigned int status = read_dma(reg);

    if(reg == MM2S_DMASR){
        printf("INFO [AXIDMA::print_status()] MM2S status (0x%04x):", status);
    }
    else if(reg == S2MM_DMASR){
        printf("INFO [AXIDMA::print_status()] S2MM status (0x%04x):", status);
    }
    else{
        printf("ERROR [AXIDMA::print_status()] Invalid register address 0x%02x\n", reg);
        return;
    }

    if (status & STATUS_HALTED)           printf(" Halted.    ");
    else                                  printf(" Running.   ");
    if (status & STATUS_IDLE)             printf(" | Idle. "); 
    if (status & STATUS_SG_INCLDED)       printf(" | SG is included.");
    if (status & STATUS_DMA_INTERNAL_ERR) printf(" | DMA internal error.");
    if (status & STATUS_DMA_SLAVE_ERR)    printf(" | DMA slave error.");
    if (status & STATUS_DMA_DECODE_ERR)   printf(" | DMA decode error.");
    if (status & STATUS_SG_INTERNAL_ERR)  printf(" | SG internal error.");
    if (status & STATUS_SG_SLAVE_ERR)     printf(" | SG slave error.");
    if (status & STATUS_SG_DECODE_ERR)    printf(" | SG decode error.");
    if (status & STATUS_IOC_IRQ)          printf(" | IOC interrupt occurred.");
    if (status & STATUS_DELAY_IRQ)        printf(" | Interrupt on delay occurred.");
    if (status & STATUS_ERR_IRQ)          printf(" | Error interrupt occurred.");
    printf("\n");
}


#ifndef DMA_SG_MODE
int AXIDMA::sync_channel(uint32_t channel_status_reg, uint32_t max_tries){
    uint32_t status = read_dma(channel_status_reg);

    // printf("INFO [AXIDMA::sync_channel()] Waiting for channel to complete. Initial status: \n");
    // print_status(channel_status_reg);

	uint32_t count = 0;
	while(!(status & IOC_IRQ_FLAG)){

		status = read_dma(channel_status_reg);
		count++;
		if(count == max_tries){
            if(channel_status_reg == MM2S_DMASR){
                printf("ERROR [AXIDMA::sync_channel()] MM2S Timeout occurred. Tried %d times\n", max_tries);
            }
            else if(channel_status_reg == S2MM_DMASR){
                printf("ERROR [AXIDMA::sync_channel()] S2MM Timeout occurred. Tried %d times\n", max_tries);
            }
			return -1;
		}
	}

    this->clear_irq_bit(channel_status_reg);

	return count;
}
#endif // DMA_SG_MODE

void AXIDMA::enable_mm2s_intr(){
    write_dma(MM2S_DMACR, ENABLE_ALL_IRQ);
}

void AXIDMA::enable_s2mm_intr(){
    write_dma(S2MM_DMACR, ENABLE_ALL_IRQ);
}

#ifndef DMA_SG_MODE

void AXIDMA::print_debug_info(){
    printf("DEBUG [AXIDMA::print_debug_info()] Base address: 0x%08X\n", this->base_address);
    printf("                                   Total good bytes MM2S: %d\n", this->total_bytes_mm2s);
    printf("                                   Total good bytes S2MM: %d\n", this->total_bytes_s2mm);
    printf("                                   Number of MM2S calls:  %d\n", this->n_mm2s_calls);
    printf("                                   Number of S2MM calls:  %d\n", this->n_s2mm_calls);
}

int AXIDMA::transfer_mm2s(uint32_t src_addr, uint32_t len, bool block){

    n_mm2s_calls++;

    this->clear_irq_bit(MM2S_DMASR);

    // this->reset_mm2s();
    // this->halt_mm2s();
    this->enable_mm2s_intr();
    this->set_mm2s_src(src_addr);

    this->start_mm2s();

    // Setting the length register must be the last step
    this->set_mm2s_len(len);

    if(block){
        int result = this->sync_channel(MM2S_DMASR);
        if(result >= 0) total_bytes_mm2s += len;
        return result;
    }

    return 0;
}

int AXIDMA::transfer_s2mm(uint32_t dst_addr, uint32_t len, bool block){

    n_s2mm_calls++;

    this->clear_irq_bit(S2MM_DMASR); //

    // this->reset_s2mm();//
    // this->halt_s2mm();//
    this->enable_s2mm_intr();//
    this->set_s2mm_dest(dst_addr);//

    this->start_s2mm();//

    // Setting the length register must be the last step
    this->set_s2mm_len(len);

    if(block) {
        int result = this->sync_channel(S2MM_DMASR);
        if(result >= 0) total_bytes_s2mm += len;
        return result;
    }

    return 0;
}

int AXIDMA::transfer(uint32_t src_addr, uint32_t dst_addr, uint32_t len, bool block){

    this->clear_irq_bits();

    this->reset_mm2s();
    this->reset_s2mm();
    this->halt_mm2s();
    this->halt_s2mm();
    this->enable_mm2s_intr();
    this->enable_s2mm_intr();
    this->set_mm2s_src(src_addr);
    this->set_s2mm_dest(dst_addr);

    this->start_mm2s();
    this->start_s2mm();

    // Setting the length register must be the last step
    this->set_mm2s_len(len);
    this->set_s2mm_len(len);

    int num_tries1 = 0, num_tries2 = 0;

    if(block){
        num_tries1 = this->sync_channel(MM2S_DMASR);
        if(num_tries1 < 0){
            return -1;
        }

        num_tries2 = this->sync_channel(S2MM_DMASR);
        if(num_tries2 < 0){
            return -1;
        }
    }
    
    return num_tries1 + num_tries2;
}
#endif // DMA_SG_MODE

#ifndef DMA_SG_MODE
int AXIDMA::self_test_dr(){
    // Get buffers from PhysMman
    // Write some random data to the source address, copy it to the dst address, and check if it's the same
    PhysMem* src_block = PMM.alloc(DMA_SELF_TEST_LEN);
    PhysMem* dst_block = PMM.alloc(DMA_SELF_TEST_LEN);
    PhysMem* src_block2 = PMM.alloc(DMA_SELF_TEST_LEN);
    PhysMem* dst_block2 = PMM.alloc(DMA_SELF_TEST_LEN);

    if(src_block == nullptr || dst_block == nullptr){
        printf("ERROR [AXIDMA::self_test()] PMM failed to allocate memory blocks\n");
        return -1;
    }

    // If size is 0x1000, that is 4096 bytes, so 1024 uint32_t
    for(uint32_t i = 0; i < DMA_SELF_TEST_LEN / 4; i++){
        src_block->write_word(i*4, (uint32_t)rand());
    }

/*
    // printf("INFO [AXIDMA::self_test()] Status registers before transfer: \n");
    // this->print_status(MM2S_DMASR);
    // this->print_status(S2MM_DMASR);
    // printf("INFO [AXIDMA::self_test()] Starting transfer\n");
    int result = this->transfer(src_block->get_phys_address(), dst_block->get_phys_address(), DMA_SELF_TEST_LEN, true);
    if(result < 0){
        printf("ERROR [AXIDMA::self_test()] Transfer failed\n");
        return -1;
    }

    // Check if the data is the same
    // printf("INFO [AXIDMA::self_test()] Checking that data matches\n");
    for(uint32_t i = 0; i < DMA_SELF_TEST_LEN / 4; i++){
        // if(src_addr[i] != dst_addr[i]){
        uint32_t src_word = 0xbeefbeef, dst_word = 0xbeefbeef;
        src_block->read_word(i*4, &src_word);
        dst_block->read_word(i*4, &dst_word);
        if(src_word != dst_word){
            printf("ERROR [AXIDMA::self_test()] Self test failed. Data mismatch at index %d: Expected 0x%08X, got 0x%08X\n", 
                i,
                src_word, 
                dst_word);
            // printf("Dumping memory contents to error_log.txt\n");
            // FILE *fp = fopen("error_log.txt", "w");
            // if(fp == NULL){
            //     printf("ERROR [AXIDMA::self_test()] Failed to open error_log.txt\n");
            //     return -1;
            // }
            // for(int j = 0; j < DMA_SELF_TEST_LEN / 4; j++){
            //     fprintf(fp, "%08d: SRC: 0x%08X    DST: 0x%08X\n", j, src_addr[j], dst_addr[j]);
            // }
            // fclose(fp);
            return -1;
        }
    }

    printf("INFO [AXIDMA::self_test()] transfer() self test passed!\n");
*/

    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    // Test with transfer_mm2s and transfer_s2mm /////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////////////////////////////////////////////////////
    
    // The FIFO is 1024 x 32 bits
    // Test on 64 entries
    // Refill the source block with new random data
    uint32_t individual_len = 2352; // = 1 tile of bytes
    uint32_t bytes_per_row = 28*3;
    for(uint32_t i = 0; i < individual_len / 4; i++){
        // src_block->write_word(i*4, (uint32_t)rand());
        src_block->write_word(i*4, i);
        src_block2->write_word(i*4, (uint32_t)rand());
    }

    // Clear the dest block
    for(uint32_t i = 0; i < individual_len / 4; i++){
        dst_block->write_word(i*4, 0);
        dst_block2->write_word(i*4, 0);
    }

    // int result = this->transfer_mm2s(src_block->get_phys_address(), individual_len, true);
    // if(result < 0){
    //     printf("ERROR [AXIDMA::self_test()] MM2S transfer failed\n");
    //     return -1;
    // }

    // result = this->transfer_s2mm(dst_block->get_phys_address(), individual_len, true);
    // if(result < 0){
    //     printf("ERROR [AXIDMA::self_test()] S2MM transfer failed\n");
    //     return -1;
    // }


    // // do it again
    // result = this->transfer_mm2s(src_block2->get_phys_address(), individual_len, true);
    // if(result < 0){
    //     printf("ERROR [AXIDMA::self_test()] MM2S transfer failed\n");
    //     return -1;
    // }

    // result = this->transfer_s2mm(dst_block2->get_phys_address(), individual_len, true);
    // if(result < 0){
    //     printf("ERROR [AXIDMA::self_test()] S2MM transfer failed\n");
    //     return -1;
    // }

    // Transfer 1 row at a time
    for(int i = 0; i < 28; i++){
        int result = this->transfer_mm2s(src_block->get_phys_address() + (bytes_per_row * i), bytes_per_row, true);
        if(result < 0){
            printf("ERROR [AXIDMA::self_test_dr()] MM2S transfer failed\n");
            return -1;
        }
    }

    for(int i = 0; i < 28; i++){
        int result = this->transfer_s2mm(dst_block->get_phys_address()+(bytes_per_row * i), bytes_per_row, true);
        if(result < 0){
            printf("ERROR [AXIDMA::self_test_dr()] S2MM transfer failed\n");
            return -1;
        }
    }

    // Check if the data is the same for src_block and dst_block
    for(uint32_t i = 0; i < (individual_len) / 4; i++){
        // if(src_addr[i] != dst_addr[i]){
        uint32_t src_word = 0xbeefbeef, dst_word = 0xbeefbeef;
        src_block->read_word(i*4, &src_word);
        dst_block->read_word(i*4, &dst_word);
        if(src_word != dst_word){
            printf("ERROR [AXIDMA::self_test()] Self test failed. Data mismatch at index %d: Expected 0x%08X, got 0x%08X\n", 
                i,
                src_word, 
                dst_word);
            PMM.free(src_block);
            PMM.free(dst_block);
            PMM.free(src_block2);
            PMM.free(dst_block2);
            return -1;
        }
    }

    // Do the same thing for src_block2 and dst_block2
    // for(uint32_t i = 0; i < (individual_len) / 4; i++){
    //     // if(src_addr[i] != dst_addr[i]){
    //     uint32_t src_word = 0xbeefbeef, dst_word = 0xbeefbeef;
    //     src_block2->read_word(i*4, &src_word);
    //     dst_block2->read_word(i*4, &dst_word);
    //     if(src_word != dst_word){
    //         printf("ERROR [AXIDMA::self_test()] Self test failed on 2nd attempt. Data mismatch at index %d: Expected 0x%08X, got 0x%08X\n", 
    //             i,
    //             src_word, 
    //             dst_word);
    //         PMM.free(src_block);
    //         PMM.free(dst_block);
    //         PMM.free(src_block2);
    //         PMM.free(dst_block2);
    //         return -1;
    //     }
    // }

    printf("INFO [AXIDMA::self_test()] transfer_mm2s and transfer_s2mm self test passed!\n");

    PMM.free(src_block);
    PMM.free(dst_block);
    PMM.free(src_block2);
    PMM.free(dst_block2);
    return 0;
}
#else // We're in scatter-gather mode

// TODO: Implement self test for scatter gather mode
int AXIDMA::self_test_sg(){

    this->reset_dma();
    uint32_t mm2s_status_reg = this->read_dma(MM2S_DMASR);
    uint32_t s2mm_status_reg = this->read_dma(S2MM_DMASR);
    if(!(mm2s_status_reg & STATUS_SG_INCLDED)){
        printf("ERROR [AXIDMA::self_test_sg()] MM2S SG not included\n");
        return -1;
    }
    if(!(s2mm_status_reg & STATUS_SG_INCLDED)){
        printf("ERROR [AXIDMA::self_test_sg()] S2MM SG not included\n");
        return -1;
    }
    // printf("INFO [AXIDMA::self_test_sg()] Confirmed that scatter gather mode is enabled\n");

    // Create source and destination blocks to test
    PhysMem* src_block = PMM.alloc(DMA_SELF_TEST_SG_LEN);
    PhysMem* dst_block = PMM.alloc(DMA_SELF_TEST_SG_LEN);

    if(src_block == nullptr || dst_block == nullptr){
        printf("ERROR [AXIDMA::self_test_sg()] PMM failed to allocate memory blocks\n");
        return -1;
    }

    // Fill the source block with random data
    // Clear the dst block
    for(uint32_t i = 0; i < DMA_SELF_TEST_SG_LEN / 4; i++){
        src_block->write_word(i*4, (uint32_t)rand());
        dst_block->write_word(i*4, 0);
    }

    // Create a BD ring for MM2S and S2MM
    PhysMem* mm2s_bds[DMA_SELF_TEST_SG_NUM_BDS];
    PhysMem* s2mm_bds[DMA_SELF_TEST_SG_NUM_BDS];
    for(uint32_t i = 0; i < DMA_SELF_TEST_SG_NUM_BDS; i++){
        mm2s_bds[i] = PMM.alloc(64); // size of a buffer descriptor
        s2mm_bds[i] = PMM.alloc(64);

        if(mm2s_bds[i] == nullptr || s2mm_bds[i] == nullptr){
            printf("ERROR [AXIDMA::self_test_sg()] PMM failed to allocate memory for buffer descriptors\n");
            return -1;
        }

        memset((void*)mm2s_bds[i]->get_mem_ptr(), 0, 64);
        memset((void*)s2mm_bds[i]->get_mem_ptr(), 0, 64);
    }

    // Each bd in mm2s should have DMA_SELF_TEST_BYTES_PER_BD bytes
    // First BD needs txsof
    // Last BD needs txeof AND to point to first BD
    uint32_t end_idx = DMA_SELF_TEST_SG_NUM_BDS - 1;
    set_sof_bit(((BD_PTR)mm2s_bds[0]->get_mem_ptr()), 1);
    for(uint32_t i = 0; i < end_idx; i++){
        BD_PTR current_bd = (BD_PTR)(mm2s_bds[i]->get_mem_ptr());

        set_buffer_length(current_bd, DMA_SELF_TEST_BYTES_PER_BD);
        current_bd->next_desc_index = i + 1;
        current_bd->next_desc_ptr = mm2s_bds[i+1]->get_phys_address();
        current_bd->buffer_address = src_block->get_phys_address() + (i * DMA_SELF_TEST_BYTES_PER_BD);
    }

    BD_PTR last_bd_mm2s = (BD_PTR)(mm2s_bds[end_idx]->get_mem_ptr());
    set_eof_bit(last_bd_mm2s, 1);
    set_buffer_length(last_bd_mm2s, DMA_SELF_TEST_BYTES_PER_BD);
    last_bd_mm2s->next_desc_ptr = mm2s_bds[0]->get_phys_address();
    last_bd_mm2s->next_desc_index = 0;
    last_bd_mm2s->buffer_address = src_block->get_phys_address() + (end_idx * DMA_SELF_TEST_BYTES_PER_BD);

    // Do the same for S2MM
    // set_sof_bit(((BD_PTR)s2mm_bds[0]->get_mem_ptr()), 1);
    for(uint32_t i = 0; i < end_idx; i++){
        BD_PTR current_bd = (BD_PTR)(s2mm_bds[i]->get_mem_ptr());

        set_buffer_length(current_bd, DMA_SELF_TEST_BYTES_PER_BD);
        current_bd->next_desc_index = i + 1;
        current_bd->next_desc_ptr = s2mm_bds[i+1]->get_phys_address();
        current_bd->buffer_address = dst_block->get_phys_address() + (i * DMA_SELF_TEST_BYTES_PER_BD);
    }
    BD_PTR last_bd_s2mm = (BD_PTR)(s2mm_bds[end_idx]->get_mem_ptr());
    set_eof_bit(last_bd_s2mm, 1);
    set_buffer_length(last_bd_s2mm, DMA_SELF_TEST_BYTES_PER_BD);
    last_bd_s2mm->next_desc_ptr = s2mm_bds[0]->get_phys_address();
    last_bd_s2mm->next_desc_index = 0;
    last_bd_s2mm->buffer_address = dst_block->get_phys_address() + (end_idx * DMA_SELF_TEST_BYTES_PER_BD);

    // Start the MM2S transfer
    printf("INFO [AXIDMA::self_test_sg()] Starting MM2S transfer\n");
    this->enable_mm2s_intr();
    this->halt_mm2s();
    this->write_dma(MM2S_CURDESC, mm2s_bds[0]->get_phys_address());
    this->start_mm2s();
    this->write_dma(MM2S_TAILDESC, mm2s_bds[end_idx]->get_phys_address());

    
    // Poll for completion by polling the cmplt bit in the mm2s_status
    uint32_t mm2s_tries = 0;
    while(!(get_bd_cmplt_bit(last_bd_mm2s))){
        mm2s_tries++;
        if(mm2s_tries > 100000){
            printf("ERROR [AXIDMA::self_test_sg()] MM2S transfer timed out\n");
            return -1;
        }
    }

    // Start the s2mm transfer
    printf("INFO [AXIDMA::self_test_sg()] Starting S2MM transfer\n");
    this->enable_s2mm_intr();
    this->halt_s2mm();
    this->write_dma(S2MM_CURDESC, s2mm_bds[0]->get_phys_address());
    this->start_s2mm();
    this->write_dma(S2MM_TAILDESC, s2mm_bds[end_idx]->get_phys_address());

    // Poll for completion
    uint32_t s2mm_tries = 0;
    while(!(get_bd_cmplt_bit(last_bd_s2mm))){
        s2mm_tries++;
        if(s2mm_tries > 100000){
            printf("ERROR [AXIDMA::self_test_sg()] S2MM transfer timed out\n");
            return -1;
        }
    }

    // clear the complete bits in all BD's

    // Check the src and dest block match data
    for(uint32_t i = 0; i < DMA_SELF_TEST_SG_LEN / 4; i++){
        uint32_t src_word = 0xbeefbeef, dst_word = 0xbeefbeef;
        src_block->read_word(i*4, &src_word);
        dst_block->read_word(i*4, &dst_word);
        if(src_word != dst_word){
            printf("ERROR [AXIDMA::self_test_sg()] Self test failed. Data mismatch at index %d: Expected 0x%08X, got 0x%08X\n", 
                i,
                src_word, 
                dst_word);
            PMM.free(src_block);
            PMM.free(dst_block);
            return -1;
        }
    }

    printf("INFO [AXIDMA::self_test_sg()] Scatter gather self test passed!!!\n");

    PMM.free(src_block);
    PMM.free(dst_block);
    for(uint32_t i = 0; i < DMA_SELF_TEST_SG_NUM_BDS; i++){
        PMM.free(mm2s_bds[i]);
        PMM.free(s2mm_bds[i]);
    }

    return 0;
}

#endif // DMA_SG_MODE

int AXIDMA::self_test(){

    srand(time(NULL));

    #ifndef DMA_SG_MODE
    return self_test_dr();
    #else
    return self_test_sg();
    #endif
}
