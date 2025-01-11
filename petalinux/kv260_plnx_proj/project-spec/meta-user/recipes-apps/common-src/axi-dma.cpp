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

    // If using scatter gather mode, allocate memory for the buffer descriptors at the physical address reserved from the kernel
    #ifdef DMA_SG_MODE

    // Map and clear the buffer descriptor memory
    this->mm2s_bd_arr = (DMA_SG_BD *)mmap(
        NULL,                    // Let the kernel decide the virtual address
        DMA_BDR_MM2S_SIZE_BYTES, // Size of memory to map 
        PROT_READ | PROT_WRITE,  // Permissions: read and write
        MAP_SHARED,              // Changes are shared with other mappings
        this->mem_fd,            // File descriptor for /dev/mem
        DMA_BDR_MM2S_BASE        // Physical address of the reserved memory
    );
    if(this->mm2s_bd_arr == MAP_FAILED){
        printf("ERROR [AXIDMA::initialize()] Failed to map MM2S buffer descriptor memory: %s\n", strerror(errno));
        return -1;
    }

    this->s2mm_bd_arr = (DMA_SG_BD *)mmap(
        NULL,                    // Let the kernel decide the virtual address
        DMA_BDR_S2MM_SIZE_BYTES, // Size of memory to map 
        PROT_READ | PROT_WRITE,  // Permissions: read and write
        MAP_SHARED,              // Changes are shared with other mappings
        this->mem_fd,            // File descriptor for /dev/mem
        DMA_BDR_S2MM_BASE        // Physical address of the reserved memory
    );
    if(this->s2mm_bd_arr == MAP_FAILED){
        printf("ERROR [AXIDMA::initialize()] Failed to map S2MM buffer descriptor memory: %s\n", strerror(errno));
        return -1;
    }

    // Clear the buffer descriptor memory
    memset((void *)this->mm2s_bd_arr, 0, DMA_BDR_MM2S_SIZE_BYTES);
    memset((void *)this->s2mm_bd_arr, 0, DMA_BDR_S2MM_SIZE_BYTES);

    #endif

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

#ifdef DMA_SG_MODE

void AXIDMA::create_s2mm_bd_ring(int num_bds){
    // Clear the existing BD memory, and start programming each BD with next pointer and such
    // For s2mm, it does not matter to set the RXSOF or RXEOF bits

    memset((void*)this->s2mm_bd_arr, 0, sizeof(DMA_SG_BD)*num_bds);

    //Set the first RXSOF bit (Not used if not in Micro DMA mode)
    //Same position as tx bit so use that function
    set_txsof_bit(&(this->mm2s_bd_arr[0]), 1);

    for(int i = 0; i < num_bds - 1; i++){
        volatile DMA_SG_BD *current_bd = &(this->s2mm_bd_arr[i]);

        uint32_t next_index = i + 1;
        uint32_t next_address = s2mm_bd_idx_to_addr(next_index);
        current_bd->next_desc_index = next_index;
        current_bd->next_desc_ptr = next_address;
    }

    // Set the last RXEOF bit (Not used if not in Micro DMA mode)
    set_txeof_bit(&(this->mm2s_bd_arr[num_bds - 1]), 1);

    // Set the last next_desc_ptr to loop around
    // Set the last index as well
    this->s2mm_bd_arr[num_bds - 1].next_desc_ptr = DMA_BDR_S2MM_BASE;
    this->s2mm_bd_arr[num_bds - 1].next_desc_index = 0;

    this->s2mm_tail = &(this->s2mm_bd_arr[num_bds - 1]);
    this->s2mm_tail_address = s2mm_bd_idx_to_addr(num_bds-1);
}

void AXIDMA::create_mm2s_bd_ring(int num_bds){
    // Clear the existing BD memory, and start programming each BD with next pointer and such
    // Set the TXSOF and TXEOF bits

    memset((void*)this->mm2s_bd_arr, 0, sizeof(DMA_SG_BD)*num_bds);

    //Set the first TXSOF bit
    set_txsof_bit(&(this->mm2s_bd_arr[0]), 1);

    for(int i = 0; i < num_bds - 1; i++){
        volatile DMA_SG_BD *current_bd = &(this->mm2s_bd_arr[i]);

        uint32_t next_index = i + 1;
        uint32_t next_address = mm2s_bd_idx_to_addr(next_index);
        current_bd->next_desc_index = next_index;
        current_bd->next_desc_ptr = next_address;
    }

    // Set the last TXEOF bit
    set_txeof_bit(&(this->mm2s_bd_arr[num_bds - 1]), 1);

    // Set the last next_desc_ptr to loop around
    // Set the last index as well
    this->mm2s_bd_arr[num_bds - 1].next_desc_ptr = DMA_BDR_MM2S_BASE;
    this->mm2s_bd_arr[num_bds - 1].next_desc_index = 0;

    this->mm2s_tail = &(this->mm2s_bd_arr[num_bds - 1]);
    this->mm2s_tail_address = mm2s_bd_idx_to_addr(num_bds-1);
}

int AXIDMA::transfer_sg(){
    // Start the MM2S transfer, then start the S2MM transfer
    this->clear_irq_bits();

    this->reset_mm2s();
    this->reset_s2mm();

    // Start MM2S transfer
    // Write the first MM2S descriptor to the CURDESC register
    write_dma(MM2S_CURDESC, DMA_BDR_MM2S_BASE);

    this->halt_mm2s();
    this->start_mm2s();
    this->enable_mm2s_intr();

    // Write the tail address to the tail descriptor register
    write_dma(MM2S_TAILDESC, this->mm2s_tail_address);

    // Start S2MM transfer
    // Write the first S2MM descriptor to the CURDESC register
    write_dma(S2MM_CURDESC, DMA_BDR_S2MM_BASE);
    this->halt_s2mm();
    this->start_s2mm();
    this->enable_s2mm_intr();

    // Write the tail address to the tail descriptor register
    write_dma(S2MM_TAILDESC, this->s2mm_tail_address);
}
#endif

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
            printf("ERROR [AXIDMA::self_test()] MM2S transfer failed\n");
            return -1;
        }
    }

    for(int i = 0; i < 28; i++){
        int result = this->transfer_s2mm(dst_block->get_phys_address()+(bytes_per_row * i), bytes_per_row, true);
        if(result < 0){
            printf("ERROR [AXIDMA::self_test()] S2MM transfer failed\n");
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
