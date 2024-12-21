#include "axi-dma.h"
#include "stdint.h"
#include <stdio.h>
#include <sys/mman.h> //for mmap
#include <errno.h>
#include <string.h> //for strerror
#include <unistd.h> // for usleep
#include <stdlib.h> // for rand

AXIDMA::AXIDMA(uint32_t base_addr, int dev_mem_fd){
    this->base_address = base_addr;
    this->mem_fd = dev_mem_fd;
}

AXIDMA::~AXIDMA(){
    // printf("INFO [AXIDMA::~AXIDMA] Deleting instance with base address 0x%08X\n", this->base_address);
    munmap((void*)dma_phys_addr, DMA_ADDRESS_SPACE_SIZE);
}

int AXIDMA::initialize(){
    this->dma_phys_addr = (uint32_t *)mmap(
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

int AXIDMA::sync_channel(uint32_t channel_status_reg, uint32_t max_tries){
    uint32_t status = read_dma(channel_status_reg);

    // printf("INFO [AXIDMA::sync_channel()] Waiting for channel to complete. Initial status: \n");
    // print_status(channel_status_reg);

	uint32_t count = 0;
	while(!(status & IOC_IRQ_FLAG)){
        // usleep(10000);

		status = read_dma(channel_status_reg);

        // if(channel_status_reg == MM2S_DMASR){
        //     this->print_status(MM2S_DMASR);
        // }
        // else if(channel_status_reg == S2MM_DMASR){
        //     this->print_status(S2MM_DMASR);
        // }
        // else{
        //     printf("ERROR [AXIDMA::sync_channel()] Invalid channel status register\n");
        //     return -1;
        // }

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

void AXIDMA::enable_mm2s_intr(){
    write_dma(MM2S_DMACR, ENABLE_ALL_IRQ);
}

void AXIDMA::enable_s2mm_intr(){
    write_dma(S2MM_DMACR, ENABLE_ALL_IRQ);
}

int AXIDMA::transfer_mm2s(uint32_t src_addr, uint32_t len, bool block){

    this->clear_irq_bit(MM2S_DMASR);

    this->reset_mm2s();
    this->halt_mm2s();
    this->enable_mm2s_intr();
    this->set_mm2s_src(src_addr);

    this->start_mm2s();

    // Setting the length register must be the last step
    this->set_mm2s_len(len);

    if(block) this->sync_channel(MM2S_DMASR);

    return 0;
}

int AXIDMA::transfer_s2mm(uint32_t dst_addr, uint32_t len, bool block){

    this->clear_irq_bit(S2MM_DMASR);

    this->reset_s2mm();
    this->halt_s2mm();
    this->enable_s2mm_intr();
    this->set_s2mm_dest(dst_addr);

    this->start_s2mm();

    // Setting the length register must be the last step
    this->set_s2mm_len(len);

    if(block) this->sync_channel(S2MM_DMASR);
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
    // if(block){
    num_tries1 = this->sync_channel(MM2S_DMASR);
    if(num_tries1 < 0){
        return -1;
    }

    num_tries2 = this->sync_channel(S2MM_DMASR);
    if(num_tries2 < 0){
        return -1;
    }
    // }
    return num_tries1 + num_tries2;
}

int AXIDMA::self_test(){

    //mmap the source and destination addresses to the process
    uint32_t *src_addr = (uint32_t *)mmap(
        NULL,                   // Let the kernel decide the virtual address
        DMA_SELF_TEST_LEN,      // Size of memory to map 
        PROT_READ | PROT_WRITE, // Permissions: read and write
        MAP_SHARED,             // Changes are shared with other mappings
        this->mem_fd,           // File descriptor for /dev/mem
        DMA_SELF_TEST_SRC_ADDR  // Physical address of the reserved memory
    );

    uint32_t *dst_addr = (uint32_t *)mmap(
        NULL,                   // Let the kernel decide the virtual address
        DMA_SELF_TEST_LEN,      // Size of memory to map 
        PROT_READ | PROT_WRITE, // Permissions: read and write
        MAP_SHARED,             // Changes are shared with other mappings
        this->mem_fd,           // File descriptor for /dev/mem
        DMA_SELF_TEST_DST_ADDR  // Physical address of the reserved memory
    );

    // Write some random data to the source address, copy it to the dst address, and check if it's the same
    // If size is 0x1000, that is 4096 bytes, so 1024 uint32_t
    for(int i = 0; i < DMA_SELF_TEST_LEN / 4; i++){
        src_addr[i] = (uint32_t)rand();
        // src_addr[i] = i;
    }

    // printf("INFO [AXIDMA::self_test()] Status registers before transfer: \n");
    // this->print_status(MM2S_DMASR);
    // this->print_status(S2MM_DMASR);

    this->transfer(DMA_SELF_TEST_SRC_ADDR, DMA_SELF_TEST_DST_ADDR, DMA_SELF_TEST_LEN, true);

    // printf("INFO [AXIDMA::self_test()] Status registers immediately after transfer (should be idle and no IOC): \n");
    // this->print_status(MM2S_DMASR);
    // this->print_status(S2MM_DMASR);

    // usleep(100000);

    // printf("INFO [AXIDMA::self_test()] Status registers 100ms after transfer: \n");
    // this->print_status(MM2S_DMASR);
    // this->print_status(S2MM_DMASR);

    // this->clear_irq_bits();
    // printf("INFO [AXIDMA::self_test()] Status registers after clearing IRQ bits: \n");
    // this->print_status(MM2S_DMASR);
    // this->print_status(S2MM_DMASR);

    // Check if the data is the same
    for(int i = 0; i < DMA_SELF_TEST_LEN / 4; i++){
        if(src_addr[i] != dst_addr[i]){
            printf("ERROR [AXIDMA::self_test()] Self test failed. Data mismatch at index %d: Expected 0x%08X, got 0x%08X\n", 
                i, 
                src_addr[i], 
                dst_addr[i]);
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

    printf("INFO [AXIDMA::self_test()] Self test passed\n");

    munmap(src_addr, DMA_SELF_TEST_LEN);
    munmap(dst_addr, DMA_SELF_TEST_LEN);

    return 0;
}