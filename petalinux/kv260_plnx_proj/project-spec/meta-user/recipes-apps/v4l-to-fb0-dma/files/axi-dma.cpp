#include "axi-dma.h"
#include "stdint.h"
#include <stdio.h>
#include <sys/mman.h> //for mmap
#include <errno.h>
#include <string.h> //for strerror
#include <unistd.h> // for usleep

AXIDMA::AXIDMA(uint32_t base_addr, int dev_mem_fd){
    this->base_address = base_addr;
    this->mem_fd = dev_mem_fd;
}

AXIDMA::~AXIDMA(){
    printf("INFO [axi-dma.cpp] Destroying instance with base address 0x%08X\n", this->base_address);
    munmap(dma_phys_addr, DMA_ADDRESS_SPACE_SIZE);
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

void AXIDMA::print_mm2s_status(){
    unsigned int status = read_dma(MM2S_DMASR);

    printf("INFO [AXIDMA@0x%08X::print_mm2s_status()] MM2S status (0x%08x@0x%02x):", this->base_address, status, MM2S_DMASR);

    if (status & STATUS_HALTED)           printf(" Halted.                       ");
    else                                  printf(" Running.                      ");
    if (status & STATUS_IDLE)             printf(" | Idle.                       "); 
    if (status & STATUS_SG_INCLDED)       printf(" | SG is included.             ");
    if (status & STATUS_DMA_INTERNAL_ERR) printf(" | DMA internal error.         ");
    if (status & STATUS_DMA_SLAVE_ERR)    printf(" | DMA slave error.            ");
    if (status & STATUS_DMA_DECODE_ERR)   printf(" | DMA decode error.           ");
    if (status & STATUS_SG_INTERNAL_ERR)  printf(" | SG internal error.          ");
    if (status & STATUS_SG_SLAVE_ERR)     printf(" | SG slave error.             ");
    if (status & STATUS_SG_DECODE_ERR)    printf(" | SG decode error.            ");
    if (status & STATUS_IOC_IRQ)          printf(" | IOC interrupt occurred.     ");
    if (status & STATUS_DELAY_IRQ)        printf(" | Interrupt on delay occurred.");
    if (status & STATUS_ERR_IRQ)          printf(" | Error interrupt occurred.   ");
    printf("\n");
}

void AXIDMA::print_s2mm_status(){
    unsigned int status = read_dma(S2MM_DMASR);

    printf("INFO [AXIDMA@0x%08X::print_s2mm_status()] S2MM status (0x%08x@0x%02x):", this->base_address, status, S2MM_DMASR);

    if (status & STATUS_HALTED)           printf(" Halted.  |                    ");
    else                                  printf(" Running. |                    ");
    if (status & STATUS_IDLE)             printf(" | Idle.                       "); 
    if (status & STATUS_SG_INCLDED)       printf(" | SG is included.             ");
    if (status & STATUS_DMA_INTERNAL_ERR) printf(" | DMA internal error.         ");
    if (status & STATUS_DMA_SLAVE_ERR)    printf(" | DMA slave error.            ");
    if (status & STATUS_DMA_DECODE_ERR)   printf(" | DMA decode error.           ");
    if (status & STATUS_SG_INTERNAL_ERR)  printf(" | SG internal error.          ");
    if (status & STATUS_SG_SLAVE_ERR)     printf(" | SG slave error.             ");
    if (status & STATUS_SG_DECODE_ERR)    printf(" | SG decode error.            ");
    if (status & STATUS_IOC_IRQ)          printf(" | IOC interrupt occurred.     ");
    if (status & STATUS_DELAY_IRQ)        printf(" | Interrupt on delay occurred.");
    if (status & STATUS_ERR_IRQ)          printf(" | Error interrupt occurred.   ");
    printf("\n");
}

int AXIDMA::wait_for_channel_completion(uint32_t channel_status_reg, uint32_t max_tries){
    uint32_t status = read_dma(channel_status_reg);

	uint32_t count = 0;
	while(!(status & IOC_IRQ_FLAG)){
        usleep(10000);

		status = read_dma(channel_status_reg);

        if(channel_status_reg == MM2S_DMASR){
            this->print_mm2s_status();
        }
        else if(channel_status_reg == S2MM_DMASR){
            this->print_s2mm_status();
        }
        else{
            printf("ERROR [axi-dma.cpp::wait_for_channel_completion()] Invalid channel status register\n");
            return -1;
        }

        // Sleep for 10ms
		count++;
		if(count == max_tries){
			printf("ERROR [axi-dma.cpp::wait_for_channel_completion()] Timeout occurred. Tried %d times\n", max_tries);
			return -1;
		}
	}

	return count;
}

void AXIDMA::enable_mm2s_intr(){
    write_dma(MM2S_DMACR, ENABLE_ALL_IRQ);
}

void AXIDMA::enable_s2mm_intr(){
    write_dma(S2MM_DMACR, ENABLE_ALL_IRQ);
}