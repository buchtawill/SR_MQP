

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <termios.h>
#include <sys/mman.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>
#include "axi-dma.h"

void print_mem(void *virtual_address, int byte_count)
{
	char *data_ptr = (char*)virtual_address;

	for(int i=0;i<byte_count;i++){
		printf("%02X", data_ptr[i]);

		// print a space every 4 bytes (0 indexed)
		if(i%4==3){
			printf(" ");
		}
	}

	printf("\n");
}

int main(int argc, char *argv[]){
	
	// Open /dev/mem
	printf("INFO [v4l-to-fb0-dma] Opening /dev/mem\n");
	int mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if(mem_fd < 0){
		printf("ERROR [v4l-to-fb0-dma] Failed to open /dev/mem: %s\n", strerror(errno));
		return -1;
	}

	//axi_dma1 configured as NOT scatter gather, MM2S --> FIFO --> S2MM
	AXIDMA dma1(DMA_1_AXI_LITE_BASE, mem_fd);
	printf("INFO [v4l-to-fb0-dma] Created DMA object with base address 0x%08X\n", dma1.getBaseAddress());
	if(dma1.initialize() != 0){
		printf("ERROR [v4l-to-fb0-dma] Failed to initialize DMA object\n");
		return -1;
	}

	//For the first test, just run a simple DMA transfer from the source of PL reserved memory to the dest
	// Access reserved memory for the PL
	printf("INFO [v4l-to-fb0-dma] Memory mapping source DRAM (for MM2S)\n");
    uint32_t *virtual_src_addr = (uint32_t*)mmap(
		NULL,                         // Let the kernel decide the virtual address
		65535,                        // Size of memory to map 
		PROT_READ | PROT_WRITE,       // Permissions: read and write
		MAP_SHARED,                   // Changes are shared with other mappings
		mem_fd,                   // File descriptor for /dev/mem
		VIRTUAL_SRC_ADDR              // Physical address of the reserved memory
	);
	if(virtual_src_addr == MAP_FAILED){
		printf("ERROR [v4l-to-fb0-dma] Failed to map MM2S source address register block: %s\n", strerror(errno));
		return -1;
	}

	printf("INFO [v4l-to-fb0-dma] Memory mapping the S2MM destination address register block.\n");
    uint32_t *virtual_dst_addr = (uint32_t *)mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, VIRTUAL_DST_ADDR);
	if(virtual_dst_addr == MAP_FAILED){
		printf("ERROR [v4l-to-fb0-dma] Failed to map S2MM destination address register block: %s\n", strerror(errno));
		return -1;
	}

	virtual_src_addr[0] = 0xDEADBEEF;
	virtual_src_addr[1] = 0xFEEDFACE;
	virtual_src_addr[2] = 0xF00DCAFE;
	virtual_src_addr[3] = 0xBEEFBEEF;
	virtual_src_addr[4] = 0x12345678;
	virtual_src_addr[5] = 0x87654321;
	virtual_src_addr[6] = 0xBEEFCAFE;
	virtual_src_addr[7] = 0xFACEBEEF;

	printf("INFO [v4l-to-fb0-dma] Clearing the destination block\n");
    memset(virtual_dst_addr, 0, 32);

	printf("INFO [v4l-to-fb0-dma] Source memory block data:      ");
	print_mem(virtual_src_addr, 32);

    printf("INFO [v4l-to-fb0-dma] Destination memory block data: ");
	print_mem(virtual_dst_addr, 32);

	dma1.print_mm2s_status();
	dma1.print_s2mm_status();
	dma1.reset_dma();
	dma1.halt_dma();
	dma1.enable_all_intr();
	dma1.print_mm2s_status();
	dma1.print_s2mm_status();
	dma1.set_mm2s_src(VIRTUAL_SRC_ADDR);
	dma1.set_s2mm_dest(VIRTUAL_DST_ADDR);
	printf("INFO [v4l-to-fb0-dma] Set MM2S src addr to 0x%08X\n", dma1.read_dma(MM2S_SA_LSB32));
	printf("INFO [v4l-to-fb0-dma] Set S2MM dst addr to 0x%08X\n", dma1.read_dma(S2MM_DA_LSB32));
	dma1.print_mm2s_status();
	dma1.print_s2mm_status();
	dma1.set_mm2s_len(32);
	dma1.set_s2mm_len(32);
	dma1.print_mm2s_status();
	dma1.print_s2mm_status();
	dma1.start_mm2s();
	dma1.start_s2mm();
	dma1.print_mm2s_status();
	dma1.print_s2mm_status();
	dma1.wait_for_channel_completion(MM2S_DMASR, 10);
	dma1.wait_for_channel_completion(S2MM_DMASR, 10);

	printf("INFO [v4l-to-fb0-dma] Destination memory block: ");
	print_mem(virtual_dst_addr, 32);

	return 0;
}


