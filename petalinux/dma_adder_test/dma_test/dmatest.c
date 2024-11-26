/*
* Copyright (C) 2013 - 2016  Xilinx, Inc.  All rights reserved.
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL XILINX  BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
* CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*
* Except as contained in this notice, the name of the Xilinx shall not be used
* in advertising or otherwise to promote the sale, use or other dealings in this
* Software without prior written authorization from Xilinx.
*

https://www.hackster.io/whitney-knitter/introduction-to-using-axi-dma-in-embedded-linux-5264ec#code

Modified by Will Buchta 11/12/2024
*/

#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <termios.h>
#include <sys/mman.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>

#define MM2S_CONTROL_REGISTER       0x00
#define MM2S_STATUS_REGISTER        0x04
#define MM2S_SRC_ADDRESS_REGISTER   0x18
#define MM2S_TRNSFR_LENGTH_REGISTER 0x28

#define S2MM_CONTROL_REGISTER       0x30
#define S2MM_STATUS_REGISTER        0x34
#define S2MM_DST_ADDRESS_REGISTER   0x48
#define S2MM_BUFF_LENGTH_REGISTER   0x58

#define IOC_IRQ_FLAG                1<<12
#define IDLE_FLAG                   1<<1

#define STATUS_HALTED               0x00000001
#define STATUS_IDLE                 0x00000002
#define STATUS_SG_INCLDED           0x00000008
#define STATUS_DMA_INTERNAL_ERR     0x00000010
#define STATUS_DMA_SLAVE_ERR        0x00000020
#define STATUS_DMA_DECODE_ERR       0x00000040
#define STATUS_SG_INTERNAL_ERR      0x00000100
#define STATUS_SG_SLAVE_ERR         0x00000200
#define STATUS_SG_DECODE_ERR        0x00000400
#define STATUS_IOC_IRQ              0x00001000
#define STATUS_DELAY_IRQ            0x00002000
#define STATUS_ERR_IRQ              0x00004000

#define HALT_DMA                    0x00000000
#define RUN_DMA                     0x00000001
#define RESET_DMA                   0x00000004
#define ENABLE_IOC_IRQ              0x00001000
#define ENABLE_DELAY_IRQ            0x00002000
#define ENABLE_ERR_IRQ              0x00004000
#define ENABLE_ALL_IRQ              0x00007000

#define DMA_AXI_LITE_BASE			0xA0010000
#define ADD_MULT_BASE_ADDR  		0xA0000000

#define VIRTUAL_SRC_ADDR 			0x0e000000
#define VIRTUAL_DST_ADDR 			0x0f000000

unsigned int write_dma(unsigned int *virtual_addr, int offset, unsigned int value)
{
    virtual_addr[offset>>2] = value;

    return 0;
}

unsigned int read_dma(unsigned int *virtual_addr, int offset)
{
    return virtual_addr[offset>>2];
}

void dma_s2mm_status(unsigned int *virtual_addr)
{
    unsigned int status = read_dma(virtual_addr, S2MM_STATUS_REGISTER);

    printf("INFO [dmatest.c::dma_s2mm_status()] S2MM status (0x%08x@0x%02x):", status, S2MM_STATUS_REGISTER);

    if (status & STATUS_HALTED) {
		printf(" Halted.\n");
	} else {
		printf(" Running.\n");
	}

    if (status & STATUS_IDLE) {
		printf(" Idle.\n");
	}

    if (status & STATUS_SG_INCLDED) {
		printf(" SG is included.\n");
	}

    if (status & STATUS_DMA_INTERNAL_ERR) {
		printf(" DMA internal error.\n");
	}

    if (status & STATUS_DMA_SLAVE_ERR) {
		printf(" DMA slave error.\n");
	}

    if (status & STATUS_DMA_DECODE_ERR) {
		printf(" DMA decode error.\n");
	}

    if (status & STATUS_SG_INTERNAL_ERR) {
		printf(" SG internal error.\n");
	}

    if (status & STATUS_SG_SLAVE_ERR) {
		printf(" SG slave error.\n");
	}

    if (status & STATUS_SG_DECODE_ERR) {
		printf(" SG decode error.\n");
	}

    if (status & STATUS_IOC_IRQ) {
		printf(" IOC interrupt occurred.\n");
	}

    if (status & STATUS_DELAY_IRQ) {
		printf(" Interrupt on delay occurred.\n");
	}

    if (status & STATUS_ERR_IRQ) {
		printf(" Error interrupt occurred.\n");
	}
}

void dma_mm2s_status(unsigned int *virtual_addr)
{
    unsigned int status = read_dma(virtual_addr, MM2S_STATUS_REGISTER);

    printf("INFO [dmatest.c::dma_mm2s_status()] MM2S status (0x%08x@0x%02x):", status, MM2S_STATUS_REGISTER);

    if (status & STATUS_HALTED) {
		printf(" Halted.\n");
	} else {
		printf(" Running.\n");
	}

    if (status & STATUS_IDLE) {
		printf(" Idle.\n");
	}

    if (status & STATUS_SG_INCLDED) {
		printf(" SG is included.\n");
	}

    if (status & STATUS_DMA_INTERNAL_ERR) {
		printf(" DMA internal error.\n");
	}

    if (status & STATUS_DMA_SLAVE_ERR) {
		printf(" DMA slave error.\n");
	}

    if (status & STATUS_DMA_DECODE_ERR) {
		printf(" DMA decode error.\n");
	}

    if (status & STATUS_SG_INTERNAL_ERR) {
		printf(" SG internal error.\n");
	}

    if (status & STATUS_SG_SLAVE_ERR) {
		printf(" SG slave error.\n");
	}

    if (status & STATUS_SG_DECODE_ERR) {
		printf(" SG decode error.\n");
	}

    if (status & STATUS_IOC_IRQ) {
		printf(" IOC interrupt occurred.\n");
	}

    if (status & STATUS_DELAY_IRQ) {
		printf(" Interrupt on delay occurred.\n");
	}

    if (status & STATUS_ERR_IRQ) {
		printf(" Error interrupt occurred.\n");
	}
}

int dma_mm2s_sync(unsigned int *virtual_addr)
{
    unsigned int mm2s_status =  read_dma(virtual_addr, MM2S_STATUS_REGISTER);

	// sit in this while loop as long as the status does not read back 0x00001002 (4098)
	// 0x00001002 = IOC interrupt has occured and DMA is idle
	while(!(mm2s_status & IOC_IRQ_FLAG) || !(mm2s_status & IDLE_FLAG))
	{
        dma_s2mm_status(virtual_addr);
        dma_mm2s_status(virtual_addr);

        mm2s_status =  read_dma(virtual_addr, MM2S_STATUS_REGISTER);
		usleep(500000); // 0.5 seconds
    }

	return 0;
}

int dma_s2mm_sync(unsigned int *virtual_addr)
{
    unsigned int s2mm_status = read_dma(virtual_addr, S2MM_STATUS_REGISTER);

	// sit in this while loop as long as the status does not read back 0x00001002 (4098)
	// 0x00001002 = IOC interrupt has occured and DMA is idle
	while(!(s2mm_status & IOC_IRQ_FLAG) || !(s2mm_status & IDLE_FLAG))
	{
        dma_s2mm_status(virtual_addr);
        dma_mm2s_status(virtual_addr);

        s2mm_status = read_dma(virtual_addr, S2MM_STATUS_REGISTER);
		usleep(500000); // 0.5 seconds
    }

	return 0;
}

void print_mem(void *virtual_address, int byte_count)
{
	char *data_ptr = virtual_address;

	for(int i=0;i<byte_count;i++){
		printf("%02X", data_ptr[i]);

		// print a space every 4 bytes (0 indexed)
		if(i%4==3){
			printf(" ");
		}
	}

	printf("\n");
}

int main()
{

	// FIFO is configured to be 256 entries deep.
	// AXIS FPU configured to calculate float of input.

    printf("INFO [dmatest.c] Running DMA transfer test application...\n");
    printf("INFO [dmatest.c] DMA Stream will compute the square root of the given inputs as 32 bit floats.\n");

	printf("INFO [dmatest.c] Opening a character device file of the Arty's DDR memeory...\n");
	int ddr_memory = open("/dev/mem", O_RDWR | O_SYNC);
	if(ddr_memory < 0){
		printf("ERROR [dmatest.c] Failed to open /dev/mem: %s\n", strerror(errno));
		return -1;
	}

	sleep(1);

	printf("INFO [dmatest.c] Memory mapping the address of the DMA AXI IP via its AXI lite control interface register block.\n");
    uint32_t *dma_virtual_addr = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, DMA_AXI_LITE_BASE);
	if(dma_virtual_addr == MAP_FAILED){
		printf("ERROR [dmatest.c] Failed to map DMA AXI Lite register block: %s\n", strerror(errno));
		return -1;
	}

	sleep(1);

	printf("INFO [dmatest.c] Memory mapping the MM2S source address register block.\n");
    float *virtual_src_addr  = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, VIRTUAL_SRC_ADDR);
	if(virtual_src_addr == MAP_FAILED){
		printf("ERROR [dmatest.c] Failed to map MM2S source address register block: %s\n", strerror(errno));
		return -1;
	}

	sleep(1);

	printf("INFO [dmatest.c] Memory mapping the S2MM destination address register block.\n");
    uint32_t *virtual_dst_addr = mmap(NULL, 65535, PROT_READ | PROT_WRITE, MAP_SHARED, ddr_memory, VIRTUAL_DST_ADDR);
	if(virtual_dst_addr == MAP_FAILED){
		printf("ERROR [dmatest.c] Failed to map S2MM destination address register block: %s\n", strerror(errno));
		return -1;
	}

	sleep(1);

	printf("INFO [dmatest.c] Writing data to source block\n");

	virtual_src_addr[0] = 1.0f;
	virtual_src_addr[1] = 2.0f;
	virtual_src_addr[2] = 3.0f;
	virtual_src_addr[3] = 4.0f;
	virtual_src_addr[4] = 10.0f;
	virtual_src_addr[5] = 100.0f;
	virtual_src_addr[6] = 256.0f;
	virtual_src_addr[7] = 482.0f;

	sleep(1);

	printf("INFO [dmatest.c] Clearing the destination block\n");
    memset(virtual_dst_addr, 0, 32);

	sleep(1);

    printf("INFO [dmatest.c] Source memory block data:      ");
	print_mem(virtual_src_addr, 32);

	sleep(1);

    printf("INFO [dmatest.c] Destination memory block data: ");
	print_mem(virtual_dst_addr, 32);

	sleep(1);

    printf("INFO [dmatest.c] Resetting DMA\n");
    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, RESET_DMA);
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, RESET_DMA);
    dma_s2mm_status(dma_virtual_addr);
    dma_mm2s_status(dma_virtual_addr);

	printf("INFO [dmatest.c] S2MM Control Register: 0x%08x\n", read_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER));
	printf("INFO [dmatest.c] MM2S Control Register: 0x%08x\n", read_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER));

	sleep(1);

	printf("INFO [dmatest.c] Halting DMA.\n");
    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, HALT_DMA);
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, HALT_DMA);
    dma_s2mm_status(dma_virtual_addr);
    dma_mm2s_status(dma_virtual_addr);

	sleep(1);

	printf("INFO [dmatest.c] Enabling all interrupts.\n");
    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, ENABLE_ALL_IRQ);
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, ENABLE_ALL_IRQ);
    dma_s2mm_status(dma_virtual_addr);
    dma_mm2s_status(dma_virtual_addr);

	sleep(1);

    printf("INFO [dmatest.c] Writing source address of the data from MM2S in DDR...\n");
    write_dma(dma_virtual_addr, MM2S_SRC_ADDRESS_REGISTER, VIRTUAL_SRC_ADDR);
    dma_mm2s_status(dma_virtual_addr);

	printf("INFO [dmatest.c] MM2S source address register: 0x%08x\n", read_dma(dma_virtual_addr, MM2S_SRC_ADDRESS_REGISTER));

	sleep(1);

    printf("INFO [dmatest.c] Writing the destination address for the data from S2MM in DDR...\n");
    write_dma(dma_virtual_addr, S2MM_DST_ADDRESS_REGISTER, VIRTUAL_DST_ADDR);
    dma_s2mm_status(dma_virtual_addr);

	sleep(1);

	printf("INFO [dmatest.c] Running MM2S channel.\n");
    write_dma(dma_virtual_addr, MM2S_CONTROL_REGISTER, RUN_DMA);
    dma_mm2s_status(dma_virtual_addr);

	sleep(1);

	printf("INFO [dmatest.c] Run S2MM channel.\n");
    write_dma(dma_virtual_addr, S2MM_CONTROL_REGISTER, RUN_DMA);
    dma_s2mm_status(dma_virtual_addr);

	sleep(1);

    printf("INFO [dmatest.c] Writing MM2S transfer length of 32 bytes...\n");
    write_dma(dma_virtual_addr, MM2S_TRNSFR_LENGTH_REGISTER, 32);
    dma_mm2s_status(dma_virtual_addr);

	sleep(1);

    printf("INFO [dmatest.c] Writing S2MM transfer length of 32 bytes...\n");
    write_dma(dma_virtual_addr, S2MM_BUFF_LENGTH_REGISTER, 32);
    dma_s2mm_status(dma_virtual_addr);

	sleep(1);

    printf("INFO [dmatest.c] Waiting for MM2S synchronization...\n");
    dma_mm2s_sync(dma_virtual_addr);

	sleep(1);

    printf("INFO [dmatest.c] Waiting for S2MM sychronization...\n");
    dma_s2mm_sync(dma_virtual_addr);

	sleep(1);

    dma_s2mm_status(dma_virtual_addr);
    dma_mm2s_status(dma_virtual_addr);

	sleep(1);

    printf("INFO [dmatest.c] Destination memory block: ");
	print_mem(virtual_dst_addr, 32);

	printf("\n");

	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[0], ((float*)virtual_dst_addr)[0]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[1], ((float*)virtual_dst_addr)[1]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[2], ((float*)virtual_dst_addr)[2]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[3], ((float*)virtual_dst_addr)[3]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[4], ((float*)virtual_dst_addr)[4]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[5], ((float*)virtual_dst_addr)[5]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[6], ((float*)virtual_dst_addr)[6]);
	printf("sqrt(%f): %f\n", ((float*)virtual_src_addr)[7], ((float*)virtual_dst_addr)[7]);

    return 0;
}
