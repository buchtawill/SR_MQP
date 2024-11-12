#include <stdio.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <stdint.h>

#include "main.h"

#define AXI_BASE_ADDR ADD_MULT_BASE_ADDR
#define AXI_SIZE      0x10000     // Define the size of the mapped region (64 KB)

#define A 32
#define B 4

int main(int argc, char **argv){

    printf("INFO [main.c] Starting test application\n");

    int fd = open("/dev/mem", O_RDWR | O_SYNC);
    if (fd < 0) {
        printf("ERROR [main.c] Failed to open /dev/mem\n");
        return -1;
    }
    
    // Map memory
    void *ptr = mmap(NULL, AXI_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, AXI_BASE_ADDR);
    if (ptr == MAP_FAILED) {
        printf("ERROR [main.c] Failed to map memory\n");
        close(fd);
        return -1;
    }

    uint32_t *p_mem32 = (uint32_t*) ptr;

    p_mem32[DATA_A_OFFSET/4] = A;
    p_mem32[DATA_B_OFFSET/4] = B;

    uint32_t add_result = p_mem32[ADD_RESULT_OFFSET/4];
    uint32_t sub_result = p_mem32[SUB_RESULT_OFFSET/4];
    uint32_t mul_result = p_mem32[MUL_RESULT_OFFSET/4];
    uint32_t div_result = p_mem32[DIV_RESULT_OFFSET/4];

    munmap(ptr, AXI_SIZE);
    close(fd);

    printf("INFO [main] A:   %d\n", A);
    printf("INFO [main] B:   %d\n", B);
    printf("INFO [main] A+B: %d\n", add_result);
    printf("INFO [main] A-B: %d\n", sub_result);
    printf("INFO [main] A*B: %d\n", mul_result);
    printf("INFO [main] A/B: %d\n", div_result);

    return 0;
}