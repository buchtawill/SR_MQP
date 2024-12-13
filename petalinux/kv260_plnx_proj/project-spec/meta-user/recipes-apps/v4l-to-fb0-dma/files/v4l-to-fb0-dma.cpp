

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

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <linux/fb.h>

#include <signal.h> //for sigint

#define INPUT_WIDTH  720
#define INPUT_HEIGHT 576
#define INPUT_PIXEL_FORMAT V4L2_PIX_FMT_YUYV // Match the framebuffer pixel format (e.g., RGB565)

// Global variable - use with caution! This should only ever be set to true 
bool die_flag = false;

// Convert YUYV to RGB565
void yuyv_to_rgb565(uint8_t *yuyv, uint16_t *rgb565, int width, int height) {
    for (int i = 0; i < width * height; i += 2) {
        int y0 = yuyv[i * 2];
        int u = yuyv[i * 2 + 1] - 128;
        int y1 = yuyv[i * 2 + 2];
        int v = yuyv[i * 2 + 3] - 128;

        int r0 = y0 + 1.403 * v;
        int g0 = y0 - 0.344 * u - 0.714 * v;
        int b0 = y0 + 1.770 * u;

        int r1 = y1 + 1.403 * v;
        int g1 = y1 - 0.344 * u - 0.714 * v;
        int b1 = y1 + 1.770 * u;

        r0 = r0 < 0 ? 0 : (r0 > 255 ? 255 : r0);
        g0 = g0 < 0 ? 0 : (g0 > 255 ? 255 : g0);
        b0 = b0 < 0 ? 0 : (b0 > 255 ? 255 : b0);

        r1 = r1 < 0 ? 0 : (r1 > 255 ? 255 : r1);
        g1 = g1 < 0 ? 0 : (g1 > 255 ? 255 : g1);
        b1 = b1 < 0 ? 0 : (b1 > 255 ? 255 : b1);

        rgb565[i] = ((r0 & 0xF8) << 8) | ((g0 & 0xFC) << 3) | (b0 >> 3);
        rgb565[i + 1] = ((r1 & 0xF8) << 8) | ((g1 & 0xFC) << 3) | (b1 >> 3);
    }
}

void die_with_error(const char* error){

    die_flag = true;
}

void print_mem(void *virtual_address, int byte_count){
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

void sigint_handler(int sig){
    die_flag = true;
}

int main(int argc, char *argv[]){

	// if(argc < 3) {
    //     printf("Usage: %s <video_device> <fb_device>\n", argv[0]);
    //     printf("Example: sudo %s /dev/video0 /dev/fb0\n", argv[0]);
    //     return 1;
    // }
	// const char *video_device = argv[1];
    // const char *fb_device = argv[2];
    const char *video_device = "/dev/video0";
    const char *fb_device = "/dev/fb0";

    signal(SIGINT, sigint_handler);
	
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
    if(dma1.self_test() < 0){
        printf("ERROR [v4l-to-fb0-dma] DMA self test failed\n");
    }

    // Open the video device
    int video_fd = open(video_device, O_RDWR);
    if (video_fd == -1) {
        perror("Opening video device");
        printf("ERROR [v4l-to-fb0-dma.cpp] Error opening video device: %s\n", strerror(errno));
        return 1;
    }

    // Open the framebuffer device
    int fb_fd = open(fb_device, O_RDWR);
    if (fb_fd == -1) {
        perror("Opening framebuffer device");
        close(video_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error opening framebuffer device: %s\n", strerror(errno));
        return 1;
    }

    // Query the framebuffer information
    struct fb_var_screeninfo fb_info;
    if (ioctl(fb_fd, FBIOGET_VSCREENINFO, &fb_info) == -1) {
        perror("Getting framebuffer info");
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error getting framebuffer info: %s\n", strerror(errno));
        return 1;
    }

    if (fb_info.bits_per_pixel != 16) {
        fprintf(stderr, "Unsupported framebuffer format (not 16 bpp YUYV)\n");
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Unsupported framebuffer format\n");
        return 1;
    }

    // Set video format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = INPUT_WIDTH;
    fmt.fmt.pix.height = INPUT_HEIGHT;
    fmt.fmt.pix.pixelformat = INPUT_PIXEL_FORMAT;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    printf("INFO [v4l-to-fb0-dma] Setting video format\n");
    if (ioctl(video_fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Setting video format");
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error setting video format: %s\n", strerror(errno));
        return 1;
    }

    // Request buffers from the video device
    printf("INFO [v4l-to-fb0-dma] Requesting buffer from %s\n", argv[1]);
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1; // Use 1 buffer
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(video_fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Requesting buffer");
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error requesting v4l2 buffer: %s\n", strerror(errno));
        return 1;
    }

    // Map the video buffer
    printf("INFO [v4l-to-fb0-dma] Mapping video buffer\n");
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    if (ioctl(video_fd, VIDIOC_QUERYBUF, &buf) == -1) {
        perror("Querying buffer");
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error querying v4l2 buffer: %s\n", strerror(errno));
        return 1;
    }

    void *video_buffer = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, video_fd, buf.m.offset);
    if (video_buffer == MAP_FAILED) {
        // perror("Mapping video buffer");
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error mapping video buffer: %s\n", strerror(errno));
        return 1;
    }

    // Query framebuffer information
    struct fb_fix_screeninfo finfo;
    if (ioctl(fb_fd, FBIOGET_FSCREENINFO, &finfo) == -1) {
        perror("Error reading fixed screen info");
        return 1;
    }
    printf("INFO [v4l-to-fb0-dma] Frame Buffer physical address: 0x%lx\n", finfo.smem_start);
    // Map the framebuffer
    printf("INFO [v4l-to-fb0-dma] Mapping framebuffer physical address\n");
    size_t fb_size = (fb_info.yres_virtual * fb_info.xres_virtual * fb_info.bits_per_pixel) / 8;
    printf("INFO [v4l-to-fb0-dma] Framebuffer size: %lu\n", fb_size);
    void *fb_buffer = mmap(NULL, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, finfo.smem_start);

    if (fb_buffer == MAP_FAILED) {
        // perror("Mapping framebuffer");
        munmap(video_buffer, buf.length);
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error mapping framebuffer: %s\n", strerror(errno));
        return 1;
    }

    // Start video streaming
    printf("INFO [v4l-to-fb0-dma] Starting video streaming from v4l2\n");
    if (ioctl(video_fd, VIDIOC_STREAMON, &buf.type) == -1) {
        perror("Starting video streaming");
        munmap(video_buffer, buf.length);
        munmap(fb_buffer, fb_size);
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error starting video streaming: %s\n", strerror(errno));
        return 1;
    }


    // uint16_t rgb565_buf[INPUT_WIDTH * INPUT_HEIGHT];
    // Create a buffer for rgb565
    
    uint16_t *rgb565_buf = (uint16_t *)mmap(NULL, KERNEL_RSVD_MEM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, mem_fd, KERNEL_RSVD_MEM_BASE);
    if (rgb565_buf == MAP_FAILED) {
        perror("Mapping reserved memory");
        munmap(video_buffer, buf.length);
        munmap(fb_buffer, fb_size);
        close(video_fd);
        close(fb_fd);
        printf("ERROR [v4l-to-fb0-dma.cpp] Error mapping reserved memory: %s\n", strerror(errno));
        return 1;
    }
    printf("INFO [v4l-to-fb0-dma] Starting continuous loop of reading frames...\n");
    
    uint32_t frame_loop_count = 0;
    while(die_flag == false){
        // printf("INFO [v4l-to-fb0-dma] Capturing frame from v4l2\n");
        if (ioctl(video_fd, VIDIOC_QBUF, &buf) == -1) {
            perror("Queueing buffer");
            munmap(video_buffer, buf.length);
            munmap(fb_buffer, fb_size);
            close(video_fd);
            close(fb_fd);
            printf("ERROR [v4l-to-fb0-dma.cpp] Error queueing buffer: %s\n", strerror(errno));
            return 1;
        }
        
        if (ioctl(video_fd, VIDIOC_DQBUF, &buf) == -1) {
            perror("Dequeueing buffer");
            munmap(video_buffer, buf.length);
            munmap(fb_buffer, fb_size);
            close(video_fd);
            close(fb_fd);
            printf("ERROR [v4l-to-fb0-dma.cpp] Error dequeueing buffer: %s\n", strerror(errno));
            return 1;
        }

        // Convert the v4l2 input formar to the framebuffer format
        // Copy the video buffer to the framebuffer to capture 1 frame
        
        yuyv_to_rgb565((uint8_t *)video_buffer, rgb565_buf, INPUT_WIDTH, INPUT_HEIGHT);
        // memcpy(fb_buffer, rgb565_buf, buf.bytesused);

        uint16_t *fb_pointer_pix = (uint16_t*)fb_buffer;
        uint32_t rgb565_phys_ptr = KERNEL_RSVD_MEM_BASE;
        uint8_t  bytes_per_pixel = 2;
        for (int row = 0; row < INPUT_HEIGHT; row++) {
            // Copy one row at a time from the rgb565 buf to the frame buffer
            // Each pixel is 2 bytes
            // Non-DMA memcpy:
            // memcpy(&fb_pointer_pix[fb_info.xres_virtual * row], &rgb565_buf[INPUT_WIDTH * row], INPUT_WIDTH * sizeof(uint16_t));
			
            // 32 byte boundary: lower 5 bits are all 0
            uint32_t dst_pix_addr = finfo.smem_start + (fb_info.xres_virtual * row * bytes_per_pixel); // * 2 for bytes per pixel
            rgb565_phys_ptr += INPUT_WIDTH * bytes_per_pixel;
            int result = dma1.transfer(rgb565_phys_ptr, dst_pix_addr, INPUT_WIDTH * bytes_per_pixel, true);
            // printf("DEBUG [v4l-to-fb0-dma] DMA transfer from 0x%08X to 0x%08X. Result: %d\n", rgb565_phys_ptr, dst_pix_addr, result);
            // if(row == 0) printf("INFO [v4l-to-fb0-dma] Destination pixel address: 0x%08X\n", dma1.read_dma(S2MM_DST_ADDRESS_REGISTER));
        }
        frame_loop_count++;
        if(frame_loop_count % 50 == 0){
            printf("INFO [v4l-to-fb0-dma] Frame count: %05u\n", frame_loop_count);
              printf("                      565 mem: ", frame_loop_count);
            for(int i = 0; i < 8; i++){
                printf("%04X", rgb565_buf[i]);
                if((i + 1) % 2 == 0)printf(" ");
            }

            printf("\n                      fb0 mem: ", frame_loop_count);
            for(int i = 0; i < 8; i++){
                printf("%04X", fb_pointer_pix[i]);
                if((i + 1) % 2 == 0)printf(" ");
            }
            printf("\n");
        }
    }

    if(die_flag){
        printf("\nINFO [v4l-to-fb0-dma] Ctrl-C pressed. Exiting...\n");
    }
    // Stop video streaming
    if (ioctl(video_fd, VIDIOC_STREAMOFF, &buf.type) == -1) {
        perror("Stopping video streaming");
        printf("ERROR [v4l-to-fb0-dma.cpp] Error stopping video streaming: %s\n", strerror(errno));
    }

    // Cleanup
    munmap(video_buffer, buf.length);
    munmap(fb_buffer, fb_size);
    close(video_fd);
    close(fb_fd);

	return 0;
}