/*

This program uses parts of the reserved memory from the kernel. 
The reserved memory is used to store the RGB565 data that is to be displayed on the framebuffer.
There are KERNEL_RSVD_MEM_SIZE bytes of reserved memory starting at KERNEL_RSVD_MEM_BASE.
This is currently 32MB
The video buffer is 720*576*2 = 720*576*2 bytes ~= 0.8MB

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
#include "axi-dma.h"

#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <linux/fb.h>

#include <signal.h> //for sigint
#include <time.h>   // for jiffies

#include "v4l-to-fb0-dma.h"

// Global variable - use with caution! This should only ever be set to true 
bool die_flag = false;

unsigned long get_jiffies(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (unsigned long)(ts.tv_sec * sysconf(_SC_CLK_TCK) + ts.tv_nsec / (1e9 / sysconf(_SC_CLK_TCK)));
}

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

/**
 * Print a given error message and clean up resources before exiting
 * @param error_msg The error message to print
 * @param err_str The error string to print (optional, given by strerror(errno))
 * @param res Pointer to the resources struct
 * @return None
 */
void die_with_error(const char* error_msg, const char* err_str, Resources* res){
    die_flag = true;

    printf("%s", error_msg);

    if(err_str != nullptr){
        printf("%s\n", err_str);
    }
    else{
        printf("\n");
    }

    cleanup_resources(res);
    exit(-1);
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
    printf("INFO [v4l-to-fb0-dma] Received signal, cleaning up...\n");
    die_flag = true;
}

/**
 * Open dev/mem
 * Open the video device
 * Open the framebuffer device
 * Setup the rgb565 buffer
 * @param p_res Pointer to the resources struct
 * @param video_device The video device
 * @param fb_device The framebuffer device
 * @return None
 */
void init_resources(Resources *p_res, const char* video_device, const char* fb_device){
    p_res->v4l2_fd    = -1;
    p_res->fb_dev_fd  = -1;
    p_res->dev_mem_fd = -1;
    p_res->fb_mem_phys      = nullptr;
    p_res->vid_mem_phys     = nullptr;
    p_res->rgb_565_buf_pix  = nullptr;
    p_res->rgb_565_buf_phys = nullptr;

    memset(&p_res->v4l2_fmt,             0, sizeof(p_res->v4l2_fmt));
    memset(&p_res->v4l2_req,             0, sizeof(p_res->v4l2_req));
    memset(&p_res->fixed_fb_info,        0, sizeof(p_res->fixed_fb_info));
    memset(&p_res->v4l2_frame_buf,       0, sizeof(p_res->v4l2_frame_buf));
    memset(&p_res->configurable_fb_info, 0, sizeof(p_res->configurable_fb_info));

    p_res->fb_phys_base           = 0;
    p_res->fb_size_bytes          = 0;
    p_res->vid_mem_phys_base      = 0;
    p_res->vid_mem_size_bytes     = 0;
    p_res->rgb_565_buf_phys_base  = RBG565_BUF_BASE;
    p_res->rgb_565_buf_size_bytes = RGB565_BUF_SIZE_BYTES;

    // Open /dev/mem
	printf("INFO [v4l-to-fb0-dma::init_resources()] Opening /dev/mem\n");
	p_res->dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if(p_res->dev_mem_fd < 0){
		die_with_error("ERROR [v4l-to-fb0-dma::init_resources()] Error opening /dev/mem: ", strerror(errno), p_res);
	}

    // Open the video device
    printf("INFO [v4l-to-fb0-dma::init_resources()] Opening video device %s\n", video_device);
    p_res->v4l2_fd = open(video_device, O_RDWR);
    if(p_res->v4l2_fd < 0){
        die_with_error("ERROR [v4l-to-fb0-dma::init_resources()] Error opening video device: ", strerror(errno), p_res);
    }

    // Open the framebuffer device
    printf("INFO [v4l-to-fb0-dma::init_resources()] Opening framebuffer device %s\n", fb_device);
    p_res->fb_dev_fd = open(fb_device, O_RDWR);
    if(p_res->fb_dev_fd < 0){
        die_with_error("ERROR [v4l-to-fb0-dma::init_resources()] Error opening framebuffer device: ", strerror(errno), p_res);
    }

    // Setup the rgb565 buffer
    p_res->rgb_565_buf_phys = mmap(NULL, RGB565_BUF_SIZE_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, p_res->dev_mem_fd, RBG565_BUF_BASE);
    p_res->rgb_565_buf_pix = (uint16_t*)p_res->rgb_565_buf_phys;
    if (p_res->rgb_565_buf_phys == MAP_FAILED) {        
        die_with_error("ERROR [v4l-to-fb0-dma::init_resources()] Error mapping reserved memory: ", strerror(errno), p_res);
    }
}

/**
 * Close and unmap all files and memory associated with the resources struct.
 * @param p_res Pointer to the resources struct
 * @return None
 */
void cleanup_resources(Resources *p_res){
    if(p_res->vid_mem_phys != nullptr){
        munmap(p_res->vid_mem_phys, DMA_SELF_TEST_LEN);
    }
    if(p_res->fb_mem_phys != nullptr){
        munmap(p_res->fb_mem_phys, DMA_SELF_TEST_LEN);
    }
    if(p_res->rgb_565_buf_phys != nullptr){
        munmap(p_res->rgb_565_buf_phys, DMA_SELF_TEST_LEN);
    }
    if(p_res->v4l2_fd != -1){
        close(p_res->v4l2_fd);
    }
    if(p_res->fb_dev_fd != -1){
        close(p_res->fb_dev_fd);
    }
    if(p_res->dev_mem_fd != -1){
        close(p_res->dev_mem_fd);
    }
}

/**
 * Setup all things v4l2.
 * Open the video device
 * Set the video format
 * Request a buffer from the video device
 * Map the video buffer
 * @param p_res Pointer to the resources struct
 * @return None
 */
void setup_video(Resources *p_res){

    // Set video format
    p_res->v4l2_fmt.type                = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_fmt.fmt.pix.width       = INPUT_VIDEO_WIDTH;
    p_res->v4l2_fmt.fmt.pix.height      = INPUT_VIDEO_HEIGHT;
    p_res->v4l2_fmt.fmt.pix.pixelformat = INPUT_VIDEO_PIXEL_FMT;
    p_res->v4l2_fmt.fmt.pix.field       = V4L2_FIELD_NONE;

    printf("INFO [v4l-to-fb0-dma::setup_video()] Setting video format\n");
    if (ioctl(p_res->v4l2_fd, VIDIOC_S_FMT, &p_res->v4l2_fmt) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_video()] Error setting video format: ", strerror(errno), p_res);
    }

    // Request buffers from the video device
    printf("INFO [v4l-to-fb0-dma::setup_video()] Requesting buffer from video device\n");
    p_res->v4l2_req.count = 1; // Use 1 buffer
    p_res->v4l2_req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(p_res->v4l2_fd, VIDIOC_REQBUFS, &p_res->v4l2_req) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_video()] Error requesting v4l2 buffer: ", strerror(errno), p_res);
    }

    // Map the video buffer
    printf("INFO [v4l-to-fb0-dma::setup_video()] Mapping video buffer\n");
    p_res->v4l2_frame_buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_frame_buf.memory = V4L2_MEMORY_MMAP;
    p_res->v4l2_frame_buf.index = 0;

    if (ioctl(p_res->v4l2_fd, VIDIOC_QUERYBUF, &p_res->v4l2_frame_buf) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_video()] Error querying v4l2 buffer: ", strerror(errno), p_res);
    }

    // Get a physical pointer to the video buffer
    p_res->vid_mem_phys = mmap(NULL, p_res->v4l2_frame_buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, p_res->v4l2_fd, p_res->v4l2_frame_buf.m.offset);
    if (p_res->vid_mem_phys == MAP_FAILED) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_video()] Error mapping video buffer: ", strerror(errno), p_res);
    }

    // Start video streaming
    printf("INFO [v4l-to-fb0-dma::setup_video()] Starting video streaming from v4l2\n");
    if (ioctl(p_res->v4l2_fd, VIDIOC_STREAMON, &p_res->v4l2_frame_buf.type) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_video()] Error starting video streaming: ", strerror(errno), p_res);
    }

    p_res->vid_mem_phys_base = p_res->v4l2_frame_buf.m.offset;
    p_res->vid_mem_size_bytes = p_res->v4l2_frame_buf.length;
}

/**
 * Set up all things framebuffer related
 */
void setup_framebuffer(Resources *p_res){
    // Query framebuffer info
    if (ioctl(p_res->fb_dev_fd, FBIOGET_VSCREENINFO, &p_res->configurable_fb_info) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_framebuffer()] Error reading variable screen info: ", strerror(errno), p_res);
    }
    if (p_res->configurable_fb_info.bits_per_pixel != 16) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_framebuffer()] Framebuffer is not in RGB565 format\n", nullptr, p_res);
    }

    // Get fixed information
    if (ioctl(p_res->fb_dev_fd, FBIOGET_FSCREENINFO, &p_res->fixed_fb_info) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_framebuffer()] Error reading fixed screen info: ", strerror(errno), p_res);
    }
    printf("INFO [v4l-to-fb0-dma::setup_framebuffer()] Frame Buffer physical address: 0x%lx\n", p_res->fixed_fb_info.smem_start);
    printf("INFO [v4l-to-fb0-dma::setup_framebuffer()] Frame Buffer size:             0x%x\n", p_res->fixed_fb_info.smem_len);

    // Get variable screen information
    if (ioctl(p_res->fb_dev_fd, FBIOGET_VSCREENINFO, &p_res->configurable_fb_info) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_framebuffer()] Error reading variable screen info: ", strerror(errno), p_res);
    }
    
    // Map the framebuffer
    size_t fb_size = (p_res->configurable_fb_info.yres_virtual * p_res->configurable_fb_info.xres_virtual * p_res->configurable_fb_info.bits_per_pixel) / 8;
    printf("INFO [v4l-to-fb0-dma::setup_framebuffer()] Mapping framebuffer physical address\n");
    p_res->fb_mem_phys = mmap(NULL, fb_size, PROT_READ | PROT_WRITE, MAP_SHARED, p_res->dev_mem_fd, p_res->fixed_fb_info.smem_start);
    p_res->fb_mem_pix = (uint16_t*)p_res->fb_mem_phys;
    if (p_res->fb_mem_phys == MAP_FAILED) {
        die_with_error("ERROR [v4l-to-fb0-dma::setup_framebuffer()] Error mapping framebuffer: ", strerror(errno), p_res);
    }

    p_res->fb_phys_base = p_res->fixed_fb_info.smem_start;
    p_res->fb_size_bytes = fb_size;
}

int main(int argc, char *argv[]){

    printf("INFO [v4l-to-fb0-dma] Entering main\n");
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

    // Open /dev/mem, video device, framebuffer device, and setup the rgb565 buffer
    Resources resources;
    init_resources(&resources, video_device, fb_device);

    // Setup video reosources
    setup_video(&resources);

    // Setup framebuffer resources
    setup_framebuffer(&resources);

	//axi_dma1 configured as NOT scatter gather, MM2S --> FIFO --> S2MM
	AXIDMA dma1(DMA_1_AXI_LITE_BASE, resources.dev_mem_fd);
	printf("INFO [v4l-to-fb0-dma] Created DMA object with base address 0x%08X\n", DMA_1_AXI_LITE_BASE);
	if(dma1.initialize() != 0){
        die_with_error("ERROR [v4l-to-fb0-dma] Failed to initialize DMA object\n", nullptr, &resources);
	}
    if(dma1.self_test() < 0){
        die_with_error("ERROR [v4l-to-fb0-dma] DMA self test failed\n", nullptr, &resources);
    }
    // for(int i = 0; i < 2; i++){
    //     int result = dma1.self_test();
    //     if(result < 0){
    //         die_with_error("ERROR [v4l-to-fb0-dma] DMA self test failed\n", nullptr, &resources);
    //     }
    // }

    printf("INFO [v4l-to-fb0-dma] Starting continuous loop of reading frames...\n");
    
    uint32_t frame_loop_count = 0;
    unsigned long jiffies_per_sec = sysconf(_SC_CLK_TCK);
    unsigned long start_jiffies = get_jiffies();
    while(die_flag == false){
        
        // Capture a frame
        if (ioctl(resources.v4l2_fd, VIDIOC_QBUF, &resources.v4l2_frame_buf) == -1) {
            die_with_error("ERROR [v4l-to-fb0-dma] Error queueing buffer\n", nullptr, &resources);
        }
        if (ioctl(resources.v4l2_fd, VIDIOC_DQBUF, &resources.v4l2_frame_buf) == -1) {
            die_with_error("ERROR [v4l-to-fb0-dma] Error dequeueing buffer: ", strerror(errno), &resources);
        }

        // Convert the v4l2 input format to the framebuffer format
        yuyv_to_rgb565((uint8_t *)resources.vid_mem_phys, resources.rgb_565_buf_pix, INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT);

        // Create BD rings
        // This programs each ring to point to the next
        // dma1.create_s2mm_bd_ring(INPUT_VIDEO_HEIGHT);
        // dma1.create_mm2s_bd_ring(INPUT_VIDEO_HEIGHT);

        // //Program the BD rings
        // for(int i = 0; i < INPUT_VIDEO_HEIGHT; i++){
        //     uint32_t rgb565_tmp_ptr = resources.rgb_565_buf_phys_base + (INPUT_VIDEO_WIDTH * i * 2);
        //     dma1.set_mm2s_bd_buff_addr(i, rgb565_tmp_ptr);
        //     dma1.set_mm2s_bd_len(i, INPUT_VIDEO_WIDTH * 2);

        //     uint32_t fb_tmp_ptr = resources.fixed_fb_info.smem_start + (resources.configurable_fb_info.xres_virtual * i * 2);
        //     dma1.set_s2mm_bd_buff_addr(i, fb_tmp_ptr);
        //     dma1.set_s2mm_bd_len(i, INPUT_VIDEO_WIDTH * 2);
        // }

        // printf("INFO [v4l-to-fb0-dma] Starting DMA transfer in scatter gather mode. No polling for completion though\n");
        // dma1.transfer_sg();

        uint32_t rgb565_tmp_ptr = resources.rgb_565_buf_phys_base;
        uint8_t  bytes_per_pixel = 2;
        for (int row = 0; (row < INPUT_VIDEO_HEIGHT); row++) {
            // Copy one row at a time from the rgb565 buf to the frame buffer
            // Each pixel is 2 bytes
            // Non-DMA memcpy:
            // memcpy(&fb_pointer_pix[fb_info.xres_virtual * row], &rgb565_buf[INPUT_VIDEO_WIDTH * row], INPUT_VIDEO_WIDTH * sizeof(uint16_t));
			
            // 32 byte boundary: lower 5 bits are all 0
            uint32_t dst_pix_addr = resources.fixed_fb_info.smem_start + (resources.configurable_fb_info.xres_virtual * row * bytes_per_pixel); // * 2 for bytes per pixel
            rgb565_tmp_ptr += INPUT_VIDEO_WIDTH * bytes_per_pixel;
            int result = dma1.transfer(rgb565_tmp_ptr, dst_pix_addr, INPUT_VIDEO_WIDTH * bytes_per_pixel);            
        }
    }

    // Stop video streaming
    if (ioctl(resources.v4l2_fd, VIDIOC_STREAMOFF, &resources.v4l2_frame_buf.type) == -1) {
        die_with_error("ERROR [v4l-to-fb0-dma] Error stopping video streaming: ", strerror(errno), &resources);
    }

    // Cleanup
    cleanup_resources(&resources);

	return 0;
}