/*

This program will read frames from a video device, convert them from YUYV 4:2:2 to RGB888, interpolate them with our HW, then 
convert to RGB565 before displaying them on the framebuffer device.

This thing is going to be so ungodly slow, but it is just a prototype. The main flow is this:
 1. Capture a frame from v4l2
 2. Convert the frame to RGB888
 - Loop:
   3. Send a tile down to the PL, one line at a time
   4. Receive the tile and store it in a temporary buffer (can be in userspace)
   5. Convert the tile to RGB565
   6. Copy the tile to the frame buffer, one line at a time

Steps 2, 4, 5, and 6 will all be deleted eventually

Author: Will Buchta
Date Modified: 12/26/2024

*/

#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <unistd.h>

#include <fcntl.h> // for open flags

#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <linux/fb.h>

#include <signal.h> //for sigint
#include <time.h>   // for jiffies

#include "interpolate2x.h"
#include "phys-mman.h"
#include "PhysMem.h"
#include "axi-dma.h"

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

// Convert YUYV to RGB888
void yuyv_to_rgb888(uint8_t *yuyv, uint8_t *rgb888, int width, int height) {
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

        rgb888[i * 3] = r0;
        rgb888[i * 3 + 1] = g0;
        rgb888[i * 3 + 2] = b0;
        rgb888[(i + 1) * 3] = r1;
        rgb888[(i + 1) * 3 + 1] = g1;
        rgb888[(i + 1) * 3 + 2] = b1;
    }
}

/**
 * Convert num_pixels pixels from RGB888 to RGB565, moving from rgb888 buffer to rgb565 buffer
 * @param rgb888 Pointer to the RGB888 pixels
 * @param rgb565 Pointer to the RGB565 pixels
 * @param num_pixels The number of pixels to move
*/
void rgb888_to_rgb565(uint8_t *rgb888, uint16_t *rgb565, uint32_t num_pixels) {

    // i is pixel index
    for (uint32_t i = 0; i < num_pixels; i++) {
        uint8_t r = rgb888[i * 3];
        uint8_t g = rgb888[i * 3 + 1];
        uint8_t b = rgb888[i * 3 + 2];

        rgb565[i] = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
    }
}

/**
 * Print a given error message and clean up resources before exiting
 * @param error_msg The error message to print
 * @param err_str The error string to print (optional, given by strerror(errno), set to nullptr if not using)
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

    if(sig == SIGINT){
        printf("INFO [interpolate2x] Received signal, cleaning up...\n");
        die_flag = true;
    }
}

/**
 * Open dev/mem
 * Open the video device
 * Open the framebuffer device
 * Set up buffers for the RGB888 frame and interpolated RGB888 frame
 * @param p_res Pointer to the resources struct
 * @param video_device The video device
 * @param fb_device The framebuffer device
 * @return None
 */
void init_resources(Resources *p_res, const char* video_device, const char* fb_device){
    p_res->v4l2_fd         = -1;
    p_res->fb_dev_fd       = -1;
    p_res->dev_mem_fd      = -1;
    p_res->vid_mem_block   = nullptr;
    p_res->fb_mem_block    = nullptr;
    p_res->input888_block  = nullptr;
    p_res->interp888_block = nullptr;

    memset(&p_res->v4l2_fmt,             0, sizeof(p_res->v4l2_fmt));
    memset(&p_res->v4l2_req,             0, sizeof(p_res->v4l2_req));
    memset(&p_res->fixed_fb_info,        0, sizeof(p_res->fixed_fb_info));
    memset(&p_res->v4l2_frame_buf,       0, sizeof(p_res->v4l2_frame_buf));
    memset(&p_res->configurable_fb_info, 0, sizeof(p_res->configurable_fb_info));

    p_res->fb_phys_base           = 0;
    p_res->fb_size_bytes          = 0;
    p_res->vid_mem_phys_base      = 0;
    p_res->vid_mem_size_bytes     = 0;

    // Open /dev/mem
	printf("INFO [interpolate2x::init_resources()] Opening /dev/mem\n");
	p_res->dev_mem_fd = open("/dev/mem", O_RDWR | O_SYNC);
	if(p_res->dev_mem_fd < 0){
		die_with_error("ERROR [interpolate2x::init_resources()] Error opening /dev/mem: ", strerror(errno), p_res);
	}

    // Initialize the Physical Memory Manager
    printf("INFO [interpolate2x::init_resources()] Initializing Physical Memory Manager\n");
    int pmm_result = PMM.init(p_res->dev_mem_fd);
    if(pmm_result != 0){
        die_with_error("ERROR [interpolate2x::init_resources()] Failed to initialize Physical Memory Manager\n", nullptr, p_res);
    }

    // Open the video device
    printf("INFO [interpolate2x::init_resources()] Opening video device %s\n", video_device);
    p_res->v4l2_fd = open(video_device, O_RDWR);
    if(p_res->v4l2_fd < 0){
        die_with_error("ERROR [interpolate2x::init_resources()] Error opening video device: ", strerror(errno), p_res);
    }

    // Open the framebuffer device
    printf("INFO [interpolate2x::init_resources()] Opening framebuffer device %s\n", fb_device);
    p_res->fb_dev_fd = open(fb_device, O_RDWR);
    if(p_res->fb_dev_fd < 0){
        die_with_error("ERROR [interpolate2x::init_resources()] Error opening framebuffer device: ", strerror(errno), p_res);
    }

    // Create buffers for the RGB888 frame and interpolated RGB888 frame
    printf("INFO [interpolate2x::init_resources()] Allocating input888 block\n");
    p_res->input888_block = PMM.alloc(INPUT_VIDEO_HEIGHT * INPUT_VIDEO_WIDTH * 3);
    if(p_res->input888_block == nullptr){
        die_with_error("ERROR [interpolate2x::init_resources()] Failed to allocate input888 block\n", nullptr, p_res);
    }

    printf("INFO [interpolate2x::init_resources()] Allocating interp888 block\n");
    p_res->interp888_block = PMM.alloc(INPUT_VIDEO_HEIGHT * UPSCALE_FACTOR * \
                                        INPUT_VIDEO_WIDTH * UPSCALE_FACTOR * 3);
    if(p_res->interp888_block == nullptr){
        die_with_error("ERROR [interpolate2x::init_resources()] Failed to allocate interp888 block\n", nullptr, p_res);
    }
}

/**
 * Close and unmap all files and memory associated with the resources struct.
 * @param p_res Pointer to the resources struct
 * @return None
 */
void cleanup_resources(Resources *p_res){

    if(p_res->v4l2_fd != -1){
        close(p_res->v4l2_fd);
    }
    if(p_res->fb_dev_fd != -1){
        close(p_res->fb_dev_fd);
    }
    if(p_res->dev_mem_fd != -1){
        close(p_res->dev_mem_fd);
    }

    PMM.free(p_res->vid_mem_block);
    PMM.free(p_res->fb_mem_block);
    PMM.free(p_res->input888_block);
    PMM.free(p_res->interp888_block);
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

    printf("INFO [interpolate2x::setup_video()] Setting video format\n");
    if (ioctl(p_res->v4l2_fd, VIDIOC_S_FMT, &p_res->v4l2_fmt) == -1) {
        die_with_error("ERROR [interpolate2x::setup_video()] Error setting video format: ", strerror(errno), p_res);
    }

    // Request buffers from the video device
    printf("INFO [interpolate2x::setup_video()] Requesting buffer from video device\n");
    p_res->v4l2_req.count = 1; // Use 1 buffer
    p_res->v4l2_req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_req.memory = V4L2_MEMORY_MMAP;

    if (ioctl(p_res->v4l2_fd, VIDIOC_REQBUFS, &p_res->v4l2_req) == -1) {
        die_with_error("ERROR [interpolate2x::setup_video()] Error requesting v4l2 buffer: ", strerror(errno), p_res);
    }

    // Map the video buffer
    printf("INFO [interpolate2x::setup_video()] Mapping video buffer\n");
    p_res->v4l2_frame_buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    p_res->v4l2_frame_buf.memory = V4L2_MEMORY_MMAP;
    p_res->v4l2_frame_buf.index  = 0;

    if (ioctl(p_res->v4l2_fd, VIDIOC_QUERYBUF, &p_res->v4l2_frame_buf) == -1) {
        die_with_error("ERROR [interpolate2x::setup_video()] Error querying v4l2 buffer: ", strerror(errno), p_res);
    }

    p_res->vid_mem_phys_base  = p_res->v4l2_frame_buf.m.offset;
    p_res->vid_mem_size_bytes = p_res->v4l2_frame_buf.length;

    // Map the video buffer to a PhysMem object
    p_res->vid_mem_block = PMM.alloc((uint32_t)p_res->vid_mem_phys_base, p_res->vid_mem_size_bytes);
    if(p_res->vid_mem_block == nullptr){
        die_with_error("ERROR [interpolate2x::setup_video()] Failed to map video buffer to PhysMem object\n", nullptr, p_res);
    }

    // Start video streaming
    printf("INFO [interpolate2x::setup_video()] Starting video streaming from v4l2\n");
    if (ioctl(p_res->v4l2_fd, VIDIOC_STREAMON, &p_res->v4l2_frame_buf.type) == -1) {
        die_with_error("ERROR [interpolate2x::setup_video()] Error starting video streaming: ", strerror(errno), p_res);
    }
}

/**
 * Set up all things framebuffer related
 */
void setup_framebuffer(Resources *p_res){
    // Query framebuffer info
    if (ioctl(p_res->fb_dev_fd, FBIOGET_VSCREENINFO, &p_res->configurable_fb_info) == -1) {
        die_with_error("ERROR [interpolate2x::setup_framebuffer()] Error reading variable screen info: ", strerror(errno), p_res);
    }
    if (p_res->configurable_fb_info.bits_per_pixel != 16) {
        die_with_error("ERROR [interpolate2x::setup_framebuffer()] Framebuffer is not in RGB565 format\n", nullptr, p_res);
    }

    // Get fixed information
    if (ioctl(p_res->fb_dev_fd, FBIOGET_FSCREENINFO, &p_res->fixed_fb_info) == -1) {
        die_with_error("ERROR [interpolate2x::setup_framebuffer()] Error reading fixed screen info: ", strerror(errno), p_res);
    }
    printf("INFO [interpolate2x::setup_framebuffer()] Frame Buffer physical address: 0x%lx\n", p_res->fixed_fb_info.smem_start);
    printf("INFO [interpolate2x::setup_framebuffer()] Frame Buffer size:             0x%x\n", p_res->fixed_fb_info.smem_len);

    // Get variable screen information
    if (ioctl(p_res->fb_dev_fd, FBIOGET_VSCREENINFO, &p_res->configurable_fb_info) == -1) {
        die_with_error("ERROR [interpolate2x::setup_framebuffer()] Error reading variable screen info: ", strerror(errno), p_res);
    }
    
    // Map the framebuffer to a PhysMem object
    size_t fb_size = (p_res->configurable_fb_info.yres_virtual * p_res->configurable_fb_info.xres_virtual * p_res->configurable_fb_info.bits_per_pixel) / 8;
    printf("INFO [interpolate2x::setup_framebuffer()] Mapping framebuffer to PhysMem\n");
    
    p_res->fb_phys_base = p_res->fixed_fb_info.smem_start;
    p_res->fb_size_bytes = fb_size;
    p_res->fb_mem_block = PMM.alloc(p_res->fb_phys_base, p_res->fb_size_bytes);

    if(p_res->fb_mem_block == nullptr){
        die_with_error("ERROR [interpolate2x::setup_framebuffer()] Failed to map framebuffer to PhysMem\n", nullptr, p_res);
    }
}

/**
 * Compute the source address of every row in a TileInfo struct given the coordinates of the tile and PhysMem buffer.
 * If the given coordinates end up with the tile being out of bounds, the tile will be shifted to the left or up,
 * overlapping some with the previous tile.
 * @param src_block Pointer to the PhysMem object containing the source image
 * @param tile Pointer to the TileInfo struct
 * @param xres The width of the source image
 * @param yres The height of the source image
 * @param bytes_pp The number of bytes per pixel in the source image
 * @return None
 */
void compute_tile_src_addr(PhysMem *src_block, TileInfo *tile, uint16_t xres, uint16_t yres, uint8_t bytes_pp){

    // Compute pixel coordinates of top left corner of tile
    uint32_t pixel_x = tile->tile_x * TILE_WIDTH_PIX;
    uint32_t pixel_y = tile->tile_y * TILE_HEIGHT_PIX;

    // Handle tiles not being fully in the image
    // 720: If pixel_x > 692, pixel_x = 692
    // 576: If pixel_y > 548, pixel_y = 548
    if(pixel_x > (xres - TILE_WIDTH_PIX)) pixel_x = xres - TILE_WIDTH_PIX;
    if(pixel_y > (yres - TILE_HEIGHT_PIX)) pixel_y = yres - TILE_HEIGHT_PIX;

    tile->src_pixel_x = pixel_x;
    tile->src_pixel_y = pixel_y;

    // Memory offset of the top left corner of the tile
    //                                Linear pixel number
    //                            vvvvvvvvvvvvvvvvvvvvvvvvvv
    uint32_t tile_start_offset = ((pixel_y * yres) + pixel_x) * bytes_pp;
    tile_start_offset += src_block->get_phys_address();

    for(uint16_t row = 0; row < TILE_HEIGHT_PIX; row++){
        uint32_t row_offset = row * xres * bytes_pp;
        tile->src_row_offset[row] = tile_start_offset + row_offset;
    }
}

/**
 * Compute the destination address of every row in a TileInfo struct given a populated tile struct and PhysMem buffer.
 * The user MUST call compute_tile_src_addr before calling this function.
 * @param tile Pointer to the TileInfo struct - Must have source offsets populated by first calling compute_tile_src_offsets
 * @param xres_in The width of the source image
 * @param yres_in The height of the source image
 * @param upscale_factor The factor by which the image is being upscaled
 * @param bytes_pp The number of bytes per pixel in the source image
 * @return None
 */
void compute_tile_dst_addr(PhysMem* dst_block, TileInfo *tile, uint16_t xres_in, uint16_t yres_in, uint32_t upscale_factor, uint8_t bytes_pp){

    uint32_t dst_pixel_x = tile->src_pixel_x * upscale_factor;
    uint32_t dst_pixel_y = tile->src_pixel_y * upscale_factor;

    uint32_t xres_out = xres_in * upscale_factor;
    uint32_t yres_out = yres_in * upscale_factor;

    // Memory offset of the top left corner of the tile
    uint32_t tile_start_offset = ((dst_pixel_y * yres_out) + dst_pixel_x) * bytes_pp;
    tile_start_offset += dst_block->get_phys_address();

    for(uint16_t row = 0; row < (TILE_HEIGHT_PIX * upscale_factor); row++){
        uint32_t row_offset = row * xres_out * bytes_pp;
        tile->dst_row_offset[row] = tile_start_offset + row_offset;
    }
}

int main(int argc, char *argv[]){

    printf("INFO [interpolate2x] Entering main\n");
    const char* video_device = nullptr;
    const char* fb_device = nullptr;
	if(argc < 3) {
        // printf("Usage: %s <video_device> <fb_device>\n", argv[0]);
        // printf("Example: sudo %s /dev/video0 /dev/fb0\n", argv[0]);
        video_device = "/dev/video0";
        fb_device = "/dev/fb0";
    }
    else{
        video_device = argv[1];
        fb_device = argv[2];
    }

    // Register CTRL-C handler
    signal(SIGINT, sigint_handler);

    // Open /dev/mem, video device, framebuffer device
    // Initialize Physical Memory Manager
    // Initialize input888 buffer, interpolated888 buffer
    Resources resources;
    init_resources(&resources, video_device, fb_device);

    // Setup video reosources
    setup_video(&resources);

    // Setup framebuffer resources
    setup_framebuffer(&resources);

	//axi_dma1 configured as NOT scatter gather, MM2S --> FIFO --> S2MM
	AXIDMA dma1(DMA_0_AXI_LITE_BASE, resources.dev_mem_fd);
	printf("INFO [interpolate2x] Created DMA object with base address 0x%08X\n", DMA_0_AXI_LITE_BASE);
	if(dma1.initialize() != 0){
        die_with_error("ERROR [interpolate2x] Failed to initialize DMA object\n", nullptr, &resources);
	}

    // Run the self test twice in a row
    if(dma1.self_test() < 0 || dma1.self_test() < 0){
        die_with_error("ERROR [interpolate2x] DMA self test failed\n", nullptr, &resources);
    }

    printf("INFO [interpolate2x] Starting video output\n");
    
    while(!die_flag){
        
        // Capture a frame
        if (ioctl(resources.v4l2_fd, VIDIOC_QBUF, &resources.v4l2_frame_buf) == -1) {
            die_with_error("ERROR [interpolate2x] Error queueing buffer\n", nullptr, &resources);
        }
        if (ioctl(resources.v4l2_fd, VIDIOC_DQBUF, &resources.v4l2_frame_buf) == -1) {
            die_with_error("ERROR [interpolate2x] Error dequeueing buffer: ", strerror(errno), &resources);
        }

        // Convert v4l2 frame to RGB888
        // Loop
        //  - Send a tile down to the PL for interpolation
        //  - Receive the tile in the interp_888 buffer / PMM block
        // Move to framebuffer, converting to RGB565 on the way

        yuyv_to_rgb888((uint8_t*)(resources.vid_mem_block->get_mem_ptr()), 
                       (uint8_t*)(resources.input888_block->get_mem_ptr()), 
                       INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT);

        TileInfo tile;
        uint16_t num_vert_tiles = INPUT_VIDEO_HEIGHT / TILE_HEIGHT_PIX;
        uint16_t num_horz_tiles = INPUT_VIDEO_WIDTH  / TILE_WIDTH_PIX;
        
        // Handle partial tiles
        if((INPUT_VIDEO_HEIGHT % TILE_HEIGHT_PIX) != 0) num_vert_tiles++;
        if((INPUT_VIDEO_WIDTH  % TILE_WIDTH_PIX)  != 0) num_horz_tiles++;

        for(uint16_t tx = 0; tx < num_horz_tiles; tx++){
            for(uint16_t ty = 0; ty < num_vert_tiles; ty++){

                // Calculate physical addresses for each row of the tile
                // Send the tile down to the PL for interpolation
                // Calculate destination addresses for each row of the tile
                // Receive the tile in the interp_888 buffer / PMM block

                tile.tile_x = tx;
                tile.tile_y = ty;
                compute_tile_src_addr(resources.input888_block, &tile, INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT, 3);

                // Send pixels from input RGB888 buffer to the PL for interpolation
                for(uint16_t row = 0; row < TILE_HEIGHT_PIX; row++){
                    int result = dma1.transfer_mm2s(tile.src_row_offset[row], TILE_WIDTH_PIX * 3, true);
                    if(result < 0) die_with_error("ERROR [interpolate2x] Error transferring data to PL ", nullptr, &resources);
                }

                // Calculate destination addresses for each row of the tile
                // In the future, this will happen concurrently with the PL interpolation
                compute_tile_dst_addr(resources.interp888_block, &tile, INPUT_VIDEO_WIDTH, INPUT_VIDEO_HEIGHT, UPSCALE_FACTOR, 3);

                // Receive pixels from the PL and store them in the interp888 buffer
                for(uint16_t row = 0; row < (TILE_HEIGHT_PIX * UPSCALE_FACTOR); row++){
                    int result = dma1.transfer_s2mm(tile.dst_row_offset[row], TILE_WIDTH_PIX * UPSCALE_FACTOR * 3, true);
                    if(result < 0) die_with_error("ERROR [interpolate2x] Error transferring data from PL ", nullptr, &resources);   
                }
            }
        }

        // Move the interpolated frame to the framebuffer, converting to RGB565 on the way
        // Transfer 1 row at a time to account for resolution differences
        for(uint16_t row = 0; row < (INPUT_VIDEO_HEIGHT * UPSCALE_FACTOR); row++){
            uint32_t rgb888_offset = row * INPUT_VIDEO_WIDTH * UPSCALE_FACTOR * 3; 
            uint8_t *rgb888_row = (uint8_t*)(resources.interp888_block->get_mem_ptr()) + rgb888_offset;

            // Do not multiply by bytes per pixel because each index in fb_row_ptr IS a pixel
            uint32_t rgb565_offset = row * resources.configurable_fb_info.xres_virtual;
            uint16_t *fb_row_ptr = (uint16_t*)(resources.fb_mem_block->get_mem_ptr()) + rgb565_offset;

            rgb888_to_rgb565(rgb888_row, fb_row_ptr, INPUT_VIDEO_WIDTH * UPSCALE_FACTOR);
        }
    }

    // Stop video streaming
    if (ioctl(resources.v4l2_fd, VIDIOC_STREAMOFF, &resources.v4l2_frame_buf.type) == -1) {
        die_with_error("ERROR [interpolate2x] Error stopping video streaming: ", strerror(errno), &resources);
    }

    // Cleanup
    cleanup_resources(&resources);

	return 0;
}
