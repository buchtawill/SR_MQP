/**
 * This file contains various definitions for the v4l-to-fb0-dma application
 */


#ifndef INTERPOLATE2X_H_
#define INTERPOLATE2X_H_

#include <linux/videodev2.h>    // V4L2 API
#include <linux/fb.h>           // Framebuffer API
#include <stdint.h> 
#include "PhysMem.h"

// Addresses found in SR_MQP/petalinux/kv260_plnx_proj/components/plnx_workspace/device-tree/device-tree/pl.dtsi
// Or in the vivado block diagram address editor
#define DMA_0_AXI_LITE_BASE		0xA0020000

#define INPUT_VIDEO_WIDTH       720
#define INPUT_VIDEO_HEIGHT      576
#define INPUT_VIDEO_PIXEL_FMT   V4L2_PIX_FMT_YUYV

#define RGB565_BUF_SIZE_BYTES   (INPUT_VIDEO_WIDTH * INPUT_VIDEO_HEIGHT * 2) // 2 bytes per pixel

#define UPSCALE_FACTOR          2

/**
 * This struct contains all resources that the main program uses.
 */
typedef struct {

    // File descriptors
    int  dev_mem_fd;
    int  v4l2_fd;
    int  fb_dev_fd;

    // Video resources
    PhysMem *vid_mem_block;
    uint32_t vid_mem_phys_base;
    uint32_t vid_mem_size_bytes;
    struct v4l2_format v4l2_fmt;
    struct v4l2_requestbuffers v4l2_req;
    struct v4l2_buffer v4l2_frame_buf;

    // Framebuffer resources
    PhysMem *fb_mem_block;
    uint16_t *fb_mem_pix;
    uint32_t fb_phys_base;  // Easier to keep track of the physical address
    uint32_t fb_size_bytes; // Easier to keep track of the size in bytes
    struct fb_var_screeninfo configurable_fb_info;
    struct fb_fix_screeninfo fixed_fb_info;

    // RGB565 buffer resources
    PhysMem *rgb_565_block;

} Resources;

void cleanup_resources(Resources *p_res);

#endif // INTERPOLATE2X_H_
