/**
 * This file contains various definitions for the v4l-to-fb0-dma application
 */


#ifndef V4L_TO_FB0_DMA_H_
#define V4L_TO_FB0_DMA_H_

#include <linux/videodev2.h>    // V4L2 API
#include <linux/fb.h>           // Framebuffer API
#include <stdint.h> 

// Addresses found in SR_MQP/petalinux/kv260_plnx_proj/components/plnx_workspace/device-tree/device-tree/pl.dtsi
#define DMA_0_AXI_LITE_BASE		0xA0010000
#define DMA_1_AXI_LITE_BASE		0xA0020000

#define INPUT_VIDEO_WIDTH       720
#define INPUT_VIDEO_HEIGHT      576
#define INPUT_VIDEO_BYTES_PP    2
#define INPUT_VIDEO_PIXEL_FMT   V4L2_PIX_FMT_YUYV

// TODO: Port to use PhysMman
#define RBG565_BUF_BASE         0x78000000
#define RGB565_BUF_SIZE_BYTES   (INPUT_VIDEO_WIDTH * INPUT_VIDEO_HEIGHT * INPUT_VIDEO_BYTES_PP)

/**
 * This struct contains all resources that the main program uses.
 */
typedef struct {
    int  dev_mem_fd;
    int  v4l2_fd;
    int  fb_dev_fd;
    void *vid_mem_phys;
    void *fb_mem_phys;
    uint16_t *fb_mem_pix;
    void *rgb_565_buf_phys;
    uint16_t *rgb_565_buf_pix;

    struct fb_var_screeninfo configurable_fb_info;
    struct fb_fix_screeninfo fixed_fb_info;

    struct v4l2_format v4l2_fmt;
    struct v4l2_requestbuffers v4l2_req;
    struct v4l2_buffer v4l2_frame_buf;

    uint32_t fb_phys_base;
    uint32_t fb_size_bytes;
    uint32_t vid_mem_phys_base;
    uint32_t vid_mem_size_bytes;
    uint32_t rgb_565_buf_phys_base;
    uint32_t rgb_565_buf_size_bytes;

} Resources;


void cleanup_resources(Resources *p_res);


#endif // V4L_TO_FB0_DMA_H_
