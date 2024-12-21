#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/videodev2.h>
#include <sys/mman.h>
#include <errno.h>

// This is a test program to capture a single frame from a V4L2 device and save it to a file.

#define WIDTH 720
#define HEIGHT 576
#define PIXEL_FORMAT V4L2_PIX_FMT_YUYV // Use RGB333 pixel format

int main(int argc, char *argv[]) {


    if(argc < 2) {
        printf("Usage: %s <video_device>\n", argv[0]);
        printf("Example: %s /dev/video0\n", argv[0]);
        return 1;
    }
    const char *device = argv[1];
    int fd = open(device, O_RDWR);
    if (fd == -1) {
        perror("Opening video device");
        return 1;
    }

    // Query device capabilities
    struct v4l2_capability cap;
    if (ioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("Querying capabilities");
        close(fd);
        return 1;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "Device does not support video capture\n");
        close(fd);
        return 1;
    }

    // Set video format
    struct v4l2_format fmt;
    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = WIDTH;
    fmt.fmt.pix.height = HEIGHT;
    fmt.fmt.pix.pixelformat = PIXEL_FORMAT;
    fmt.fmt.pix.field = V4L2_FIELD_INTERLACED;
    if (ioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("Setting Pixel Format");
        close(fd);
        return 1;
    }

    // Request buffers
    struct v4l2_requestbuffers req;
    memset(&req, 0, sizeof(req));
    req.count = 1; // Request 1 buffer
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("Requesting Buffer");
        close(fd);
        return 1;
    }

    // Map the buffer
    struct v4l2_buffer buf;
    memset(&buf, 0, sizeof(buf));
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = 0;

    if (ioctl(fd, VIDIOC_QUERYBUF, &buf) == -1) {
        perror("Querying Buffer");
        close(fd);
        return 1;
    }

    void *buffer_start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd, buf.m.offset);
    if (buffer_start == MAP_FAILED) {
        perror("Memory Mapping");
        close(fd);
        return 1;
    }

    // Queue the buffer
    if (ioctl(fd, VIDIOC_QBUF, &buf) == -1) {
        perror("Queueing Buffer");
        munmap(buffer_start, buf.length);
        close(fd);
        return 1;
    }

    // Start streaming
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("Starting Capture");
        munmap(buffer_start, buf.length);
        close(fd);
        return 1;
    }

    // Dequeue the buffer to get the frame
    if (ioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
        perror("Dequeueing Buffer");
        munmap(buffer_start, buf.length);
        close(fd);
        return 1;
    }

    // Save the frame to a file
    FILE *file = fopen("frame.raw", "wb");
    if (!file) {
        perror("Opening file for writing");
        munmap(buffer_start, buf.length);
        close(fd);
        return 1;
    }
    fwrite(buffer_start, buf.bytesused, 1, file);
    fclose(file);
    printf("Frame captured and saved to frame.raw\n");

    // Stop streaming
    if (ioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
        perror("Stopping Capture");
    }

    // Cleanup
    munmap(buffer_start, buf.length);
    close(fd);

    return 0;
}
