`timescale 1ns / 1ps

module RGB888_to_565 (
    input  wire        aclk,
    input  wire        aresetn,

    // AXI Stream Input (Slave)
    input  wire        s_tvalid,        // Incoming valid signal
    output wire        s_tready,        // This module is ready to receive data
    input  wire        s_tlast,         // Last transfer in packet
    input  wire [127:0] rgb888_in,      // 4 pixels of 32-bit 0BGR888

    // AXI Stream Output (Master)
    output wire        m_tvalid,        // Output valid signal
    input  wire        m_tready,        // Downstream module ready to receive data
    output wire        m_tlast,         // Last transfer in packet
    output wire [63:0] rgb565_out       // 4 pixels of 16-bit RGB565
);
    // Handshake: Ready when downstream is ready
    assign s_tready = m_tready;
    assign m_tvalid = s_tvalid;
    assign m_tlast  = s_tlast;
    
    // Conversion logic: Extract RGB888 and pack into RGB565
    assign rgb565_out[15:0]  = {rgb888_in[23:19], rgb888_in[15:10], rgb888_in[7:3]};       // Pixel 1
    assign rgb565_out[31:16] = {rgb888_in[55:51], rgb888_in[47:42], rgb888_in[39:35]};     // Pixel 2
    assign rgb565_out[47:32] = {rgb888_in[87:83], rgb888_in[79:74], rgb888_in[71:67]};     // Pixel 3
    assign rgb565_out[63:48] = {rgb888_in[119:115], rgb888_in[111:106], rgb888_in[103:99]}; // Pixel 4

endmodule