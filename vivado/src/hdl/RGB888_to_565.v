`timescale 1ns / 1ps

module RGB888_to_565 (
    input  wire        aclk,
    input  wire        aresetn,

    // AXI Stream Input (Slave)
    input  wire         s_tvalid,        // Incoming valid signal
    output wire         s_tready,        // This module is ready to receive data
    input  wire         s_tlast,         // Last transfer in packet
    input  wire [15:0]  s_tkeep,         // How many to keep cuh
    input  wire [15:0]  s_tstrb,         // I don't fucking know
    input  wire [127:0] s_tdata,       // 4 pixels of 32-bit 0BGR888

    // AXI Stream Output (Master)
    output wire        m_tvalid,        // Output valid signal
    input  wire        m_tready,        // Downstream module ready to receive data
    output wire        m_tlast,         // Last transfer in packet
    output wire [7:0]  m_tkeep,         // 
    output wire [7:0]  m_tstrb,         // 
    output wire [63:0] m_tdata       // 4 pixels of 16-bit RGB565
);
    // Handshake: Ready when downstream is ready
    assign s_tready = m_tready;
    assign m_tvalid = s_tvalid;
    assign m_tlast  = s_tlast;
    
    // Conversion logic: Extract RGB888 and pack into RGB565
    assign m_tdata[15:0]  = {s_tdata[23:19], s_tdata[15:10], s_tdata[7:3]};       // Pixel 1
    assign m_tdata[31:16] = {s_tdata[55:51], s_tdata[47:42], s_tdata[39:35]};     // Pixel 2
    assign m_tdata[47:32] = {s_tdata[87:83], s_tdata[79:74], s_tdata[71:67]};     // Pixel 3
    assign m_tdata[63:48] = {s_tdata[119:115], s_tdata[111:106], s_tdata[103:99]}; // Pixel 4
    
    assign m_tkeep = 8'hff;
    assign m_tstrb = 8'hff;
    

endmodule