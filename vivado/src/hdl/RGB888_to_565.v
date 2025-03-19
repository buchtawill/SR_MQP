`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 03/19/2025 03:14:38 PM
// Design Name: 
// Module Name: RGB888_to_565
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module RGB888_to_565 (
    input  wire [127:0] rgb888_in,  // 4 pixels of 32-bit 0BGR888
    output wire [63:0]  rgb565_out  // 4 pixels of 16-bit RGB565
);
    // Extract individual pixels (32-bit 0BGR888 format)
    wire [7:0] B0 = rgb888_in[7:0];
    wire [7:0] G0 = rgb888_in[15:8];
    wire [7:0] R0 = rgb888_in[23:16];

    wire [7:0] B1 = rgb888_in[39:32];
    wire [7:0] G1 = rgb888_in[47:40];
    wire [7:0] R1 = rgb888_in[55:48];

    wire [7:0] B2 = rgb888_in[71:64];
    wire [7:0] G2 = rgb888_in[79:72];
    wire [7:0] R2 = rgb888_in[87:80];

    wire [7:0] B3 = rgb888_in[103:96];
    wire [7:0] G3 = rgb888_in[111:104];
    wire [7:0] R3 = rgb888_in[119:112];

    // Convert RGB888 to RGB565 (packing 4 pixels into 64 bits)
    assign rgb565_out = { 
        R3[7:3], G3[7:2], B3[7:3],  // Pixel 3
        R2[7:3], G2[7:2], B2[7:3],  // Pixel 2
        R1[7:3], G1[7:2], B1[7:3],  // Pixel 1
        R0[7:3], G0[7:2], B0[7:3]   // Pixel 0
    };

endmodule