`timescale 1ns / 1ps

module tile_input_buffer (
    input  wire         aclk,
    input  wire         aresetn,

    // Native video signals from ADV7182A (8-bit bus, HS, VS, LLC)
    input  wire [7:0]   data_in,

    // AXI4-Stream output (8 bits)
    output wire [31:0]   m_axis_tdata,
    output wire         reset
    
);
    assign m_axis_tdata = {24'h000000, data_in}; // 8 bits in LSB
    assign reset = ~aresetn;

endmodule
