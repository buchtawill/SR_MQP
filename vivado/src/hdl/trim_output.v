`timescale 1ns / 1ps
module trim_output (
    input  wire         aclk,
    input  wire         aresetn,

    // AXI-Stream slave interface (32-bit input)
    input  wire [31:0]  s_axis_tdata,
    input  wire         s_axis_tvalid,
    output wire         s_axis_tready,
    input  wire         s_axis_tlast,
    input  wire         s_axis_tuser,

    // AXI-Stream master interface (8-bit output)
    output wire [7:0]   m_axis_tdata,
    output wire         m_axis_tvalid,
    input  wire         m_axis_tready,
    output wire         m_axis_tlast,
    output wire         m_axis_tuser
);

assign s_axis_tready   = m_axis_tready;
assign m_axis_tvalid   = s_axis_tvalid;

assign m_axis_tlast    = s_axis_tlast;
assign m_axis_tuser    = s_axis_tuser;

// clip 32-bit data down to 8 bits
assign m_axis_tdata    = s_axis_tdata[7:0];

endmodule
