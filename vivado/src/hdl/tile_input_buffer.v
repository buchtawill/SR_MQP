`timescale 1ns / 1ps

module tile_input_buffer (
    input  wire         aclk,
    input  wire         aresetn,

    // Native video signals from ADV7182A (8-bit bus, HS, VS, LLC)
    input  wire         LLC,
    input  wire [7:0]   data_in,
    input  wire         HSYNC,
    input  wire         VSYNC,
    input  wire         FIELD,

    // AXI4-Stream output (8 bits)
    output wire [7:0]   m_axis_tdata,
    output wire         m_axis_tvalid,
    input  wire         m_axis_tready,
    output wire         m_axis_tlast,
    output wire         m_axis_tuser
    
);
    //----------------------------------------------------------------
    // 1) Video In to AXI4-Stream IP
    //----------------------------------------------------------------

    // The IP generates 32-bit tdata (only lower 8 bits meaningful)
    wire [31:0] s_axis_video_tdata;
    wire        s_axis_video_tvalid;
    wire        s_axis_video_tready;
    wire        s_axis_video_tlast;
    wire        s_axis_video_tuser;

    wire [31:0] vid_data_32 = {24'h000000, data_in}; // 8 bits in LSB
    wire reset = ~aresetn;

    v_vid_in_axi4s_0 u_v_vid_in_axi4s_0 (
        // Signals from eval board
        .vid_io_in_clk     (LLC),
        .vid_io_in_ce      (1'b1),
        .vid_io_in_reset   (reset),
        .vid_vsync         (VSYNC),
        .vid_hsync         (HSYNC),
        .vid_field_id      (FIELD),
        .vid_data          (vid_data_32),

        // AXI4-Stream Outputs (32 bits)
        .m_axis_video_tdata (s_axis_video_tdata),
        .m_axis_video_tvalid(s_axis_video_tvalid),
        .m_axis_video_tready(s_axis_video_tready),
        .m_axis_video_tlast (s_axis_video_tlast),
        .m_axis_video_tuser (s_axis_video_tuser),

        // AXI4-Stream clock domain
        .aclk              (aclk),
        .aresetn           (aresetn),
        .aclken            (1'b1),
        
        .axis_enable       (1'b1),
        .vid_active_video  (), 
        .vid_vblank        (),
        .vid_hblank        () 
    );


    //----------------------------------------------------------------
    // 2) Clipper: 32-bit to 8-bit AXI4-Stream
    //----------------------------------------------------------------

    trim_output u_trim_output (
        .aclk        (aclk),
        .aresetn     (aresetn),

        // 32-bit input from Video In IP
        .s_axis_tdata (s_axis_video_tdata),
        .s_axis_tvalid(s_axis_video_tvalid),
        .s_axis_tready(s_axis_video_tready),
        .s_axis_tlast (s_axis_video_tlast),
        .s_axis_tuser (s_axis_video_tuser),

        // 8-bit output to downstream modules
        .m_axis_tdata (m_axis_tdata),
        .m_axis_tvalid(m_axis_tvalid),
        .m_axis_tready(m_axis_tready),
        .m_axis_tlast (m_axis_tlast),
        .m_axis_tuser (m_axis_tuser)
    );

endmodule
