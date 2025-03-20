`timescale 1ns / 1ps

module tb_RGB888_to_rgb565;

    // Clock and reset
    reg aclk;
    reg aresetn;

    // AXI Stream input (Slave)
    reg s_tvalid;
    wire s_tready;
    reg s_tlast;
    reg [127:0] rgb888_in;

    // AXI Stream output (Master)
    wire m_tvalid;
    reg m_tready;
    wire m_tlast;
    wire [63:0] rgb565_out;

    // Instantiate the DUT (Device Under Test)
    RGB888_to_565 uut (
        .aclk(aclk),
        .aresetn(aresetn),
        .s_tvalid(s_tvalid),
        .s_tready(s_tready),
        .s_tlast(s_tlast),
        .rgb888_in(rgb888_in),
        .m_tvalid(m_tvalid),
        .m_tready(m_tready),
        .m_tlast(m_tlast),
        .rgb565_out(rgb565_out)
    );

    // Clock generation
    always #5 aclk = ~aclk; // 10 ns period (100 MHz)

    // Test sequence
    initial begin
        // Initialize signals
        aclk = 0;
        aresetn = 0;
        s_tvalid = 0;
        s_tlast = 0;
        rgb888_in = 0;
        m_tready = 1; // Assume downstream is always ready

        // Reset pulse
        #20 aresetn = 1;

        // Test case 1: Simple RGB888 to RGB565 conversion
        #10;
        s_tvalid = 1;
        s_tlast = 0;
        rgb888_in = 128'h00_FF0000_00_00FF00_00_0000FF_00_FFFFFF; // Red, Green, Blue, White
        #10;
        s_tvalid = 0; // Hold off input to check handshake
        #10;

        // Test case 2: Another conversion
        #10;
        s_tvalid = 1;
        rgb888_in = 128'h00_123456_00_789ABC_00_DEF012_00_345678; // Random test values
        #10;
        s_tvalid = 0;
        #10;

        // Test case 3: Last signal active (end of frame)
        #10;
        s_tvalid = 1;
        s_tlast = 1; // Last transfer in frame
        rgb888_in = 128'h00_000000_00_FF00FF_00_808080_00_C0C0C0; // Black, Magenta, Gray, Silver
        #10;
        s_tvalid = 0;
        s_tlast = 0;
        #10;

        // End simulation
        $stop;
    end

endmodule