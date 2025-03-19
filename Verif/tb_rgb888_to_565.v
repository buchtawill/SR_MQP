`timescale 1ns / 1ps

module tb_RGB888_to_565;

    reg [127:0] rgb888_in;
    wire [63:0] rgb565_out;

    // Instantiate the DUT
    RGB888_to_565 uut (
        .rgb888_in(rgb888_in),
        .rgb565_out(rgb565_out)
    );

    initial begin
        // Apply test vectors
        rgb888_in = 128'h00_FF0000_00_00FF00_00_0000FF_00_FFFFFF; // Red, Green, Blue, White
        #10;
        $display("RGB888 Input: %h", rgb888_in);
        $display("RGB565 Output: %h", rgb565_out);

        rgb888_in = 128'h00_123456_00_789ABC_00_DEF012_00_345678; // Random test values
        #10;
        $display("RGB888 Input: %h", rgb888_in);
        $display("RGB565 Output: %h", rgb565_out);

        rgb888_in = 128'h00_000000_00_FF00FF_00_808080_00_C0C0C0; // Black, Magenta, Gray, Silver
        #10;
        $display("RGB888 Input: %h", rgb888_in);
        $display("RGB565 Output: %h", rgb565_out);

        $stop;
    end

endmodule
