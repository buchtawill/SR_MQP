module FeatureExtraction #(
    parameter FILTER_SIZE = 5,
    parameter NUM_FILTERS = 48, //other value tested by paper was 56
    parameter NUM_CHANNELS = 1
) (
    clk_in,
    rst_in,
    map_in,
    start,
    map_out,
    save,
    ready
);

    input clk_in;
    input rst_in;
    input signed [127:0] map_in; //map size will need to change
    input start;

    output reg signed [127:0] map_out; //size will change, needs to match map_in
    output reg save = 0;
    output reg ready = 0;

    parameter kernal_size = 3'd5; //dimensions of kernal
    parameter num_mult = 5'd25; //number of multiplications with kernal required to produce output feature map


    //assign kernal values. Eventually pull from board mem, so doesn't require resynthesis
    assign k1[0];
    assign k1[1];
    assign k1[2];
    assign k1[3];
    assign k1[4];
    assign k1[5];
    assign k1[6];
    assign k1[7];
    assign k1[8];
    assign k1[9];
    assign k1[10];
    assign k1[11];
    assign k1[12];
    assign k1[13];
    assign k1[14];
    assign k1[15];
    assign k1[16];
    assign k1[17];
    assign k1[18];
    assign k1[19];
    assign k1[20];
    assign k1[21];
    assign k1[22];
    assign k1[23];
    assign k1[24];
    assign k1[25];

endmodule