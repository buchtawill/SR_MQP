module topModule(
    input clk_in,
    input rst_in,
    input [16383:0] image_frame //I think eventually this will need to be input from somewhere else, place holder
);

    parameter INPUT_IMAGE_SIZE;
    parameter FE_MAP_SIZE;
    parameter SHRINK_MAP_SIZE;
    parameter MAPPING_MAP_SIZE;
    parameter EXPAND_MAP_SIZE;
    parameter DECONV_MAP_SIZE;

    wire FEDone;
    wire ShrinkingDone;
    wire MappingDone;
    wire ExpandDone;
    wire DeconvDone;

    wire PReLU1Done;
    wire PReLU2Done;
    wire PReLU3Done;
    wire PReLU4Done;

    //depending on how we want flow of system might change to wires, but probably not since blocks have dif latency
    reg [FE_MAP_SIZE-1:0] FEMap;
    reg [FE_MAP_SIZE-1:0] postPReLUFEMap;

    reg [SHRINK_MAP_SIZE-1:0] shrinkMap;
    reg [SHIRNK_MAP_SIZE-1:0] postPReLUShrinkMap;

    reg [MAPPING_MAP_SIZE-1:0] mappingMap;
    reg [MAPPING_MAP_SIZE-1:0] postPReLUMappingMap;

    reg [EXPAND_MAP_SIZE-1:0] expandMap;
    reg [EXPAND_MAP_SIZE-1:0] postPReLUExpandMap;

    reg [DECONV_MAP_SIZE-1:0] deconvMap;

    wire []

    FeatureExtraction fe1(.clk_in(clk_in), .rst_in(rst_in), .map_in(), .start(), .map_out(FEMap), .save(), .ready(FeatureExtractionDone));
    PreLU PreLU1(.clk_in(clk_in), .rst_in(rst_in), .map_in(FEMap), .start(FeatureExtractionDone), .map_out(postPReLUFEMap), .ready(PReLU1Done));

    //shrinking
    PreLU PreLU2(.clk_in(clk_in), .rst_in(rst_in), .map_in(shrinkMap), .start(ShrinkingDone), .map_out(postPReLUShrinkMap), .ready(PReLU2Done));

    //mapping
    PreLU PreLU3(.clk_in(clk_in), .rst_in(rst_in), .map_in(mappingMap), .start(MappingDone), .map_out(postPReLUMappingMap), .ready(PReLU3Done));

    //expanding
    PreLU PreLU4(.clk_in(clk_in), .rst_in(rst_in), .map_in(expandMap), .start(ExpandDone), .map_out(postPReLUExpandMap), .ready(PReLU4Done));
    
    //deconvolution

    
endmodule