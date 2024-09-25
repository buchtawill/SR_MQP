module PReLU #(
    parameter NEG_COEFFICIENT = 1 //replace with value from training
    parameter MAP_SIZE = 128;
) (
    clk_in,
    rst_in,
    map_in,
    start,
    map_out,
    ready
);

    input clk_in;
    input rst_in;
    input signed [MAP_SIZE-1:0] map_in;
    input start;
    output signed [MAP_SIZE-1:0] map_out = 0;
    output ready = 0;

    input reg maxIn1 = 0;
    input reg maxIn2 = 0;
    output reg maxResult;

    input reg minIn1 = 0;
    input reg minIn2 = 0;
    output reg minResult;

    maxFunction max1(.a(maxIn1), .b(maxIn2), .out(maxResult));
    minFunction min1(.a(minIn1), .b(minIn1), .out(minResult));

    always @(clk_in) begin
        if(start) begin
            //apply PReLU function to whole feature map
            for(int i = 0; i <= MAP_SIZE; i++) begin
                if(rst_in) begin
                    i = 0;
                end
                //signal that the feature map is ready after function has been applied
                else if (i >= MAP_SIZE) begin
                    ready = 1;
                end
                else begin
                    maxIn1 = map_in[i];
                    minIn1 = map_in[i];
                    //PReLU function
                    map_out[i] = maxResult + NEG_COEFFICIENT*minResult; 
                end
            end //end for
        end //end if
    end //end always
    
endmodule