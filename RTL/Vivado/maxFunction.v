module maxFunction #(
    parameter NUM_SIZE = 1;
) (
    input [NUM_SIZE : 0] a,
    input [NUM_SIZE : 0] b,
    output reg [NUM_SIZE : 0] out
);

    always @ (*) begin
        if(a > b) begin
            out = a;
        end
        else begin
            out = b;
        end
    end
    
endmodule