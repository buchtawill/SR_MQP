

def make_hls_1x1(name:str, in_ch:int, out_ch:int, in_width_pix:int, num_pe:int):
    func =  f"void conv_{name}(ch_stream_t tile_in[IN_CHN_LAYER_{name.upper()}], ch_stream_t map_out[OUT_CHN_LAYER_{name.upper()}]){{\n"
    func += f"    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py\n"
    func += f"    fixed_4_8_t slider[IN_CHN_LAYER_{name.upper()}];\n"
    func +=  "    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete\n"
    if(num_pe < out_ch):
        func += f"    ch_stream_t inbuf[IN_CHN_LAYER_{name.upper()}];\n\n"
    
    # Calculate how many times need to go thru PE's
    func += f"    int num_pe_loops = OUT_CHN_LAYER_{name.upper()} / NUM_PE_LAYER_{name.upper()};\n"
    func += f"    if((OUT_CHN_LAYER_{name.upper()} % NUM_PE_LAYER_{name.upper()}) != 0) num_pe_loops++;\n\n"
    
    func +=  "    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){\n"
    func += f"        // WARNING: if number fmap % num_pe != 0, utilization explodes!!\n"
    func += f"        int low_filter = (pe_loop*NUM_PE_LAYER_{name.upper()});\n"
    func += f"        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) < OUT_CHN_LAYER_{name.upper()} ? ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) : OUT_CHN_LAYER_{name.upper()};\n"
    func += f"        for(int row = 0; row < {in_width_pix}; row++){{\n\n" # Calculate size of padding
    
    
    func += f"            // Go across the row\n"
    func += f"            for(int col = 0; col < {in_width_pix}; col++){{\n"
    func +=  "                #pragma HLS PIPELINE II=1\n"
    # Read next slider value
    func +=  "                // Read the next value into the slider\n"
    func += f"                for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func +=  "                    #pragma HLS UNROLL\n"
    func += f"                    fixed_4_8_t next_data;\n"
    func += f"                    if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    if(num_pe < out_ch):
        func += f"                    else             next_data = inbuf[ch].read();\n\n"
    func += f"                    slider[ch] = next_data;\n"
    if(num_pe < out_ch):
        func +=  "                    if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func += f"                }}\n\n" # channel loop (line 202)
    
    func += f"                for(int filter = low_filter; filter < high_filter; filter++){{\n"
    func += f"                    fixed_4_8_t mac = 0.0;\n"
    func += f"                    for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func += f"                        #pragma HLS UNROLL\n"
    func += f"                        mac += slider[ch] * weights_layer_{name}[filter][ch];\n"
    func += f"                    }}\n" # Channel loop 
    func += f"                    map_out[filter].write(prelu(conv_{name}_prelu[filter], (mac + conv_{name}_bias[filter])));\n"
    func += f"                }} // For every filter \n " # Filter loop 2 (line 214)    
    func +=  "            } // For every column \n" # column loop (line 190)    
    func +=  "        } // For every row\n" # row loop (line 170)
    func +=  "    } // For number of times thru PE\n" # pe loop (line 166)
    func +=  "}\n" # Function body
    
    defines  = f"#define IN_CHN_LAYER_{name.upper()}    {in_ch}\n"
    defines += f"#define OUT_CHN_LAYER_{name.upper()}   {out_ch}\n"
    defines += f"#define NUM_PE_LAYER_{name.upper()}    {num_pe}\n"
    weight_arr  = f"const fixed_4_8_t conv_{name}_prelu[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t conv_{name}_bias[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t weights_layer_{name}[{out_ch}][{in_ch}];\n"
    
    return func, (defines, weight_arr)

def make_hls_conv_func(name:str, in_ch:int, out_ch:int, kernel_size:int, in_width_pix:int, num_pe:int):
    
    # Note: Number of PE's, input channels, and output channels has to be defined at compile time 
    # and thus must use macros
    
    padding = kernel_size // 2
    if(kernel_size == 1):
        padding = 0
        
    in_padded_size = in_width_pix + 2*padding
    
    func =  f"void conv_{name}(ch_stream_t tile_in[IN_CHN_LAYER_{name.upper()}], ch_stream_t map_out[OUT_CHN_LAYER_{name.upper()}]){{\n"
    func += f"    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py\n"
    func += f"    fixed_4_8_t slider[IN_CHN_LAYER_{name.upper()}][{kernel_size}];\n"
    func +=  "    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete\n"
    if(num_pe < out_ch):
        func += f"    ch_stream_t inbuf[IN_CHN_LAYER_{name.upper()}];\n\n"
    # Declare PE's
    for i in range(kernel_size-1):
        func += f"    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum{i+1}[NUM_PE_LAYER_{name.upper()}];\n"

    # Partition and assign to BRAM
    for i in range(kernel_size-1):
        func += f"    #pragma HLS STREAM variable=psum{i+1} depth={in_width_pix}\n"
        func += f"    #pragma HLS RESOURCE variable=psum{i+1} core=FIFO_BRAM\n"
    
    func +='\n'
    # Calculate how many times need to go thru PE's
    func += f"    int num_pe_loops = OUT_CHN_LAYER_{name.upper()} / NUM_PE_LAYER_{name.upper()};\n"
    func += f"    if((OUT_CHN_LAYER_{name.upper()} % NUM_PE_LAYER_{name.upper()}) != 0) num_pe_loops++;\n\n"
    
    func +=  "    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){\n"
    func += f"        // WARNING: if number fmap % num_pe != 0, utilization explodes!!\n"
    func += f"        int low_filter = (pe_loop*NUM_PE_LAYER_{name.upper()});\n"
    func += f"        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) < OUT_CHN_LAYER_{name.upper()} ? ((pe_loop+1)*NUM_PE_LAYER_{name.upper()}) : OUT_CHN_LAYER_{name.upper()};\n"
    func += f"        for(int row = 0; row < {in_width_pix + 2*padding}; row++){{\n\n" # Calculate size of padding
    func +=  "            // Prep the slider\n"
    func += f"            for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func +=  "                #pragma HLS UROLL\n"
    func += f"                for(int idx = 0; idx < {kernel_size-1}; idx++){{\n"
    func +=  "                    #pragma HLS PIPELINE II=1\n"
    
    # First two or last two rows, pad with zeros
    func += f"                    if((row < {padding}) || (row >= {in_width_pix + padding*1}) || (idx < {padding})) slider[ch][idx] = 0;\n"
    func += f"                    else{{\n"
    func += f"                        fixed_4_8_t next_data;\n"
    func += f"                        if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    if(num_pe < out_ch):
        func += f"                        else             next_data = inbuf[ch].read();\n\n"
    func += f"                        slider[ch][idx] = next_data;\n"
    if(num_pe < out_ch):
        func +=  "                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func +=  "                    }\n" # else not middle
    func +=  "                }\n" # idx loop
    func +=  "            }\n\n" # ch loop
    
    func += f"            // Go across the row\n"
    func += f"            for(int col = {kernel_size -1}; col < {in_width_pix + 2*padding}; col++){{\n"
    func +=  "                #pragma HLS PIPELINE II=1\n"
    
    # Read next slider value
    func +=  "                // Read the next value into the slider\n"
    func += f"                for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func +=  "                    #pragma HLS UNROLL\n\n"
    func += f"                    if((row < {padding}) || (row >= {in_width_pix + padding*1}) || (col >= {in_width_pix + padding*1})) slider[ch][{kernel_size-1}] = 0;\n"
    func += f"                    else{{\n"
    func += f"                        fixed_4_8_t next_data;\n"
    func += f"                        if(pe_loop == 0) next_data = tile_in[ch].read();\n"
    if(num_pe < out_ch):
        func += f"                        else             next_data = inbuf[ch].read();\n\n"
    func += f"                        slider[ch][{kernel_size-1}] = next_data;\n"
    if(num_pe < out_ch):
        func +=  "                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);\n"
    func += f"                    }}\n" # else read data (line 205)
    func += f"                }}\n\n" # channel loop (line 202)
    
    func += f"                for(int filter = low_filter; filter < high_filter; filter++){{\n"
    func += f"                    #pragma HLS UNROLL\n"
    func += f"                    int pe_idx = filter % NUM_PE_LAYER_{name.upper()};\n"
    
    for i in range(kernel_size):
        func += f"                    fixed_4_8_t mac{i} = 0.0;\n"
        
    func += f"                    fixed_4_8_t "
    for i in range(1,kernel_size-1):
        func += f"row{i}_psum, "
    func += f"row{kernel_size-1}_psum;\n\n"
    
    func += f"                    for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func += f"                        #pragma HLS UNROLL\n"
    last_row_conv = in_padded_size - (kernel_size - 1)
    
    func += f"                        if(row < {last_row_conv})             "
    func += f"mac0 += perform_mac{kernel_size}(weights_layer_{name}[filter][ch][0], slider[ch]);\n"
    for i in range(1, kernel_size-1):
        func += f"                        if(row >= {i} && row < {last_row_conv + i}) "
        func += f"mac{i} += perform_mac{kernel_size}(weights_layer_{name}[filter][ch][{i}], slider[ch]);\n"
    func += f"                        if(row >= {kernel_size - 1})             "
    func += f"mac{kernel_size-1} += perform_mac{kernel_size}(weights_layer_{name}[filter][ch][{kernel_size-1}], slider[ch]);\n"
    func += f"                    }}\n" # Channel loop 
    func += f"\n"
    
    func += f"                    if(row < {last_row_conv}){{\n"
    func += f"                        psum1[pe_idx].write(mac0);\n"
    func += f"                    }}\n"
    for i in range(1, kernel_size-1):
        func += f"                    if(row >= {i} && row < {last_row_conv + i}) {{\n"
        func += f"                        row{i}_psum = psum{i}[pe_idx].read();\n"
        func += f"                        psum{i+1}[pe_idx].write(row{i}_psum + mac{i});\n"
        func += f"                    }}\n"
    func += f"                    if(row >= {kernel_size - 1}) {{\n"
    func += f"                        row{kernel_size - 1}_psum = psum{kernel_size - 1}[pe_idx].read();\n"
    func += f"                        fixed_4_8_t pre_activation = row{kernel_size - 1}_psum + mac{kernel_size-1} + conv_{name}_bias[filter];\n"
    func += f"                        map_out[filter].write(prelu(conv_{name}_prelu[filter], pre_activation));\n"
    func += f"                    }}\n"
    
        
    func += f"                }} // For every filter \n\n" # Filter loop 2 (line 214)
    
    func += f"               for(int ch = 0; ch < IN_CHN_LAYER_{name.upper()}; ch++){{\n"
    func += f"                   #pragma HLS_UNROLL\n"
    for i in range(kernel_size-1):
        func += f"                   slider[ch][{i}] = slider[ch][{i+1}];\n"
    func += f"                }}\n"
    
    func +=  "            } // For every column \n" # column loop (line 190)
    func +=  "        } // For every row\n" # row loop (line 170)
    func +=  "    } // For number of times thru PE\n" # pe loop (line 166)
    func +=  "}\n" # Function body
    
    defines  = f"#define IN_CHN_LAYER_{name.upper()}    {in_ch}\n"
    defines += f"#define OUT_CHN_LAYER_{name.upper()}   {out_ch}\n"
    defines += f"#define NUM_PE_LAYER_{name.upper()}    {num_pe}\n"
    
    weight_arr  = f"const fixed_4_8_t conv_{name}_prelu[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t conv_{name}_bias[{out_ch}];\n"
    weight_arr += f"const fixed_4_8_t weights_layer_{name}[{out_ch}][{in_ch}][{kernel_size}][{kernel_size}];\n"
    return func, (defines, weight_arr)


if __name__ == '__main__':
    
    extraction_func, extraction_defines = make_hls_conv_func('feature_extraction0', in_ch=3, out_ch=44, kernel_size=5, in_width_pix=28, num_pe=4)
    # print(extraction_func)
    # print(extraction_defines)
    # exit()
    shrink_body, shrink_defines = make_hls_1x1('shrink0', in_ch=44, out_ch=12, in_width_pix=28, num_pe=2)
    map0_body, map0_defines = make_hls_conv_func('map0', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    map2_body, map2_defines = make_hls_conv_func('map2', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    map4_body, map4_defines = make_hls_conv_func('map4', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    map6_body, map6_defines = make_hls_conv_func('map6', in_ch=12, out_ch=12, kernel_size=3, in_width_pix=28, num_pe=4)
    expand_body, expand_defines = make_hls_1x1('expand0', in_ch=12, out_ch=44, in_width_pix=28, num_pe=2)
    
    # # The defines
    # print(extraction_defines[0])
    # print(shrink_defines[0])
    # print(map0_defines[0])
    # print(map2_defines[0])
    # print(map4_defines[0])
    # print(map6_defines[0])
    # print(expand_defines[0])
    
    # # The weight array declarations
    # print(extraction_defines[1])
    # print(shrink_defines[1])
    # print(map0_defines[1])
    # print(map2_defines[1])
    # print(map4_defines[1])
    # print(map6_defines[1])
    # print(expand_defines[1])
    
    print(extraction_func)
    print(shrink_body)
    print(map0_body)
    print(map2_body)
    print(map4_body)
    print(map6_body)
    print(expand_body)