#include "conv2d.h"
#include <hls_stream.h>
#include <ap_axi_sdata.h>
#include <iostream>
#include <stdio.h>
#include "fsrcnn_weights.h"

void fill_input_fifo(hls::stream<axis_t> &axis_in,
					 hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &data_fifo_in){

	axis_t tmp_stream;
	tmp_stream.last = 0;

	int in_ptr_abs = 0;
	while(!tmp_stream.last){
		// if(!axis_in.empty()) {
		tmp_stream = axis_in.read();
		stream_data_t tmp_data = tmp_stream.data;

		data_fifo_in.write(tmp_data);
		// }
	}
}



void stream_samples_out(hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX * 2 * 2> map_upscaled[OUT_CHN_LAYER_DECONV0],
						hls::stream<axis_t> &out_stream){
	

    // Each stream here is 3136 deep
    // 784 beats --> 4 pixels per
    // Each beat is 4 elements
	axis_t tmp_stream;
    tmp_stream.data;
    // int num_beats = STREAM_BEATS_PER_TILE *2 *2;
    // int num_beats = INPUT_WIDTH_PIX * INPUT_WIDTH_PIX * 2 * 2 * 4 / BYTES_PER_TRANSFER;
    int num_beats = 1024;
	for(int out_ptr = 0; out_ptr < num_beats; out_ptr++){
        #pragma HLS PIPELINE II=4

        stream_data_t tmp_data = 0;
        
        fixed_9_8_t r0 = ((fixed_9_8_t)map_upscaled[0].read()) << 8;
        fixed_9_8_t r1 = ((fixed_9_8_t)map_upscaled[0].read()) << 8;
        fixed_9_8_t r2 = ((fixed_9_8_t)map_upscaled[0].read()) << 8;
        fixed_9_8_t r3 = ((fixed_9_8_t)map_upscaled[0].read()) << 8;
        
        fixed_9_8_t g0 = ((fixed_9_8_t)map_upscaled[1].read()) << 8;
        fixed_9_8_t g1 = ((fixed_9_8_t)map_upscaled[1].read()) << 8;
        fixed_9_8_t g2 = ((fixed_9_8_t)map_upscaled[1].read()) << 8;
        fixed_9_8_t g3 = ((fixed_9_8_t)map_upscaled[1].read()) << 8;
        
        fixed_9_8_t b0 = ((fixed_9_8_t)map_upscaled[2].read()) << 8;
        fixed_9_8_t b1 = ((fixed_9_8_t)map_upscaled[2].read()) << 8;
        fixed_9_8_t b2 = ((fixed_9_8_t)map_upscaled[2].read()) << 8;
        fixed_9_8_t b3 = ((fixed_9_8_t)map_upscaled[2].read()) << 8;
        
        // Clip to 0...255
        uint8_t r0_8b = (r0 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((r0 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : r0);
        uint8_t r1_8b = (r1 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((r1 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : r1);
        uint8_t r2_8b = (r2 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((r2 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : r2);
        uint8_t r3_8b = (r3 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((r3 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : r3);
        
        uint8_t g0_8b = (g0 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((g0 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : g0);
        uint8_t g1_8b = (g1 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((g1 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : g1);
        uint8_t g2_8b = (g2 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((g2 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : g2);
        uint8_t g3_8b = (g3 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((g3 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : g3);
        
        uint8_t b0_8b = (b0 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((b0 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : b0);
        uint8_t b1_8b = (b1 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((b1 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : b1);
        uint8_t b2_8b = (b2 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((b2 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : b2);
        uint8_t b3_8b = (b3 > (fixed_9_8_t)255) ? (fixed_9_8_t)255 : ((b3 < (fixed_9_8_t)0) ? (fixed_9_8_t)0 : b3);
        
        // Put in the right location
        // Discard tmp_data.range(31, 24)
        // Discard tmp_data.range(63, 56)
        // Discard tmp_data.range(95, 88)
		// Discard tmp_data.range(127, 120)

		tmp_data.range(7, 0)   = r0_8b;
		tmp_data.range(15, 8)  = g0_8b;
		tmp_data.range(23, 16) = b0_8b;
	
		tmp_data.range(39, 32) = r1_8b;
		tmp_data.range(47, 40) = g1_8b;
		tmp_data.range(55, 48) = b1_8b;
	
		tmp_data.range(71, 64) = r2_8b;
		tmp_data.range(79, 72) = g2_8b;
		tmp_data.range(87, 80) = b2_8b;
	
		tmp_data.range(103, 96)  = r3_8b;
		tmp_data.range(111, 104) = g3_8b;
		tmp_data.range(119, 112) = b3_8b;
        
		// 128 bits = 16 bytes = 16 bits of keep
		// Max width is 128 bits
		// Fewer bytes will truncate
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
		tmp_stream.last = (out_ptr == (num_beats - 1));
        tmp_stream.data = tmp_data;

		out_stream.write(tmp_stream);
	}
}

// Read from the input FIFO and convert the data to fixed point format
void fill_img_arr(hls::stream<axis_t> &input_fifo,
				  fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL]){

	int fmap_ptr_abs = 0;
	FILL_IMG_ARR: 
	for(int out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
		axis_t tmp_strm = input_fifo.read();
		stream_data_t tmp_data = tmp_strm.data;

		for(int pixel_no = 0; pixel_no < 4; pixel_no++){
			for(int byte = 0; byte < 3; byte++){
				int row_idx = (fmap_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
				int col_idx = (fmap_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
				int chn_idx =  fmap_ptr_abs % BYTES_PER_PIXEL;
	
				int hi = (pixel_no * 4 * 8) + ((byte + 1) * 8) - 1;
				int low = hi - 7;
				fixed_9_8_t tmp_98 = tmp_data.range(hi, low);
				fixed_4_8_t tmp_val = tmp_98 >> 8;

				img_in[row_idx][col_idx][chn_idx] = tmp_val;
	
				fmap_ptr_abs++;
			}
		}
	}
}

/**
 * Load the tile_in array with values, ignoring any padding
 */
void prep_tile(hls::stream<axis_t> &in_stream, hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> tile_in[3]){
	axis_t tmp_stream;
	tmp_stream.last = 0;

	// Do the 28 rows of pixels
	READ_ROWS:
	for(int row = 0; row < INPUT_HEIGHT_PIX; row++){

		// Fill the meat and potatos 
		for(int beat = 0; beat < BEATS_PER_ROW; beat++){
		#pragma HLS PIPELINE II=4
			tmp_stream = in_stream.read();
			
			// 4 pixels per transfer from a 128-bit stream
			stream_data_t tmp_data = tmp_stream.data;

			// Pixel 0
			fixed_9_8_t r0 = tmp_data.range(7, 0);
			fixed_9_8_t g0 = tmp_data.range(15, 8);
			fixed_9_8_t b0 = tmp_data.range(23, 16);
			// Discard tmp_data.range(31, 24)
	
			// Pixel 1
			fixed_9_8_t r1 = tmp_data.range(39, 32);
			fixed_9_8_t g1 = tmp_data.range(47, 40);
			fixed_9_8_t b1 = tmp_data.range(55, 48);
			// Discard tmp_data.range(63, 56)
	
			// Pixel 2
			fixed_9_8_t r2 = tmp_data.range(71, 64);
			fixed_9_8_t g2 = tmp_data.range(79, 72);
			fixed_9_8_t b2 = tmp_data.range(87, 80);
			// Discard tmp_data.range(95, 88)
	
			// Pixel 3
			fixed_9_8_t r3 = tmp_data.range(103, 96);
			fixed_9_8_t g3 = tmp_data.range(111, 104);
			fixed_9_8_t b3 = tmp_data.range(119, 112);
			// Discard tmp_data.range(127, 120)

            // printf("Got RGB %3d, %3d, %3d\n", (int)r0, (int)g0, (int)b0);
            // printf("Got RGB %3d, %3d, %3d\n", (int)r1, (int)g1, (int)b1);
            // printf("Got RGB %3d, %3d, %3d\n", (int)r2, (int)g2, (int)b2);
            // printf("Got RGB %3d, %3d, %3d\n", (int)r3, (int)g3, (int)b3);

			// Divide by 256, cast to 12 bit fixed, write to FIFO
			tile_in[0].write((fixed_4_8_t)(r0 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r1 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r2 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r3 >> 8));

			tile_in[1].write((fixed_4_8_t)(g0 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g1 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g2 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g3 >> 8));

			tile_in[2].write((fixed_4_8_t)(b0 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b1 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b2 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b3 >> 8));
		}
	}
}

fixed_4_8_t perform_mac9(const fixed_4_8_t weights[9], fixed_4_8_t slider[9]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	for(int w = 0; w < 9; w++){
		#pragma HLS UNROLL factor=3
		sum += weights[w] * slider[w];
	}
	return sum;
}

fixed_4_8_t perform_mac7(const fixed_4_8_t weights[7], fixed_4_8_t slider[7]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	for(int w = 0; w < 7; w++){
		#pragma HLS UNROLL factor=1
		sum += weights[w] * slider[w];
	}
	return sum;
}

fixed_4_8_t perform_mac5(const fixed_4_8_t weights[5], fixed_4_8_t slider[5]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	for(int w = 0; w < 5; w++){
		#pragma HLS UNROLL
		sum += weights[w] * slider[w];
	}
	return sum;
}

fixed_4_8_t perform_mac3(const fixed_4_8_t weights[3], fixed_4_8_t slider[3]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	for(int w = 0; w < 3; w++){
		#pragma HLS UNROLL
		sum += weights[w] * slider[w];
	}
	return sum;
}

fixed_4_8_t prelu(const fixed_4_8_t weight, fixed_4_8_t value){
	#pragma HLS INLINE
	if(value >= 0) return value;
	else return value * weight;
}

/**
 * Get the next value from the input given the row, col, and input stream. 
 * @warning Assumes 9x9 kernel for transposed convolution 28x28 --> 56x56
 */
fixed_4_8_t get_next_tconv_9(int row, int col, ch_stream_t *input, bool *zero){
	#pragma HLS INLINE
    if((row <= 3) || ((row % 2) == 1) || (row >= 59)) {
        *zero = true;
        return (fixed_4_8_t)0.0f;
    }
    if((col <= 3) || ((col % 2) == 1) || (col >= 59)) {
        *zero = true;
        return (fixed_4_8_t)0.0f;
    }
    else {
        *zero = false;
        return input->read();
    }
}

/**
 * Get the next value from the input given the row, col, and input stream. 
 * @warning Assumes 7x7 kernel for transposed convolution 28x28 --> 56x56
 */
fixed_4_8_t get_next_tconv_7(int row, int col, ch_stream_t *input, bool *zero){

    // Input is supposed to be 56x56, padding is 3x3 so input is 62x62
    // Input is supposed to be 64x64, padding is 3x3 so input is 70x70
    // 0 1 2 3 4 5 6 7 8 9 10
    // ---------------------
    // 0 0 0 0 0 0 0 0 0 0 0 ... 
    // 0 0 0 0 0 0 0 0 0 0 0 ... 
    // 0 0 0 0 0 0 0 0 0 0 0 ... 
    // 0 0 0 d 0 d 0 d 0 d 0 ... 
    // 0 0 0 d 0 d 0 d 0 d 0 ...  
    
    // ....data zero data zero pad pad pad
    // ... 59 60 61 62 63 64 65 66 67 68 69 
    // ... d  0  d  0  d  0  d  0  0  0  0
	#pragma HLS INLINE
    if((row <= 2) || ((row % 2) == 0) || (row >= 66)) {
        *zero = true;
        return (fixed_4_8_t)0.0f;
    }
    if((col <= 2) || ((col % 2) == 0) || (col >= 66)) {
        *zero = true;
        return (fixed_4_8_t)0.0f;
    }
    else {
        *zero = false;
        return input->read();
    }
}

// 

////////////////////////////////// Auto-generated code goes here //////////////////////////////////
void conv_feature_extraction0(ch_stream_t tile_in[IN_CHN_LAYER_FEATURE_EXTRACTION0], ch_stream_t map_out[OUT_CHN_LAYER_FEATURE_EXTRACTION0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_FEATURE_EXTRACTION0][5];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_FEATURE_EXTRACTION0];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_FEATURE_EXTRACTION0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_FEATURE_EXTRACTION0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum3[NUM_PE_LAYER_FEATURE_EXTRACTION0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum4[NUM_PE_LAYER_FEATURE_EXTRACTION0];
    #pragma HLS STREAM variable=psum1 depth=32
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=32
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum3 depth=32
    #pragma HLS RESOURCE variable=psum3 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum4 depth=32
    #pragma HLS RESOURCE variable=psum4 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_FEATURE_EXTRACTION0 / NUM_PE_LAYER_FEATURE_EXTRACTION0;
    if((OUT_CHN_LAYER_FEATURE_EXTRACTION0 % NUM_PE_LAYER_FEATURE_EXTRACTION0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_FEATURE_EXTRACTION0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_FEATURE_EXTRACTION0) < OUT_CHN_LAYER_FEATURE_EXTRACTION0 ? ((pe_loop+1)*NUM_PE_LAYER_FEATURE_EXTRACTION0) : OUT_CHN_LAYER_FEATURE_EXTRACTION0;
        for(int row = 0; row < 36; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                #pragma HLS UNROLL
                for(int idx = 0; idx < 4; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 2) || (row >= 34) || (idx < 2)) slider[ch][idx] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][idx] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 4; col < 36; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                    #pragma HLS UNROLL

                    if((row < 2) || (row >= 34) || (col >= 34)) slider[ch][4] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][4] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    int pe_idx = filter % NUM_PE_LAYER_FEATURE_EXTRACTION0;
                    fixed_4_8_t mac0 = 0.0;
                    fixed_4_8_t mac1 = 0.0;
                    fixed_4_8_t mac2 = 0.0;
                    fixed_4_8_t mac3 = 0.0;
                    fixed_4_8_t mac4 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum, row3_psum, row4_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                        #pragma HLS UNROLL
                        if(row < 32)             mac0 += perform_mac5(weights_layer_feature_extraction0[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 33) mac1 += perform_mac5(weights_layer_feature_extraction0[filter][ch][1], slider[ch]);
                        if(row >= 2 && row < 34) mac2 += perform_mac5(weights_layer_feature_extraction0[filter][ch][2], slider[ch]);
                        if(row >= 3 && row < 35) mac3 += perform_mac5(weights_layer_feature_extraction0[filter][ch][3], slider[ch]);
                        if(row >= 4)             mac4 += perform_mac5(weights_layer_feature_extraction0[filter][ch][4], slider[ch]);
                    }

                    if(row < 32){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 33) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2 && row < 34) {
                        row2_psum = psum2[pe_idx].read();
                        psum3[pe_idx].write(row2_psum + mac2);
                    }
                    if(row >= 3 && row < 35) {
                        row3_psum = psum3[pe_idx].read();
                        psum4[pe_idx].write(row3_psum + mac3);
                    }
                    if(row >= 4) {
                        row4_psum = psum4[pe_idx].read();
                        fixed_4_8_t pre_activation = row4_psum + mac4 + conv_feature_extraction0_bias[filter];
                        map_out[filter].write(prelu(conv_feature_extraction0_prelu[filter], pre_activation));
                    }
                } // For every filter 

               for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                   #pragma HLS UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                   slider[ch][2] = slider[ch][3];
                   slider[ch][3] = slider[ch][4];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_shrink0(ch_stream_t tile_in[IN_CHN_LAYER_SHRINK0], ch_stream_t map_out[OUT_CHN_LAYER_SHRINK0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_SHRINK0];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_SHRINK0];

    int num_pe_loops = OUT_CHN_LAYER_SHRINK0 / NUM_PE_LAYER_SHRINK0;
    if((OUT_CHN_LAYER_SHRINK0 % NUM_PE_LAYER_SHRINK0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_SHRINK0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_SHRINK0) < OUT_CHN_LAYER_SHRINK0 ? ((pe_loop+1)*NUM_PE_LAYER_SHRINK0) : OUT_CHN_LAYER_SHRINK0;
        for(int row = 0; row < 32; row++){

            // Go across the row
            for(int col = 0; col < 32; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_SHRINK0; ch++){
                    #pragma HLS UNROLL
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = tile_in[ch].read();
                    else             next_data = inbuf[ch].read();

                    slider[ch] = next_data;
                    if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    fixed_4_8_t mac = 0.0;
                    for(int ch = 0; ch < IN_CHN_LAYER_SHRINK0; ch++){
                        #pragma HLS UNROLL
                        mac += slider[ch] * weights_layer_shrink0[filter][ch];
                    }
                    map_out[filter].write(prelu(conv_shrink0_prelu[filter], (mac + conv_shrink0_bias[filter])));
                } // For every filter 
             } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_map0(ch_stream_t tile_in[IN_CHN_LAYER_MAP0], ch_stream_t map_out[OUT_CHN_LAYER_MAP0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_MAP0][3];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_MAP0];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP0];
    #pragma HLS STREAM variable=psum1 depth=32
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=32
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP0 / NUM_PE_LAYER_MAP0;
    if((OUT_CHN_LAYER_MAP0 % NUM_PE_LAYER_MAP0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP0) < OUT_CHN_LAYER_MAP0 ? ((pe_loop+1)*NUM_PE_LAYER_MAP0) : OUT_CHN_LAYER_MAP0;
        for(int row = 0; row < 34; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                #pragma HLS UNROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 33) || (idx < 1)) slider[ch][idx] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][idx] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 34; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 33) || (col >= 33)) slider[ch][2] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    int pe_idx = filter % NUM_PE_LAYER_MAP0;
                    fixed_4_8_t mac0 = 0.0;
                    fixed_4_8_t mac1 = 0.0;
                    fixed_4_8_t mac2 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                        #pragma HLS UNROLL
                        if(row < 32)             mac0 += perform_mac3(weights_layer_map0[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 33) mac1 += perform_mac3(weights_layer_map0[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map0[filter][ch][2], slider[ch]);
                    }

                    if(row < 32){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 33) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2) {
                        row2_psum = psum2[pe_idx].read();
                        fixed_4_8_t pre_activation = row2_psum + mac2 + conv_map0_bias[filter];
                        map_out[filter].write(prelu(conv_map0_prelu[filter], pre_activation));
                    }
                } // For every filter 

               for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                   #pragma HLS UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_map2(ch_stream_t tile_in[IN_CHN_LAYER_MAP2], ch_stream_t map_out[OUT_CHN_LAYER_MAP2]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_MAP2][3];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_MAP2];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP2];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP2];
    #pragma HLS STREAM variable=psum1 depth=32
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=32
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP2 / NUM_PE_LAYER_MAP2;
    if((OUT_CHN_LAYER_MAP2 % NUM_PE_LAYER_MAP2) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP2);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP2) < OUT_CHN_LAYER_MAP2 ? ((pe_loop+1)*NUM_PE_LAYER_MAP2) : OUT_CHN_LAYER_MAP2;
        for(int row = 0; row < 34; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                #pragma HLS UNROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 33) || (idx < 1)) slider[ch][idx] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][idx] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 34; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 33) || (col >= 33)) slider[ch][2] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    int pe_idx = filter % NUM_PE_LAYER_MAP2;
                    fixed_4_8_t mac0 = 0.0;
                    fixed_4_8_t mac1 = 0.0;
                    fixed_4_8_t mac2 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                        #pragma HLS UNROLL
                        if(row < 32)             mac0 += perform_mac3(weights_layer_map2[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 33) mac1 += perform_mac3(weights_layer_map2[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map2[filter][ch][2], slider[ch]);
                    }

                    if(row < 32){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 33) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2) {
                        row2_psum = psum2[pe_idx].read();
                        fixed_4_8_t pre_activation = row2_psum + mac2 + conv_map2_bias[filter];
                        map_out[filter].write(prelu(conv_map2_prelu[filter], pre_activation));
                    }
                } // For every filter 

               for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                   #pragma HLS UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_map4(ch_stream_t tile_in[IN_CHN_LAYER_MAP4], ch_stream_t map_out[OUT_CHN_LAYER_MAP4]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_MAP4][3];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_MAP4];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP4];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP4];
    #pragma HLS STREAM variable=psum1 depth=32
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=32
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP4 / NUM_PE_LAYER_MAP4;
    if((OUT_CHN_LAYER_MAP4 % NUM_PE_LAYER_MAP4) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP4);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP4) < OUT_CHN_LAYER_MAP4 ? ((pe_loop+1)*NUM_PE_LAYER_MAP4) : OUT_CHN_LAYER_MAP4;
        for(int row = 0; row < 34; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                #pragma HLS UNROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 33) || (idx < 1)) slider[ch][idx] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][idx] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 34; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 33) || (col >= 33)) slider[ch][2] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    int pe_idx = filter % NUM_PE_LAYER_MAP4;
                    fixed_4_8_t mac0 = 0.0;
                    fixed_4_8_t mac1 = 0.0;
                    fixed_4_8_t mac2 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                        #pragma HLS UNROLL
                        if(row < 32)             mac0 += perform_mac3(weights_layer_map4[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 33) mac1 += perform_mac3(weights_layer_map4[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map4[filter][ch][2], slider[ch]);
                    }

                    if(row < 32){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 33) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2) {
                        row2_psum = psum2[pe_idx].read();
                        fixed_4_8_t pre_activation = row2_psum + mac2 + conv_map4_bias[filter];
                        map_out[filter].write(prelu(conv_map4_prelu[filter], pre_activation));
                    }
                } // For every filter 

               for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                   #pragma HLS UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_expand0(ch_stream_t tile_in[IN_CHN_LAYER_EXPAND0], ch_stream_t map_out[OUT_CHN_LAYER_EXPAND0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_EXPAND0];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_EXPAND0];

    int num_pe_loops = OUT_CHN_LAYER_EXPAND0 / NUM_PE_LAYER_EXPAND0;
    if((OUT_CHN_LAYER_EXPAND0 % NUM_PE_LAYER_EXPAND0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_EXPAND0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_EXPAND0) < OUT_CHN_LAYER_EXPAND0 ? ((pe_loop+1)*NUM_PE_LAYER_EXPAND0) : OUT_CHN_LAYER_EXPAND0;
        for(int row = 0; row < 32; row++){

            // Go across the row
            for(int col = 0; col < 32; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_EXPAND0; ch++){
                    #pragma HLS UNROLL
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = tile_in[ch].read();
                    else             next_data = inbuf[ch].read();

                    slider[ch] = next_data;
                    if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    fixed_4_8_t mac = 0.0;
                    for(int ch = 0; ch < IN_CHN_LAYER_EXPAND0; ch++){
                        #pragma HLS UNROLL
                        mac += slider[ch] * weights_layer_expand0[filter][ch];
                    }
                    map_out[filter].write(prelu(conv_expand0_prelu[filter], (mac + conv_expand0_bias[filter])));
                } // For every filter 
             } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_deconv0(ch_stream_t tile_in[IN_CHN_LAYER_DECONV0], upscaled_stream_t map_out[OUT_CHN_LAYER_DECONV0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_DECONV0][7];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_DECONV0];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum1[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum2[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum3[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum4[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum5[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum6[NUM_PE_LAYER_DECONV0];
    #pragma HLS STREAM variable=psum1 depth=64
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=64
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum3 depth=64
    #pragma HLS RESOURCE variable=psum3 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum4 depth=64
    #pragma HLS RESOURCE variable=psum4 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum5 depth=64
    #pragma HLS RESOURCE variable=psum5 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum6 depth=64
    #pragma HLS RESOURCE variable=psum6 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_DECONV0 / NUM_PE_LAYER_DECONV0;
    if((OUT_CHN_LAYER_DECONV0 % NUM_PE_LAYER_DECONV0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_DECONV0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_DECONV0) < OUT_CHN_LAYER_DECONV0 ? ((pe_loop+1)*NUM_PE_LAYER_DECONV0) : OUT_CHN_LAYER_DECONV0;
        for(int row = 0; row < 70; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                #pragma HLS UNROLL
                for(int idx = 0; idx < 6; idx++){
                    #pragma HLS PIPELINE II=1
                    bool is_pad = false;
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = get_next_tconv_7(row, idx, &tile_in[ch], &is_pad); // Read from actual tile
                    else             next_data = get_next_tconv_7(row, idx, &inbuf[ch],   &is_pad); // Read from input buffer
                    slider[ch][idx] = next_data;
                    if((!is_pad) && (pe_loop != (num_pe_loops - 1))) inbuf[ch].write(next_data); // save for later 
                }
            }

            // Go across the row
            for(int col = 6; col < 70; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                    #pragma HLS UNROLL

                    bool is_pad = false;
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = get_next_tconv_7(row, col, &tile_in[ch], &is_pad); // Read from actual tile
                    else             next_data = get_next_tconv_7(row, col, &inbuf[ch],   &is_pad); // Read from input buffer

                    slider[ch][6] = next_data;
                    if((!is_pad) && (pe_loop != (num_pe_loops - 1))) inbuf[ch].write(next_data);
                }

                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    int pe_idx = filter % NUM_PE_LAYER_DECONV0;
                    fixed_4_8_t mac0 = 0.0;
                    fixed_4_8_t mac1 = 0.0;
                    fixed_4_8_t mac2 = 0.0;
                    fixed_4_8_t mac3 = 0.0;
                    fixed_4_8_t mac4 = 0.0;
                    fixed_4_8_t mac5 = 0.0;
                    fixed_4_8_t mac6 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum, row3_psum, row4_psum, row5_psum, row6_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                        #pragma HLS UNROLL
                        if(row < 64)             mac0 += perform_mac7(weights_layer_deconv0[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 65) mac1 += perform_mac7(weights_layer_deconv0[filter][ch][1], slider[ch]);
                        if(row >= 2 && row < 66) mac2 += perform_mac7(weights_layer_deconv0[filter][ch][2], slider[ch]);
                        if(row >= 3 && row < 67) mac3 += perform_mac7(weights_layer_deconv0[filter][ch][3], slider[ch]);
                        if(row >= 4 && row < 68) mac4 += perform_mac7(weights_layer_deconv0[filter][ch][4], slider[ch]);
                        if(row >= 5 && row < 69) mac5 += perform_mac7(weights_layer_deconv0[filter][ch][5], slider[ch]);
                        if(row >= 6)             mac6 += perform_mac7(weights_layer_deconv0[filter][ch][6], slider[ch]);
                    }

                    if(row < 64){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 65) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2 && row < 66) {
                        row2_psum = psum2[pe_idx].read();
                        psum3[pe_idx].write(row2_psum + mac2);
                    }
                    if(row >= 3 && row < 67) {
                        row3_psum = psum3[pe_idx].read();
                        psum4[pe_idx].write(row3_psum + mac3);
                    }
                    if(row >= 4 && row < 68) {
                        row4_psum = psum4[pe_idx].read();
                        psum5[pe_idx].write(row4_psum + mac4);
                    }
                    if(row >= 5 && row < 69) {
                        row5_psum = psum5[pe_idx].read();
                        psum6[pe_idx].write(row5_psum + mac5);
                    }
                    if(row >= 6) {
                        row6_psum = psum6[pe_idx].read();
                        fixed_4_8_t pre_activation = row6_psum + mac6 + conv_deconv0_bias[filter];
                        map_out[filter].write(pre_activation);
                    }
                } // For every filter 

                for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                   #pragma HLS UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                   slider[ch][2] = slider[ch][3];
                   slider[ch][3] = slider[ch][4];
                   slider[ch][4] = slider[ch][5];
                   slider[ch][5] = slider[ch][6];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE
}

///////////////////////////////// End of auto-generated conv code /////////////////////////////////


void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream){
	#pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return

//#pragma HLS bind_storage variable=weights_layer_deconv0 type=RAM_2P impl=BRAM
// #pragma HLS bind_storage variable=weights_layer_map0 type=RAM_2P impl=BRAM
// #pragma HLS bind_storage variable=weights_layer_map2 type=RAM_2P impl=BRAM
// #pragma HLS bind_storage variable=weights_layer_map4 type=RAM_2P impl=BRAM
// #pragma HLS bind_storage variable=weights_layer_map6 type=RAM_2P impl=BRAM

	
	// 1. Load the image into 3 separate streams (FIFOs), converting to fixed_4_8_t on the fly
	ch_stream_t tile_in[3];
	ch_stream_t map_extraction[OUT_CHN_LAYER_FEATURE_EXTRACTION0];
	ch_stream_t map_shrink[OUT_CHN_LAYER_SHRINK0];
	ch_stream_t map_map0[OUT_CHN_LAYER_MAP0];
	ch_stream_t map_map2[OUT_CHN_LAYER_MAP2];
	ch_stream_t map_map4[OUT_CHN_LAYER_MAP4];
	ch_stream_t map_expand0[OUT_CHN_LAYER_EXPAND0];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX * 2 * 2> map_upscaled[OUT_CHN_LAYER_DECONV0];

    #pragma HLS RESOURCE variable=tile_in core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_extraction core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_shrink core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map0 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map2 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map4 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_expand0 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_upscaled core=FIFO_BRAM

        
    #pragma HLS DATAFLOW
    prep_tile(in_stream, tile_in);
	conv_feature_extraction0(tile_in, map_extraction);
	conv_shrink0(map_extraction, map_shrink);
	conv_map0(map_shrink, map_map0);
	conv_map2(map_map0, map_map2);
	conv_map4(map_map2, map_map4);
    conv_expand0(map_map4, map_expand0);
    conv_deconv0(map_expand0, map_upscaled);
    stream_samples_out(map_upscaled, out_stream);

	//  for(int i = 0; i < OUT_CHN_LAYER_EXPAND0; i++){
	//  	printf("INFO [conv2d] Feature map %d:\n", i);
	//  	for (int col = 0; col < 28*28; col++){
	//  		printf("%.8f \n", map_expand0[i].read().to_float());
	//  	}
	//  	printf("\n");
	//  }

    // for(int i = 0; i < OUT_CHN_LAYER_FEATURE_EXTRACTION0; i++){
	// 	printf("INFO [conv2d] Feature map %d:\n", i);
	// 	for (int col = 0; col < 28*28; col++){
	// 		printf("%.8f \n", map_extraction[i].read().to_float());
	// 	}
	// 	printf("\n");
	// }

    // for(int i = 0; i < OUT_CHN_LAYER_DECONV0; i++){
	// 	printf("INFO [conv2d] Feature map %d:\n", i);
	// 	for (int col = 0; col < 28*28*2*2; col++){
	// 		printf("%.8f \n", map_upscaled[i].read().to_float());
	// 	}
	// 	printf("\n");
	// }
}

