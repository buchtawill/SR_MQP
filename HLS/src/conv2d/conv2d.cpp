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

void stream_samples_out(hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &data_fifo,
						hls::stream<axis_t> &out_stream){
	
	axis_t tmp_stream;
	int out_ptr = 0;
	int fmap_ptr_abs = 0;
	for(out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
		tmp_stream.last = (out_ptr == (STREAM_BEATS_PER_TILE - 1));

		// 128 bits = 16 bytes = 16 bits of keep
		// Max width is 128 bits
		// Fewer bytes will truncate
		tmp_stream.keep = 0xffff;
		tmp_stream.strb = 0xffff;
		tmp_stream.data = data_fifo.read();

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

/*
void fill_output_fifo(fixed_4_8_t img_in[INPUT_HEIGHT_PIX][INPUT_WIDTH_PIX][BYTES_PER_PIXEL],
					  hls::stream<stream_data_t, STREAM_BEATS_PER_TILE> &output_fifo){

	int fmap_ptr_abs = 0;
	WRITE_OUTPUT:for(int out_ptr = 0; out_ptr < STREAM_BEATS_PER_TILE; out_ptr++){
#pragma HLS PIPELINE II=20
		stream_data_t tmp_data;
		for(int i = 0; i < BYTES_PER_TRANSFER; i++){
			// #pragma HLS UNROLL factor=BYTES_PER_TRANSFER

			int row_idx = (fmap_ptr_abs / (INPUT_WIDTH_PIX * BYTES_PER_PIXEL));
			int col_idx = (fmap_ptr_abs / BYTES_PER_PIXEL) % INPUT_WIDTH_PIX;
			int chn_idx =  fmap_ptr_abs % BYTES_PER_PIXEL;

			fixed_4_8_t tmp_val = img_in[row_idx][col_idx][chn_idx];
			tmp_data.range(8 * (i + 1) - 1, 8 * i) = tmp_val.range(7, 0);

			// Cast to fixed 9_8, multiply by 256, cast to uint8_t
			// fixed_9_8_t tmp_val = img_in[row_idx][col_idx][chn_idx];
			// tmp_val = tmp_val << 8;
			// uint8_t bits = (uint8_t)tmp_val.to_uint();

			fmap_ptr_abs++;
		}
		output_fifo.write(tmp_data);
	}
}*/

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
		#pragma HLS UNROLL
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
fixed_4_8_t get_next_tconv(int row, int col, ch_stream_t *input, bool *zero){
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
    #pragma HLS STREAM variable=psum1 depth=28
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=28
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum3 depth=28
    #pragma HLS RESOURCE variable=psum3 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum4 depth=28
    #pragma HLS RESOURCE variable=psum4 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_FEATURE_EXTRACTION0 / NUM_PE_LAYER_FEATURE_EXTRACTION0;
    if((OUT_CHN_LAYER_FEATURE_EXTRACTION0 % NUM_PE_LAYER_FEATURE_EXTRACTION0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_FEATURE_EXTRACTION0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_FEATURE_EXTRACTION0) < OUT_CHN_LAYER_FEATURE_EXTRACTION0 ? ((pe_loop+1)*NUM_PE_LAYER_FEATURE_EXTRACTION0) : OUT_CHN_LAYER_FEATURE_EXTRACTION0;
        for(int row = 0; row < 32; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 4; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 2) || (row >= 30) || (idx < 2)) slider[ch][idx] = 0;
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
            for(int col = 4; col < 32; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                    #pragma HLS UNROLL

                    if((row < 2) || (row >= 30) || (col >= 30)) slider[ch][4] = 0;
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
                        if(row < 28)             mac0 += perform_mac5(weights_layer_feature_extraction0[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 29) mac1 += perform_mac5(weights_layer_feature_extraction0[filter][ch][1], slider[ch]);
                        if(row >= 2 && row < 30) mac2 += perform_mac5(weights_layer_feature_extraction0[filter][ch][2], slider[ch]);
                        if(row >= 3 && row < 31) mac3 += perform_mac5(weights_layer_feature_extraction0[filter][ch][3], slider[ch]);
                        if(row >= 4)             mac4 += perform_mac5(weights_layer_feature_extraction0[filter][ch][4], slider[ch]);
                    }

                    if(row < 28){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 29) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2 && row < 30) {
                        row2_psum = psum2[pe_idx].read();
                        psum3[pe_idx].write(row2_psum + mac2);
                    }
                    if(row >= 3 && row < 31) {
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
                   #pragma HLS_UNROLL
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
        for(int row = 0; row < 28; row++){

            // Go across the row
            for(int col = 0; col < 28; col++){
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
    #pragma HLS STREAM variable=psum1 depth=28
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=28
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP0 / NUM_PE_LAYER_MAP0;
    if((OUT_CHN_LAYER_MAP0 % NUM_PE_LAYER_MAP0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP0) < OUT_CHN_LAYER_MAP0 ? ((pe_loop+1)*NUM_PE_LAYER_MAP0) : OUT_CHN_LAYER_MAP0;
        for(int row = 0; row < 30; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 29) || (idx < 1)) slider[ch][idx] = 0;
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
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 29) || (col >= 29)) slider[ch][2] = 0;
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
                        if(row < 28)             mac0 += perform_mac3(weights_layer_map0[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 29) mac1 += perform_mac3(weights_layer_map0[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map0[filter][ch][2], slider[ch]);
                    }

                    if(row < 28){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 29) {
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
                   #pragma HLS_UNROLL
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
    #pragma HLS STREAM variable=psum1 depth=28
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=28
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP2 / NUM_PE_LAYER_MAP2;
    if((OUT_CHN_LAYER_MAP2 % NUM_PE_LAYER_MAP2) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP2);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP2) < OUT_CHN_LAYER_MAP2 ? ((pe_loop+1)*NUM_PE_LAYER_MAP2) : OUT_CHN_LAYER_MAP2;
        for(int row = 0; row < 30; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 29) || (idx < 1)) slider[ch][idx] = 0;
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
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 29) || (col >= 29)) slider[ch][2] = 0;
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
                        if(row < 28)             mac0 += perform_mac3(weights_layer_map2[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 29) mac1 += perform_mac3(weights_layer_map2[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map2[filter][ch][2], slider[ch]);
                    }

                    if(row < 28){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 29) {
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
                   #pragma HLS_UNROLL
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
    #pragma HLS STREAM variable=psum1 depth=28
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=28
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP4 / NUM_PE_LAYER_MAP4;
    if((OUT_CHN_LAYER_MAP4 % NUM_PE_LAYER_MAP4) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP4);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP4) < OUT_CHN_LAYER_MAP4 ? ((pe_loop+1)*NUM_PE_LAYER_MAP4) : OUT_CHN_LAYER_MAP4;
        for(int row = 0; row < 30; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 29) || (idx < 1)) slider[ch][idx] = 0;
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
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 29) || (col >= 29)) slider[ch][2] = 0;
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
                        if(row < 28)             mac0 += perform_mac3(weights_layer_map4[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 29) mac1 += perform_mac3(weights_layer_map4[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map4[filter][ch][2], slider[ch]);
                    }

                    if(row < 28){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 29) {
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
                   #pragma HLS_UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE
}

void conv_map6(ch_stream_t tile_in[IN_CHN_LAYER_MAP6], ch_stream_t map_out[OUT_CHN_LAYER_MAP6]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_MAP6][3];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_MAP6];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP6];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP6];
    #pragma HLS STREAM variable=psum1 depth=28
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=28
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_MAP6 / NUM_PE_LAYER_MAP6;
    if((OUT_CHN_LAYER_MAP6 % NUM_PE_LAYER_MAP6) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_MAP6);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_MAP6) < OUT_CHN_LAYER_MAP6 ? ((pe_loop+1)*NUM_PE_LAYER_MAP6) : OUT_CHN_LAYER_MAP6;
        for(int row = 0; row < 30; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_MAP6; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 2; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 1) || (row >= 29) || (idx < 1)) slider[ch][idx] = 0;
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
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_MAP6; ch++){
                    #pragma HLS UNROLL

                    if((row < 1) || (row >= 29) || (col >= 29)) slider[ch][2] = 0;
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
                    int pe_idx = filter % NUM_PE_LAYER_MAP6;
                    fixed_4_8_t mac0 = 0.0;
                    fixed_4_8_t mac1 = 0.0;
                    fixed_4_8_t mac2 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_MAP6; ch++){
                        #pragma HLS UNROLL
                        if(row < 28)             mac0 += perform_mac3(weights_layer_map6[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 29) mac1 += perform_mac3(weights_layer_map6[filter][ch][1], slider[ch]);
                        if(row >= 2)             mac2 += perform_mac3(weights_layer_map6[filter][ch][2], slider[ch]);
                    }

                    if(row < 28){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 29) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2) {
                        row2_psum = psum2[pe_idx].read();
                        fixed_4_8_t pre_activation = row2_psum + mac2 + conv_map6_bias[filter];
                        map_out[filter].write(prelu(conv_map6_prelu[filter], pre_activation));
                    }
                } // For every filter 

               for(int ch = 0; ch < IN_CHN_LAYER_MAP6; ch++){
                   #pragma HLS_UNROLL
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
        for(int row = 0; row < 28; row++){

            // Go across the row
            for(int col = 0; col < 28; col++){
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
    fixed_4_8_t slider[IN_CHN_LAYER_DECONV0][9];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_DECONV0];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum1[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum2[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum3[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum4[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum5[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum6[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum7[NUM_PE_LAYER_DECONV0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*2> psum8[NUM_PE_LAYER_DECONV0];
    #pragma HLS STREAM variable=psum1 depth=56
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=56
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum3 depth=56
    #pragma HLS RESOURCE variable=psum3 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum4 depth=56
    #pragma HLS RESOURCE variable=psum4 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum5 depth=56
    #pragma HLS RESOURCE variable=psum5 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum6 depth=56
    #pragma HLS RESOURCE variable=psum6 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum7 depth=56
    #pragma HLS RESOURCE variable=psum7 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum8 depth=56
    #pragma HLS RESOURCE variable=psum8 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_DECONV0 / NUM_PE_LAYER_DECONV0;
    if((OUT_CHN_LAYER_DECONV0 % NUM_PE_LAYER_DECONV0) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_DECONV0);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_DECONV0) < OUT_CHN_LAYER_DECONV0 ? ((pe_loop+1)*NUM_PE_LAYER_DECONV0) : OUT_CHN_LAYER_DECONV0;
        printf("INFO [conv_deconv0] PE loop %d\n", pe_loop);
        for(int row = 0; row < 64; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 8; idx++){
                    #pragma HLS PIPELINE II=1
                    bool is_pad = false;
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = get_next_tconv(row, idx, &tile_in[ch], &is_pad); // Read from actual tile
                    else             next_data = get_next_tconv(row, idx, &inbuf[ch],   &is_pad); // Read from input buffer
                    slider[ch][idx] = next_data;
                    if((!is_pad) && (pe_loop != (num_pe_loops - 1))) inbuf[ch].write(next_data); // save for later 
                }
            }

            // Go across the row
            for(int col = 8; col < 64; col++){
                #pragma HLS PIPELINE II=1
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                    #pragma HLS UNROLL

                    bool is_pad = false;
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = get_next_tconv(row, col, &tile_in[ch], &is_pad); // Read from actual tile
                    else             next_data = get_next_tconv(row, col, &inbuf[ch],   &is_pad); // Read from input buffer

                    slider[ch][8] = next_data;
                    if((!is_pad) && (pe_loop != (num_pe_loops - 1))) inbuf[ch].write(next_data);
                }

                // if(row == 4){
                //     printf("First slider values:\n");
                //     for(int i = 0; i < 9; i++){
                //         printf("%8.6f\n", slider[0][i].to_float());
                //     }
                //     return;
                // }
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
                    fixed_4_8_t mac7 = 0.0;
                    fixed_4_8_t mac8 = 0.0;
                    fixed_4_8_t row1_psum, row2_psum, row3_psum, row4_psum, row5_psum, row6_psum, row7_psum, row8_psum;

                    for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                        #pragma HLS UNROLL
                        if(row < 56)             mac0 += perform_mac9(weights_layer_deconv0[filter][ch][0], slider[ch]);
                        if(row >= 1 && row < 57) mac1 += perform_mac9(weights_layer_deconv0[filter][ch][1], slider[ch]);
                        if(row >= 2 && row < 58) mac2 += perform_mac9(weights_layer_deconv0[filter][ch][2], slider[ch]);
                        if(row >= 3 && row < 59) mac3 += perform_mac9(weights_layer_deconv0[filter][ch][3], slider[ch]);
                        if(row >= 4 && row < 60) mac4 += perform_mac9(weights_layer_deconv0[filter][ch][4], slider[ch]);
                        if(row >= 5 && row < 61) mac5 += perform_mac9(weights_layer_deconv0[filter][ch][5], slider[ch]);
                        if(row >= 6 && row < 62) mac6 += perform_mac9(weights_layer_deconv0[filter][ch][6], slider[ch]);
                        if(row >= 7 && row < 63) mac7 += perform_mac9(weights_layer_deconv0[filter][ch][7], slider[ch]);
                        if(row >= 8)             mac8 += perform_mac9(weights_layer_deconv0[filter][ch][8], slider[ch]);
                    }

                    // if(row == 4){
                    //     printf("%8.6f\n", mac0.to_float());
                    //     printf("%8.6f\n", mac1.to_float());
                    //     printf("%8.6f\n", mac2.to_float());
                    //     printf("%8.6f\n", mac3.to_float());
                    //     printf("%8.6f\n", mac4.to_float());
                    //     return;
                    // }

                    if(row < 56){
                        psum1[pe_idx].write(mac0);
                    }
                    if(row >= 1 && row < 57) {
                        row1_psum = psum1[pe_idx].read();
                        psum2[pe_idx].write(row1_psum + mac1);
                    }
                    if(row >= 2 && row < 58) {
                        row2_psum = psum2[pe_idx].read();
                        psum3[pe_idx].write(row2_psum + mac2);
                    }
                    if(row >= 3 && row < 59) {
                        row3_psum = psum3[pe_idx].read();
                        psum4[pe_idx].write(row3_psum + mac3);
                    }
                    if(row >= 4 && row < 60) {
                        row4_psum = psum4[pe_idx].read();
                        psum5[pe_idx].write(row4_psum + mac4);
                    }
                    if(row >= 5 && row < 61) {
                        row5_psum = psum5[pe_idx].read();
                        psum6[pe_idx].write(row5_psum + mac5);
                    }
                    if(row >= 6 && row < 62) {
                        row6_psum = psum6[pe_idx].read();
                        psum7[pe_idx].write(row6_psum + mac6);
                    }
                    if(row >= 7 && row < 63) {
                        row7_psum = psum7[pe_idx].read();
                        psum8[pe_idx].write(row7_psum + mac7);
                    }
                    if(row >= 8) {
                        row8_psum = psum8[pe_idx].read();
                        fixed_4_8_t pre_activation = row8_psum + mac8 + conv_deconv0_bias[filter];
                        map_out[filter].write(pre_activation);
                    }
                } // For every filter 

                for(int ch = 0; ch < IN_CHN_LAYER_DECONV0; ch++){
                   #pragma HLS_UNROLL
                   slider[ch][0] = slider[ch][1];
                   slider[ch][1] = slider[ch][2];
                   slider[ch][2] = slider[ch][3];
                   slider[ch][3] = slider[ch][4];
                   slider[ch][4] = slider[ch][5];
                   slider[ch][5] = slider[ch][6];
                   slider[ch][6] = slider[ch][7];
                   slider[ch][7] = slider[ch][8];
                }
            } // For every column 
        } // For every row
    } // For number of times thru PE

    printf("INFO [conv_deconv0] Finished convolution. psum1 size: %d\n", psum1[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum2 size: %d\n", psum2[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum3 size: %d\n", psum3[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum4 size: %d\n", psum4[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum5 size: %d\n", psum5[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum6 size: %d\n", psum6[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum7 size: %d\n", psum7[0].size());
    printf("INFO [conv_deconv0] Finished convolution. psum8 size: %d\n", psum8[0].size());
    printf("INFO [conv_deconv0] Finished convolution. inbuf size: %d\n", inbuf[0].size());
    printf("INFO [conv_deconv0] Finished convolution. tilin size: %d\n", tile_in[0].size());
}

///////////////////////////////// End of auto-generated conv code /////////////////////////////////


void conv2d_top(hls::stream<axis_t> &in_stream, hls::stream<axis_t> &out_stream){
	#pragma HLS INTERFACE axis port=in_stream
    #pragma HLS INTERFACE axis port=out_stream
    #pragma HLS INTERFACE ap_ctrl_none port=return
	
	// 1. Load the image into 3 separate streams (FIFOs), converting to fixed_4_8_t on the fly
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> tile_in[3];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_extraction[OUT_CHN_LAYER_FEATURE_EXTRACTION0];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_shrink[OUT_CHN_LAYER_SHRINK0];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_map0[OUT_CHN_LAYER_MAP0];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_map2[OUT_CHN_LAYER_MAP2];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_map4[OUT_CHN_LAYER_MAP4];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_map6[OUT_CHN_LAYER_MAP6];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX> map_expand0[OUT_CHN_LAYER_EXPAND0];
	hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX*INPUT_HEIGHT_PIX * 2 * 2> map_upscaled[OUT_CHN_LAYER_DECONV0];
    #pragma HLS RESOURCE variable=tile_in core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_extraction core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_shrink core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map0 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map2 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map4 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_map6 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_expand0 core=FIFO_BRAM
    #pragma HLS RESOURCE variable=map_upscaled core=FIFO_BRAM

//	#pragma HLS BIND_STORAGE variable=tile_in type=bram
	// #pragma HLS array_partition variable=tile_in dim=0 type=complete

	#pragma HLS DATAFLOW
	prep_tile(in_stream, tile_in);
    // for(int i = 0; i < 784; i++){
    //     printf("%f,\n", tile_in[2].read());
    // }
    // return;

	conv_feature_extraction0(tile_in, map_extraction);
	conv_shrink0(map_extraction, map_shrink);
	conv_map0(map_shrink, map_map0);
	conv_map2(map_map0, map_map2);
	conv_map4(map_map2, map_map4);
	conv_map6(map_map4, map_map6);
	conv_expand0(map_map6, map_expand0);
    conv_deconv0(map_expand0, map_upscaled);

	// for(int i = 0; i < OUT_CHN_LAYER_EXPAND0; i++){
	// 	printf("INFO [conv2d] Feature map %d:\n", i);
	// 	for (int col = 0; col < 28*28; col++){
	// 		printf("%.8f \n", map_expand0[i].read().to_float());
	// 	}
	// 	printf("\n");
	// }

    for(int i = 0; i < OUT_CHN_LAYER_DECONV0; i++){
		printf("INFO [conv2d] Feature map %d:\n", i);
		for (int col = 0; col < 28*28*2*2; col++){
			printf("%.8f \n", map_upscaled[i].read().to_float());
		}
		printf("\n");
	}
}

