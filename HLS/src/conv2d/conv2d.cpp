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
		#pragma HLS PIPELINE II=5
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

			// Divide by 256, cast to 12 bit fixed, write to FIFO
			tile_in[0].write((fixed_4_8_t)(r0 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r1 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r2 >> 8)); 
			tile_in[0].write((fixed_4_8_t)(r3 >> 8));

			tile_in[1].write((fixed_4_8_t)(g0 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g1 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g2 >> 8)); 
			tile_in[1].write((fixed_4_8_t)(g3 >> 8));

			tile_in[2].write((fixed_4_8_t)(b2 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b1 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b2 >> 8)); 
			tile_in[2].write((fixed_4_8_t)(b3 >> 8));
		}
	}
}

fixed_4_8_t perform_mac5(const fixed_4_8_t weights[5], fixed_4_8_t slider[5]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	DO_MAC5:
	for(int w = 0; w < 5; w++){
		#pragma HLS UNROLL
		sum += weights[w] * slider[w];
	}
	return sum;
}

fixed_4_8_t perform_mac3(const fixed_4_8_t weights[3], fixed_4_8_t slider[3]){
	#pragma HLS INLINE
	fixed_4_8_t sum = 0.0;
	DO_MAC5:
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

/*
void conv_extraction(ch_stream_t tile_in[IN_CHN_LAYER_EXTRACTION], ch_stream_t map_out[OUT_CHN_LAYER_EXTRACTION]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_EXTRACTION][5];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_EXTRACTION];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_EXTRACTION][IN_CHN_LAYER_EXTRACTION];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_EXTRACTION][IN_CHN_LAYER_EXTRACTION];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum3[NUM_PE_LAYER_EXTRACTION][IN_CHN_LAYER_EXTRACTION];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum4[NUM_PE_LAYER_EXTRACTION][IN_CHN_LAYER_EXTRACTION];
    #pragma HLS STREAM variable=psum1 depth=28
    #pragma HLS RESOURCE variable=psum1 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum2 depth=28
    #pragma HLS RESOURCE variable=psum2 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum3 depth=28
    #pragma HLS RESOURCE variable=psum3 core=FIFO_BRAM
    #pragma HLS STREAM variable=psum4 depth=28
    #pragma HLS RESOURCE variable=psum4 core=FIFO_BRAM

    int num_pe_loops = OUT_CHN_LAYER_EXTRACTION / NUM_PE_LAYER_EXTRACTION;
    if((OUT_CHN_LAYER_EXTRACTION % NUM_PE_LAYER_EXTRACTION) != 0) num_pe_loops++;
    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_EXTRACTION);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_EXTRACTION) < OUT_CHN_LAYER_EXTRACTION ? ((pe_loop+1)*NUM_PE_LAYER_EXTRACTION) : OUT_CHN_LAYER_EXTRACTION;
        for(int row = 0; row < 32; row++){

            // Prep the slider
            for(int ch = 0; ch < IN_CHN_LAYER_EXTRACTION; ch++){
                #pragma HLS UROLL
                for(int idx = 0; idx < 4; idx++){
                    #pragma HLS PIPELINE II=1
                    if((row < 2) || (row >= 30) || (idx < 2)) slider[ch][idx] = 0;
                    else{
                        fixed_4_8_t next_data;
                        if(pe_loop == 0) next_data = tile_in[ch].read();
                        else             next_data = inbuf[ch].read();

                        slider[ch][4] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 4; col < 32; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_EXTRACTION];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_EXTRACTION; ch++){
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
                    for(int ch = 0; ch < IN_CHN_LAYER_EXTRACTION; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2, mac3, mac4;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum, row3_psum, row4_psum;
                        if(row < 28){
                            mac0 = perform_mac5(weights_layer_extraction[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_EXTRACTION][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_EXTRACTION][ch].read();
                            mac1 = perform_mac5(weights_layer_extraction[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_EXTRACTION][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2 && row < 30) {
                            row2_psum = psum2[filter % NUM_PE_LAYER_EXTRACTION][ch].read();
                            mac2 = perform_mac5(weights_layer_extraction[filter][ch][2], slider[ch]);
                            psum3[filter % NUM_PE_LAYER_EXTRACTION][ch].write(row2_psum + mac2);
                        }
                        if(row >= 3 && row < 31) {
                            row3_psum = psum3[filter % NUM_PE_LAYER_EXTRACTION][ch].read();
                            mac3 = perform_mac5(weights_layer_extraction[filter][ch][3], slider[ch]);
                            psum4[filter % NUM_PE_LAYER_EXTRACTION][ch].write(row3_psum + mac3);
                        }
                        if(row >= 4){
                            row4_psum = psum4[filter % NUM_PE_LAYER_EXTRACTION][ch].read();
                            mac4 = perform_mac5(weights_layer_extraction[filter][ch][4], slider[ch]);
                            fixed_4_8_t pre_activation = row4_psum + mac4;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 4) map_out[filter].write(prelu(conv_extraction_prelu[filter], \
                                                            (final_sum[filter] + conv_extraction_bias[filter])));
                } // For every filter
                for(int ch = 0; ch < IN_CHN_LAYER_EXTRACTION; ch++){
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
void conv_map0(ch_stream_t tile_in[IN_CHN_LAYER_MAP0], ch_stream_t map_out[OUT_CHN_LAYER_MAP0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_MAP0][3];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_MAP0];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP0][IN_CHN_LAYER_MAP0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP0][IN_CHN_LAYER_MAP0];
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

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_MAP0];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

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
                    for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum;
                        if(row < 28){
                            mac0 = perform_mac3(weights_layer_map0[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_MAP0][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_MAP0][ch].read();
                            mac1 = perform_mac3(weights_layer_map0[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_MAP0][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2){
                            row2_psum = psum2[filter % NUM_PE_LAYER_MAP0][ch].read();
                            mac2 = perform_mac3(weights_layer_map0[filter][ch][2], slider[ch]);
                            fixed_4_8_t pre_activation = row2_psum + mac2;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 2) map_out[filter].write(prelu(conv_map0_prelu[filter], \
                                                            (final_sum[filter] + conv_map0_bias[filter])));
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
void conv_shrink_huma(ch_stream_t tile_in[IN_CHN_LAYER_SHRINK], ch_stream_t map_out[OUT_CHN_LAYER_SHRINK]){

	fixed_4_8_t slider[IN_CHN_LAYER_SHRINK];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete

    ch_stream_t inbuf[IN_CHN_LAYER_SHRINK];

    int num_pe_loops = OUT_CHN_LAYER_EXTRACTION / NUM_PE_LAYER_EXTRACTION;
    if((OUT_CHN_LAYER_EXTRACTION % NUM_PE_LAYER_EXTRACTION) != 0) num_pe_loops++;

    for(int pe_loop = 0; pe_loop < num_pe_loops; pe_loop++){
        // WARNING: if number fmap % num_pe != 0, utilization explodes!!
        int low_filter = (pe_loop*NUM_PE_LAYER_EXTRACTION);
        int high_filter = ((pe_loop+1)*NUM_PE_LAYER_EXTRACTION) < OUT_CHN_LAYER_EXTRACTION ? ((pe_loop+1)*NUM_PE_LAYER_EXTRACTION) : OUT_CHN_LAYER_EXTRACTION;
        for(int row = 0; row < 28; row++){
            // Go across the row
            for(int col = 0; col < 28; col++){
                #pragma HLS PIPELINE II=1
    
                // Read the next value into the slider
                for(int ch = 0; ch < IN_CHN_LAYER_SHRINK; ch++){
                    #pragma HLS UNROLL
                    fixed_4_8_t next_data;
                    if(pe_loop == 0) next_data = tile_in[ch].read();
                    else             next_data = inbuf[ch].read();

                    slider[ch] = next_data;
                    if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                }

                for(int filter = low_filter; filter < high_filter; filter++){
					fixed_4_8_t mac = 0.0;
                    for(int ch = 0; ch < IN_CHN_LAYER_SHRINK; ch++){
                        #pragma HLS UNROLL

						// Multiply-accumulate across all channels of the 1x1
						mac += slider[ch] * weights_layer_shrink0[filter][ch];
                    }

                    map_out[filter].write(prelu(conv_shrink0_prelu[filter], (mac + conv_shrink0_bias[filter])));
                } // For every filter
            } // For every column
        } // For every row
    } // For number of times thru PE
}
*/

////////////////////////////////// Auto-generated code goes here //////////////////////////////////

void conv_feature_extraction0(ch_stream_t tile_in[IN_CHN_LAYER_FEATURE_EXTRACTION0], ch_stream_t map_out[OUT_CHN_LAYER_FEATURE_EXTRACTION0]){
    // NOTE: This function was auto generated. Do not edit here, edit FSRCNN/conv_ideal.py
    fixed_4_8_t slider[IN_CHN_LAYER_FEATURE_EXTRACTION0][5];
    #pragma HLS ARRAY_PARTITION variable=slider dim=0 type=complete
    ch_stream_t inbuf[IN_CHN_LAYER_FEATURE_EXTRACTION0];

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_FEATURE_EXTRACTION0][IN_CHN_LAYER_FEATURE_EXTRACTION0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_FEATURE_EXTRACTION0][IN_CHN_LAYER_FEATURE_EXTRACTION0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum3[NUM_PE_LAYER_FEATURE_EXTRACTION0][IN_CHN_LAYER_FEATURE_EXTRACTION0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum4[NUM_PE_LAYER_FEATURE_EXTRACTION0][IN_CHN_LAYER_FEATURE_EXTRACTION0];
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

                        slider[ch][4] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 4; col < 32; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_FEATURE_EXTRACTION0];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

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
                    for(int ch = 0; ch < IN_CHN_LAYER_FEATURE_EXTRACTION0; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2, mac3, mac4;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum, row3_psum, row4_psum;
                        if(row < 28){
                            mac0 = perform_mac5(weights_layer_feature_extraction0[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].read();
                            mac1 = perform_mac5(weights_layer_feature_extraction0[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2 && row < 30) {
                            row2_psum = psum2[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].read();
                            mac2 = perform_mac5(weights_layer_feature_extraction0[filter][ch][2], slider[ch]);
                            psum3[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].write(row2_psum + mac2);
                        }
                        if(row >= 3 && row < 31) {
                            row3_psum = psum3[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].read();
                            mac3 = perform_mac5(weights_layer_feature_extraction0[filter][ch][3], slider[ch]);
                            psum4[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].write(row3_psum + mac3);
                        }
                        if(row >= 4){
                            row4_psum = psum4[filter % NUM_PE_LAYER_FEATURE_EXTRACTION0][ch].read();
                            mac4 = perform_mac5(weights_layer_feature_extraction0[filter][ch][4], slider[ch]);
                            fixed_4_8_t pre_activation = row4_psum + mac4;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 4) map_out[filter].write(prelu(conv_feature_extraction0_prelu[filter], \
                                                            (final_sum[filter] + conv_feature_extraction0_bias[filter])));
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

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP0][IN_CHN_LAYER_MAP0];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP0][IN_CHN_LAYER_MAP0];
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

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_MAP0];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

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
                    for(int ch = 0; ch < IN_CHN_LAYER_MAP0; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum;
                        if(row < 28){
                            mac0 = perform_mac3(weights_layer_map0[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_MAP0][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_MAP0][ch].read();
                            mac1 = perform_mac3(weights_layer_map0[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_MAP0][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2){
                            row2_psum = psum2[filter % NUM_PE_LAYER_MAP0][ch].read();
                            mac2 = perform_mac3(weights_layer_map0[filter][ch][2], slider[ch]);
                            fixed_4_8_t pre_activation = row2_psum + mac2;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 2) map_out[filter].write(prelu(conv_map0_prelu[filter], \
                                                            (final_sum[filter] + conv_map0_bias[filter])));
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

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP2][IN_CHN_LAYER_MAP2];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP2][IN_CHN_LAYER_MAP2];
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

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_MAP2];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

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
                    for(int ch = 0; ch < IN_CHN_LAYER_MAP2; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum;
                        if(row < 28){
                            mac0 = perform_mac3(weights_layer_map2[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_MAP2][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_MAP2][ch].read();
                            mac1 = perform_mac3(weights_layer_map2[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_MAP2][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2){
                            row2_psum = psum2[filter % NUM_PE_LAYER_MAP2][ch].read();
                            mac2 = perform_mac3(weights_layer_map2[filter][ch][2], slider[ch]);
                            fixed_4_8_t pre_activation = row2_psum + mac2;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 2) map_out[filter].write(prelu(conv_map2_prelu[filter], \
                                                            (final_sum[filter] + conv_map2_bias[filter])));
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

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP4][IN_CHN_LAYER_MAP4];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP4][IN_CHN_LAYER_MAP4];
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

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_MAP4];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

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
                    for(int ch = 0; ch < IN_CHN_LAYER_MAP4; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum;
                        if(row < 28){
                            mac0 = perform_mac3(weights_layer_map4[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_MAP4][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_MAP4][ch].read();
                            mac1 = perform_mac3(weights_layer_map4[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_MAP4][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2){
                            row2_psum = psum2[filter % NUM_PE_LAYER_MAP4][ch].read();
                            mac2 = perform_mac3(weights_layer_map4[filter][ch][2], slider[ch]);
                            fixed_4_8_t pre_activation = row2_psum + mac2;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 2) map_out[filter].write(prelu(conv_map4_prelu[filter], \
                                                            (final_sum[filter] + conv_map4_bias[filter])));
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

    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum1[NUM_PE_LAYER_MAP6][IN_CHN_LAYER_MAP6];
    hls::stream<fixed_4_8_t, INPUT_WIDTH_PIX> psum2[NUM_PE_LAYER_MAP6][IN_CHN_LAYER_MAP6];
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

                        slider[ch][2] = next_data;
                        if(pe_loop != (num_pe_loops - 1)) inbuf[ch].write(next_data);
                    }
                }
            }

            // Go across the row
            for(int col = 2; col < 30; col++){
                #pragma HLS PIPELINE II=1
                fixed_4_8_t final_sum[OUT_CHN_LAYER_MAP6];
                #pragma HLS array_partition variable=final_sum dim=0 type=complete
                for(int filter = low_filter; filter < high_filter; filter++){
                    #pragma HLS UNROLL
                    final_sum[filter] = 0.0;
                }

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
                    for(int ch = 0; ch < IN_CHN_LAYER_MAP6; ch++){
                        #pragma HLS UNROLL
                        fixed_4_8_t mac0, mac1, mac2;
                        fixed_4_8_t row0_psum, row1_psum, row2_psum;
                        if(row < 28){
                            mac0 = perform_mac3(weights_layer_map6[filter][ch][0], slider[ch]);
                            psum1[filter % NUM_PE_LAYER_MAP6][ch].write(mac0);
                        }
                        if(row >= 1 && row < 29) {
                            row1_psum = psum1[filter % NUM_PE_LAYER_MAP6][ch].read();
                            mac1 = perform_mac3(weights_layer_map6[filter][ch][1], slider[ch]);
                            psum2[filter % NUM_PE_LAYER_MAP6][ch].write(row1_psum + mac1);
                        }
                        if(row >= 2){
                            row2_psum = psum2[filter % NUM_PE_LAYER_MAP6][ch].read();
                            mac2 = perform_mac3(weights_layer_map6[filter][ch][2], slider[ch]);
                            fixed_4_8_t pre_activation = row2_psum + mac2;
                            final_sum[filter] += pre_activation;
                        }
                    }

                    if(row >= 2) map_out[filter].write(prelu(conv_map6_prelu[filter], \
                                                            (final_sum[filter] + conv_map6_bias[filter])));
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
//	#pragma HLS BIND_STORAGE variable=tile_in type=bram
	// #pragma HLS array_partition variable=tile_in dim=0 type=complete

	 #pragma HLS DATAFLOW
	prep_tile(in_stream, tile_in);

	conv_feature_extraction0(tile_in, map_extraction);
	conv_shrink0(map_extraction, map_shrink);
	conv_map0(map_shrink, map_map0);
	conv_map2(map_map0, map_map2);
	conv_map4(map_map2, map_map4);
	conv_map6(map_map4, map_map6);
	conv_expand0(map_map6, map_expand0);

	for(int i = 0; i < OUT_CHN_LAYER_EXPAND0; i++){
		printf("INFO [conv2d] Feature map %d:\n", i);
		for (int col = 0; col < 28*28; col++){
			// printf("%.8f \n", map_extraction[i].read().to_float());
			printf("%.8f \n", map_expand0[i].read().to_float());
		}
		printf("\n");
	}
}

