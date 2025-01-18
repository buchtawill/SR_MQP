# Run this script from the vivado directory of SR_MQP

set project_name "kv260_vivado_proj"
set project_dir "./$project_name"
set bitstream_dir "./bitstreams"

open_project "$project_dir/$project_name.xpr"

open_bd_design "$project_dir/$project_name.srcs/sources_1/bd/design_1/design_1.bd"
if {[catch {save_bd_design} result]} {
    puts "Error saving block design: $result"
    exit 1
}
if {[catch {validate_bd_design -force} result]} {
    puts "Error validating block design: $result"
    exit 1
}
if {[catch {save_bd_design} result]} {
    puts "Error saving block design: $result"
    exit 1
}

make_wrapper -files [get_files "$project_dir/$project_name.srcs/sources_1/bd/design_1/design_1.bd"] -top
update_compile_order -fileset sources_1

# Run synthesis
reset_run synth_1
launch_runs synth_1 -jobs 12
if {[catch {wait_on_run synth_1} result]} {
    puts "Error during synthesis: $result"
    exit 1
}

# Run implementation, generate the bitstream
reset_run impl_1
launch_runs impl_1 -to_step write_bitstream -jobs 12
if {[catch {wait_on_run impl_1} result]} {
    puts "Error during implementation: $result"
    exit 1
}

open_run impl_1

# Write the bitstream and export it to ./bitstreams
write_bitstream -force "$bitstream_dir/fpga_image.bit"

close_project
