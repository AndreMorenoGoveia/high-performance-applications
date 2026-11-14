
read_lef /home/andre/opt/pdk/orfs/flow/platforms/nangate45/lef/NangateOpenCellLibrary.tech.lef
read_lef /home/andre/opt/pdk/orfs/flow/platforms/nangate45/lef/NangateOpenCellLibrary.macro.mod.lef
read_liberty /home/andre/opt/pdk/orfs/flow/platforms/nangate45/lib/NangateOpenCellLibrary_typical.lib
read_verilog /home/andre/repos/high-performance-applications/article/hw/asic/build/scan_mult_q15_s32_g10_T2.5/netlist.v
link_design scan_lane
create_clock -name clk -period 2.5 [get_ports clk]
set io [expr {0.2 * 2.5}]
set_input_delay $io -clock clk [delete_from_list [all_inputs] [get_ports clk]]
set_output_delay $io -clock clk [all_outputs]
set_load 0.005 [all_outputs]
set_driving_cell -lib_cell BUF_X2 -pin Z [delete_from_list [all_inputs] [get_ports clk]]
source /home/andre/opt/pdk/orfs/flow/platforms/nangate45/setRC.tcl
initialize_floorplan -utilization 45 -aspect_ratio 1 -core_space 4 -site FreePDK45_38x28_10R_NP_162NW_34O
source /home/andre/opt/pdk/orfs/flow/platforms/nangate45/make_tracks.tcl
place_pins -hor_layers metal3 -ver_layers metal4
global_placement -density 0.65
estimate_parasitics -placement
repair_design
detailed_placement
clock_tree_synthesis -root_buf CLKBUF_X3 -buf_list CLKBUF_X3 -sink_clustering_enable
set_propagated_clock [all_clocks]
estimate_parasitics -placement
repair_timing -setup
detailed_placement
check_placement
set_routing_layers -signal metal2-metal10 -clock metal2-metal10
global_route -allow_congestion
estimate_parasitics -global_routing
set_power_activity -input -activity 0.125
puts "RES wns [sta::worst_slack -max]"
puts "RES tns [sta::total_negative_slack -max]"
report_checks -path_delay max -fields {slew cap} -digits 3
report_design_area
report_power
exit
