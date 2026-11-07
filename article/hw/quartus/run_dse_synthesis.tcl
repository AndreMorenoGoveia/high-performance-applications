# ==============================================================================
# Script: run_dse_synthesis.tcl
# Target: Intel Quartus Prime (Cyclone V SoC: 5CSEBA6U23I7 DE10-Nano)
# Usage:  quartus_sh -t run_dse_synthesis.tcl
# Description:
#   Automates synthesis, fitting, TimeQuest Timing Analysis (Slow 85C & 0C),
#   and VCD-annotated PowerPlay Power Analysis across P in {8, 16, 32, 64}
#   for both BitMamba-CV (Zero-DSP Projections) and Same-Chip INT8-DSP Baseline.
# ==============================================================================

package require ::quartus::project
package require ::quartus::flow

set project_name "bitmamba_cv"
set top_entity   "bitmamba_axi_top"
set device_part  "5CSEBA6U23I7" ;# Intel Cyclone V SoC (DE10-Nano: 41,910 ALMs, 553 M10Ks, 112 DSPs)

project_new $project_name -overwrite -revision $project_name

set_global_assignment -name FAMILY "Cyclone V"
set_global_assignment -name DEVICE $device_part
set_global_assignment -name TOP_LEVEL_ENTITY $top_entity

# Add RTL source files
set_global_assignment -name VERILOG_FILE ../rtl/radix3_decoder.v
set_global_assignment -name VERILOG_FILE ../rtl/fwht_16_stream.v
set_global_assignment -name VERILOG_FILE ../rtl/bitlinear_pe.v
set_global_assignment -name VERILOG_FILE ../rtl/baseline_int8_pe.v
set_global_assignment -name VERILOG_FILE ../rtl/cpot_selective_scan.v
set_global_assignment -name VERILOG_FILE ../rtl/bitmamba_axi_top.v

# Optimization & synthesis directives
set_global_assignment -name OPTIMIZATION_MODE "HIGH PERFORMANCE EFFORT"
set_global_assignment -name AUTO_RAM_RECOGNITION ON
set_global_assignment -name AUTO_DSP_RECOGNITION ON

# PowerPlay VCD-annotated switching activity setup
set_global_assignment -name POWER_PRESET_COOLING_SOLUTION "23 MM HEAT SINK WITH 200 LFPM AIRFLOW"
set_global_assignment -name POWER_BOARD_THERMAL_MODEL "NONE (CONSERVATIVE)"
set_global_assignment -name POWER_USE_INPUT_FILES ON
set_global_assignment -name POWER_INPUT_FILE_NAME "bitmamba_activity.vcd" -section_id bitmamba_activity.vcd

foreach p_val {8 16 32 64} {
    post_message -type info "=== Running Cyclone V Synthesis DSE for P_LANES = $p_val ==="
    set_parameter -name P_LANES $p_val
    export_assignments
    execute_flow -compile
    execute_module -tool sta
    execute_module -tool pow
}

project_close

