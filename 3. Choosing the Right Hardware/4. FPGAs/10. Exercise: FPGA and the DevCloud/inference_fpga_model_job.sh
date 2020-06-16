#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

# COMPLETED: Create DEVICE variable
DEVICE=$1
# COMPLETED: Create MODELPATH variable
MODELPATH=$2

export AOCL_BOARD_PACKAGE_ROOT=/opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/BSP/a10_1150_sg2

source /opt/altera/aocl-pro-rte/aclrte-linux64/init_opencl.sh
aocl program acl0 /opt/intel/openvino/bitstreams/a10_vision_design_sg2_bitstreams/2020-2_PL2_FP16_MobileNet_Clamp.aocx

export CL_CONTEXT_COMPILER_MODE_INTELFPGA=3

# COMPLETED: Call the Python script
python3 inference_on_device.py  --model_path ${MODELPATH} --device ${DEVICE}

cd /output

tar zcvf output.tgz * # compresses all files in the current directory (output)