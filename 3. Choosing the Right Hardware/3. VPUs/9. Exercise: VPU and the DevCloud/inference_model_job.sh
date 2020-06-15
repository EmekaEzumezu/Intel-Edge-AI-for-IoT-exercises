#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

# COMPLETED: Create DEVICE variable
DEVICE=$1
# COMPLETED: Create MODELPATH variable
MODELPATH=$2
# COMPLETED: Create SAVEPATH variable
SAVEPATH=$3

# COMPLETED: Call the Python script
python3 inference_on_device.py  --model_path ${MODELPATH} --device ${DEVICE} --path ${SAVEPATH}

cd /output

tar zcvf output.tgz * # compresses all files in the current directory (output)