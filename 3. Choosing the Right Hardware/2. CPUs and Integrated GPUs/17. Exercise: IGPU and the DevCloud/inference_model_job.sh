#!/bin/bash

exec 1>/output/stdout.log 2>/output/stderr.log

mkdir -p /output

# COMPLETED: Create DEVICE variable
DEVICE = $1

# COMPLETED: Create BATCHES variable
BATCHES = $2

# COMPLETED: Create MODELPATH variable
MODELPATH = $3

# COMPLETED: Create SAVEPATH variable
SAVEPATH = $4

# COMPLETED: Call the Python script
python3 inference_on_device.py  --model_path ${MODELPATH} --device ${DEVICE} --path ${SAVEPATH} --batches ${BATCHES}

cd /output

tar zcvf output.tgz * # compresses all files in the current directory (output)