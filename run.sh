#!/bin/bash

RUN_ID=$(python genwandbid.py)
export RUN_ID
echo "RUN_ID set to $RUN_ID"

SCRIPT=$1
echo "Running $SCRIPT"
python $SCRIPT
