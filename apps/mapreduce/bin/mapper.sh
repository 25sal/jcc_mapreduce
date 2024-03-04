#!/bin/bash
export PYTHONPATH=/home/spark/apps/mapreduce/gmaps:/home/spark/apps/mapreduce
export PATH=$PATH:/home/spark/.local/bin
python3 /home/spark/apps/mapreduce/bin/mapperhdfs.py
