#!/bin/bash

python3 train.py --num_epochs=100000 --data_dir=data --log_dir=save/deepffm_v0 --batch_size=1000 --num_workers=10 
