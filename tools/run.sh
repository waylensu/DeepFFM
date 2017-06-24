#!/bin/bash

python3 train.py --num_epochs=100 --lr=0.01 --data_dir=data --log_dir=save/deepffm_v0 --batch_size=1000 --num_workers=2 
